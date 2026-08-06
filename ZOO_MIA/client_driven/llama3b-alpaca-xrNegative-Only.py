"""
FedMeZO membership inference experiment (Alpaca + Open-Llama-3B sequence classification + PEFT LoRA).
LoRA hyperparameters match the default example in `llm.adapter.args` from federatedscope/llm/README.md
(the same set passed via yaml in `main.py`: r / lora_alpha / lora_dropout).
Standalone run: python federatedscope/llama3b-alpaca-MIA.py
"""
import copy
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mia_roc_plotting import compute_roc_auc_and_maybe_plot


@dataclass
class Config:
    seed: int = 42
    data_path: str = "/home/zhike/JWH/fedmezo-MIA/FedMeZO-main/data/alpaca_data.json"
    model_name: str = "/home/zhike/JWH/model/open_llama_3b_v2/"
    num_labels: int = 4
    max_length: int = 512
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Match LoRA defaults in llm/README.md (mezo_testcase.yaml uses alpha=16, dropout=0.05; adjust as needed)
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    num_clients: int = 2
    fl_samples: int = 2000
    local_steps: int = 10
    batch_size: int = 1
    lr: float = 1e-5
    zo_eps: float = 1e-3

    attack_samples: int = 100
    target_cache_path: str = "outputs/llama3b_alpaca_mia_target_cache.pt"

    round1_server_trials: int = 100
    round1_inject_step: int = 0

    # Binary membership detection: each trial injects the target sample with this probability (otherwise no client injects)
    member_present_prob: float = 0.5
    # det_score mode (consistent with distilbert-agnews-MIA-zhibiao)
    server_det_score_mode: str = "gap"
    # Threshold: online_quantile / roc_youden / fixed
    server_threshold_mode: str = "online_quantile"
    threshold_calib_fraction: float = 0.5
    server_fixed_threshold: float = 0.0
    online_warmup: int = 20
    online_window: int = 40
    online_alpha: float = 0.3
    online_min_threshold: float = 10000
    online_max_threshold: float = 1000000000000
    online_update_with_neg_only: bool = True
    online_auto_clip_by_mode: bool = True
    online_init_with_warmup_low_cluster: bool = True
    roc_plot_path: str = ""
    roc_scores_cache_path: str = "outputs/mia_roc_scores_llama3b_alpaca.npz"
    roc_dataset_label: str = "Alpaca (Llama3B)"

    # Replay online_quantile on a fixed trial stream; sweep window and plot (needs server_threshold_mode=online_quantile)
    neg_only_window_ablation_enable: bool = True
    neg_only_ablation_windows: tuple = (20, 40, 60, 80, 100, 120)
    neg_only_ablation_plot_path: str = "outputs/llama3b_alpaca_neg_only_window_ablation.png"
    # Ablation curve values; empty string defaults to .npz with same stem as neg_only_ablation_plot_path (for replotting via standalone script)
    neg_only_ablation_npz_path: str = ""
    # Left y: mean/last threshold; right y: online presence accuracy only
    neg_only_ablation_y_metric: str = "mean"

    adv_init_use: bool = True
    adv_init_steps: int = 500
    # Large models + high-order gradients are unstable: recommend 1e-4–5e-4; original 1e-3 often causes mid-run spikes and end-step collapse
    adv_init_lr: float = 1e-4
    adv_init_w_target: float = 1.5
    adv_init_w_anchor: float = 0.15
    adv_init_anchor_power: float = 2.0
    # The max term introduces spikes/numerical instability more easily; keep off initially to avoid blow-ups, enable gradually later
    adv_init_w_anchor_max: float = 0.1
    # Clip trainable-parameter gradient L2 norm before each update; 0 disables clipping
    adv_init_clip_grad_norm: float = 0.2
    adv_init_anchors_per_client: int = 50
    # Random anchor subset size per step (<= total anchors) to reduce second-order graph VRAM peaks
    adv_init_anchor_subset_size: int = 4
    # Adversarial init anchor source: "clients"=each client DataLoader (default); "attack_pool"=attack pool attack_batches
    adv_init_anchor_source: str = "attack_pool"
    adv_init_log_every: int = 10
    adv_init_bundle_path: str = "outputs/llama3b_alpaca_mia_adv_init_bundle.pt"
    adv_init_bundle_use: bool = True  # True: skip adversarial init if cache hit
    adv_init_gradient_checkpointing: bool = False


class AlpacaDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_cuda_oom_error(err: Exception) -> bool:
    msg = str(err).lower()
    return ("out of memory" in msg) or ("cuda error" in msg and "memory" in msg)


def load_alpaca_data(cfg):
    total_needed = cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    rng = random.Random(cfg.seed)

    try:
        with open(cfg.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read Alpaca data: {cfg.data_path}, err={e}") from e

    if not isinstance(raw, list) or len(raw) == 0:
        raise RuntimeError(f"Alpaca data format invalid or empty: {cfg.data_path}")

    texts = []
    for item in raw:
        instruction = str(item.get("instruction", "")).strip()
        inp = str(item.get("input", "")).strip()
        output = str(item.get("output", "")).strip()
        text = (
            f"Instruction: {instruction}\n"
            f"Input: {inp}\n"
            f"Output: {output}"
        )
        texts.append(text)

    n = min(total_needed, len(texts))
    if n < total_needed:
        print(f"Warning: Alpaca has only {n} samples, fewer than required {total_needed}.")
    idxs = list(range(len(texts)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    texts = [texts[i] for i in idxs]
    return texts


def get_model_and_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        use_fast=False,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # bf16 is usually more stable than fp16 (especially for adversarial init with high-order gradients/second-order graphs).
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = model.to(cfg.device)
    model.config.pad_token_id = tokenizer.pad_token_id

    target_modules = list(cfg.lora_target_modules)
    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameter count (LoRA+score): {trainable_params}")
    return model, tokenizer


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def flatten_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params]).float()


def assign_params(params, vec):
    offset = 0
    with torch.no_grad():
        for p in params:
            numel = p.numel()
            p.copy_(vec[offset : offset + numel].view_as(p))
            offset += numel


def compute_loss(model, batch):
    dev = next(model.parameters()).device
    outputs = model(
        input_ids=batch["input_ids"].to(dev),
        attention_mask=batch["attention_mask"].to(dev),
        labels=batch["labels"].to(dev),
    )
    return outputs.loss


def mezo_step(model, batch, params, theta, z, cfg):
    theta_orig = theta.clone()
    # Zero-order diff only needs scalar loss; do not build autograd graphs for both forwards; otherwise two Llama forward graphs are linked by (loss_pos-loss_neg) and VRAM fills quickly.
    with torch.no_grad():
        assign_params(params, theta_orig + cfg.zo_eps * z)
        loss_pos = compute_loss(model, batch)
        assign_params(params, theta_orig - cfg.zo_eps * z)
        loss_neg = compute_loss(model, batch)
        assign_params(params, theta_orig)
    g_dir = (loss_pos - loss_neg) / (2 * cfg.zo_eps)
    g_est = g_dir * z
    theta_new = theta_orig - cfg.lr * g_est
    return theta_new, g_est


def _grad_norm_classifier(model, batch, create_graph=True):
    params = get_trainable_params(model)
    dev = next(model.parameters()).device
    if not params:
        t = torch.zeros((), device=dev)
        return t.requires_grad_(True) if create_graph else t
    loss = compute_loss(model, batch)
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, allow_unused=True)
    parts = [g.reshape(-1) for g in grads if g is not None]
    if not parts:
        t = torch.zeros((), device=dev)
        return t.requires_grad_(True) if create_graph else t
    flat = torch.cat(parts)
    return flat.norm()


def collect_anchor_batches(client_loaders, target_batch, cfg, seed):
    loaders = list(client_loaders)
    random.Random(seed).shuffle(loaders)
    anchors = []
    tid = None
    if target_batch is not None:
        tid = target_batch["input_ids"].detach().cpu()
    for loader in loaders:
        data_iter = iter(loader)
        count = 0
        tries = 0
        while count < cfg.adv_init_anchors_per_client and tries < 2000:
            tries += 1
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
            b = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
            }
            if tid is not None and b["input_ids"].shape == tid.shape:
                if torch.equal(b["input_ids"].cpu(), tid):
                    continue
            anchors.append(b)
            count += 1
    return anchors


def collect_anchor_batches_from_attack_pool(attack_batches, target_batch, cfg, seed):
    """
    Sample anchor batches from the attack pool list (skip items with the same input_ids as target).
    Total count matches client mode: num_clients * adv_init_anchors_per_client.
    """
    rng = random.Random(seed)
    tid = target_batch["input_ids"].detach().cpu() if target_batch is not None else None
    candidates = []
    for b in attack_batches:
        bdict = {
            "input_ids": b["input_ids"],
            "attention_mask": b["attention_mask"],
            "labels": b["labels"],
        }
        if tid is not None and bdict["input_ids"].shape == tid.shape:
            if torch.equal(bdict["input_ids"].cpu(), tid):
                continue
        candidates.append(bdict)

    total_want = cfg.num_clients * cfg.adv_init_anchors_per_client
    if not candidates:
        print("[adv-init] Attack-pool anchors: no available candidates (pool may contain only the target).")
        return []

    rng.shuffle(candidates)
    if len(candidates) >= total_want:
        return candidates[:total_want]

    out = []
    while len(out) < total_want:
        rng.shuffle(candidates)
        for c in candidates:
            out.append(c)
            if len(out) >= total_want:
                break
    return out


def _set_adversarial_grad_checkpointing(model, enable: bool) -> bool:
    """Try to enable/disable gradient checkpointing on PEFT-wrapped Llama; returns True on success."""
    try:
        if enable:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            if hasattr(base, "gradient_checkpointing_enable"):
                base.gradient_checkpointing_enable()
                return True
            mid = getattr(base, "model", None)
            if mid is not None and hasattr(mid, "gradient_checkpointing_enable"):
                mid.gradient_checkpointing_enable()
                return True
        else:
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            if hasattr(base, "gradient_checkpointing_disable"):
                base.gradient_checkpointing_disable()
                return True
            mid = getattr(base, "model", None)
            if mid is not None and hasattr(mid, "gradient_checkpointing_disable"):
                mid.gradient_checkpointing_disable()
                return True
    except Exception as e:
        print(f"[adv-init] gradient checkpointing toggle failed (can ignore): {e}")
    return False


def _adv_init_loss_terms(model, target_batch, anchor_batches, cfg):
    was_training = model.training
    model.eval()
    try:
        n_t = _grad_norm_classifier(model, target_batch, create_graph=False).item()
        n_div = max(1, len(anchor_batches))
        p = float(cfg.adv_init_anchor_power)
        if anchor_batches:
            norms = [
                _grad_norm_classifier(model, ab, create_graph=False).item() for ab in anchor_batches
            ]
            mean_anchor = float(np.mean(norms))
            max_anchor = float(max(norms))
            pen = cfg.adv_init_w_anchor / n_div * sum(x**p for x in norms)
            if cfg.adv_init_w_anchor_max > 0:
                pen += cfg.adv_init_w_anchor_max * max_anchor
        else:
            mean_anchor = 0.0
            max_anchor = 0.0
            pen = 0.0
        L = -cfg.adv_init_w_target * n_t + pen
    finally:
        model.train(was_training)
    return n_t, mean_anchor, max_anchor, L


def adversarial_sharpness_init(model, target_batch, anchor_batches, cfg):
    if target_batch is None:
        print("[adv-init] No target sample, skipping.")
        return
    params = get_trainable_params(model)
    if not params:
        print("[adv-init] No trainable parameters, skipping.")
        return

    gc_on = False
    if getattr(cfg, "adv_init_gradient_checkpointing", False):
        gc_on = _set_adversarial_grad_checkpointing(model, True)
        if gc_on:
            print("[adv-init] Enabled gradient checkpointing (adversarial init phase only).")

    model.train()
    opt = torch.optim.Adam(params, lr=cfg.adv_init_lr)
    n_anchor = max(1, len(anchor_batches))
    subset_k = max(
        1,
        min(int(getattr(cfg, "adv_init_anchor_subset_size", n_anchor)), n_anchor),
    )
    if subset_k < len(anchor_batches):
        print(
            f"[adv-init] Using random subset of {subset_k}/{len(anchor_batches)} anchors per step in loss."
        )
    model.zero_grad(set_to_none=True)

    n_t0, mean_a0, max_a0, L0 = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
    print(
        f"[adv-init] Start step 0/{cfg.adv_init_steps}: "
        f"||∇L_target||={n_t0:.6f}, anchor mean||∇L||={mean_a0:.6f}, max||∇L||={max_a0:.6f}, "
        f"L={L0:.6f} (=-w_t||g_tgt||+(w_a/n)Σ||g||^{cfg.adv_init_anchor_power:g}"
        f"+{'w_max·max||g||' if cfg.adv_init_w_anchor_max > 0 else '0'}) "
        f"(num anchors={len(anchor_batches)})"
    )

    log_every = max(0, int(cfg.adv_init_log_every))
    anchor_power = float(cfg.adv_init_anchor_power)
    nan_happened = False
    for step in tqdm(range(cfg.adv_init_steps), desc="Adversarial init (LoRA+score)"):
        opt.zero_grad(set_to_none=True)

        n_t = _grad_norm_classifier(model, target_batch, create_graph=True)
        if not torch.isfinite(n_t).item():
            nan_happened = True
            print(f"[adv-init] Non-finite n_t detected: {n_t}, stopping early at step={step+1}.")
            break
        # Random anchor subset per step, single scalar loss, one backward
        loss = -cfg.adv_init_w_target * n_t
        anchor_norms = []
        if anchor_batches:
            idxs = list(range(len(anchor_batches)))
            random.Random(cfg.seed + 10007 + step * 17).shuffle(idxs)
            use_idx = idxs[:subset_k]
            for j in use_idx:
                ab = anchor_batches[j]
                na = _grad_norm_classifier(model, ab, create_graph=True)
                if not torch.isfinite(na).item():
                    nan_happened = True
                    print(f"[adv-init] Non-finite anchor na detected: {na}, stopping early at step={step+1}.")
                    break
                anchor_norms.append(na)
                loss = loss + (cfg.adv_init_w_anchor / max(1, subset_k)) * na.pow(anchor_power)
            if nan_happened:
                break
            if cfg.adv_init_w_anchor_max > 0 and anchor_norms:
                loss = loss + cfg.adv_init_w_anchor_max * torch.stack(anchor_norms).max()

        if not torch.isfinite(loss).item():
            nan_happened = True
            print(f"[adv-init] Non-finite loss detected: {loss}, stopping early at step={step+1}.")
            break
        loss.backward()

        clip = float(getattr(cfg, "adv_init_clip_grad_norm", 0.0) or 0.0)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)

        # Check whether gradients became NaN/Inf (extra cost only when an anomaly occurs)
        for prm in params:
            if prm.grad is not None and not torch.isfinite(prm.grad).all():
                nan_happened = True
                print("[adv-init] NaN/Inf in parameter gradients detected, stopping early.")
                break
        if nan_happened:
            break
        opt.step()

        done = step + 1
        if log_every > 0 and (done % log_every == 0 or done == cfg.adv_init_steps):
            nt, ma, mxa, Lm = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
            print(
                f"[adv-init]  step {done:4d}/{cfg.adv_init_steps}: "
                f"||∇L_target||={nt:.6f}, anchor mean||∇L||={ma:.6f}, max||∇L||={mxa:.6f}, L={Lm:.6f}"
            )

    if gc_on:
        _set_adversarial_grad_checkpointing(model, False)

    if nan_happened:
        print("[adv-init] NaN/Inf detected, stopping adversarial init early (keeping current parameters).")
        return

    model.eval()
    model.zero_grad(set_to_none=True)
    n_tf, mean_af, max_af, Lf = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
    rel = (n_tf - n_t0) / (n_t0 + 1e-12) * 100.0
    print(
        f"[adv-init] Done: ||∇L_target|| {n_t0:.6f} -> {n_tf:.6f} "
        f"(relative {rel:+.2f}%), anchor mean||∇L|| {mean_a0:.6f} -> {mean_af:.6f}, "
        f"max||∇L|| {max_a0:.6f} -> {max_af:.6f}, L {L0:.6f} -> {Lf:.6f}"
    )
    if n_tf < 0.05 * max(n_t0, 1e-8):
        print(
            "[adv-init] Warning: final ||∇L_target|| is still much lower than start; try reducing adv_init_lr, "
            "tightening adv_init_clip_grad_norm, or temporarily setting adv_init_w_anchor_max to 0."
        )


def client_update_with_inject_step(model, dataloader, target_batch, cfg, seed, inject_at_step=None):
    params = get_trainable_params(model)
    theta = flatten_params(params)
    g_sum = torch.zeros_like(theta)

    if theta.is_cuda:
        gen = torch.Generator(device=theta.device)
        gen.manual_seed(seed)
    else:
        gen = torch.Generator().manual_seed(seed)

    data_iter = iter(dataloader)
    for step in range(cfg.local_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if inject_at_step is not None and step == inject_at_step and target_batch is not None:
            batch = target_batch

        z = torch.randn(theta.shape, generator=gen, dtype=theta.dtype, device=theta.device)
        theta, g_est = mezo_step(model, batch, params, theta, z, cfg)
        g_sum += g_est.detach()

    assign_params(params, theta)
    return g_sum


def server_pick_member_by_grad_norm(g_sums):
    scores = [float(g.float().norm().item()) for g in g_sums]
    return int(np.argmax(scores)), scores


def compute_det_score(scores, mode: str):
    """Given per-client gradient-norm scores, compute detection score for membership presence (consistent with zhibiao)."""
    if not scores:
        return 0.0
    x = np.asarray(scores, dtype=np.float64)
    x = np.maximum(x, 0.0)
    s_sorted = np.sort(x)
    n = len(s_sorted)
    s_max = float(s_sorted[-1])
    s_second = float(s_sorted[-2]) if n >= 2 else 0.0
    eps = 1e-12
    if mode == "gap":
        return s_max - s_second
    if mode == "ratio":
        return s_max / (float(np.mean(s_sorted)) + eps)
    if mode == "max":
        return s_max
    if mode == "norm_mean":
        return float(np.mean(s_sorted))
    if mode == "zmax":
        mu = float(np.mean(s_sorted))
        std = float(np.std(s_sorted))
        return (s_max - mu) / (std + eps)
    if mode == "top1_top2":
        return s_max / (s_second + eps)
    if mode == "iqr_outlier":
        q1 = float(np.quantile(s_sorted, 0.25))
        q3 = float(np.quantile(s_sorted, 0.75))
        iqr = q3 - q1
        return (s_max - q3) / (iqr + eps)
    if mode == "gini":
        s = float(np.sum(s_sorted))
        if s <= eps:
            return 0.0
        i = np.arange(1, n + 1, dtype=np.float64)
        g = (2.0 * float(np.sum(i * s_sorted)) / (n * s)) - (n + 1.0) / n
        return float(max(0.0, g))
    if mode == "entropy":
        s = float(np.sum(s_sorted))
        if s <= eps:
            return 0.0
        p = s_sorted / s
        ent = -float(np.sum(p * np.log(p + eps)))
        if n <= 1:
            return 0.0
        return ent / float(np.log(n))
    raise ValueError(f"Unknown server_det_score_mode: {mode!r}")


def get_online_clip_bounds(cfg, mode: str):
    min_thr = float(cfg.online_min_threshold)
    max_thr = float(cfg.online_max_threshold)
    if bool(getattr(cfg, "online_auto_clip_by_mode", False)):
        if mode in ("ratio", "top1_top2"):
            min_thr = min(min_thr, 1.0)
            max_thr = min(max_thr, 1e3)
        elif mode in ("gini", "entropy"):
            min_thr = 0.0
            max_thr = 1.0
    if max_thr < min_thr:
        max_thr = min_thr
    return min_thr, max_thr


def warmup_low_cluster_max_threshold(scores, fallback: float):
    """
    Split warmup scores into two clusters by largest gap after sorting; return the max of the smaller cluster as initial threshold.
    Fall back if too few samples or split fails.
    """
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return float(fallback), 0, 0, 0.0
    x = np.sort(arr)
    gaps = np.diff(x)
    if gaps.size < 1:
        return float(fallback), 0, 0, 0.0
    split = int(np.argmax(gaps)) + 1
    if split <= 0 or split >= x.size:
        return float(fallback), 0, 0, 0.0
    low = x[:split]
    high = x[split:]
    if low.size < 1 or high.size < 1:
        return float(fallback), 0, 0, 0.0
    return float(low[-1]), int(low.size), int(high.size), float(gaps[split - 1])


def replay_online_quantile_thresholds(test_records, cfg, *, window: int, neg_only: bool):
    """
    Replay the online_quantile decision loop (same as main); only `window` and neg-only history
    update differ. Used to ablate buffer window vs. neg_only on a fixed det_score sequence.
    """
    online_thresholds = []
    online_history = []
    warmup = int(max(0, cfg.online_warmup))
    window = int(max(1, window))
    alpha = float(np.clip(cfg.online_alpha, 1e-6, 1.0 - 1e-6))
    q = 1.0 - alpha
    min_thr, max_thr = get_online_clip_bounds(cfg, cfg.server_det_score_mode)
    online_init_threshold = None
    eval_steps = []

    for r in test_records:
        score = float(r["det_score"])
        if len(online_history) < warmup:
            online_history.append(score)
            continue
        if online_init_threshold is None:
            hist_init = list(online_history)
            fallback_thr = float(np.quantile(np.asarray(hist_init, dtype=np.float64), q))
            if bool(getattr(cfg, "online_init_with_warmup_low_cluster", True)):
                init_thr, _, _, _ = warmup_low_cluster_max_threshold(hist_init, fallback=fallback_thr)
            else:
                init_thr = fallback_thr
            online_init_threshold = min(max(float(init_thr), min_thr), max_thr)
            dynamic_thr = online_init_threshold
        else:
            hist = online_history[-window:]
            dynamic_thr = float(np.quantile(np.asarray(hist, dtype=np.float64), q))
            dynamic_thr = min(max(dynamic_thr, min_thr), max_thr)
        pred_present = score > dynamic_thr
        online_thresholds.append(dynamic_thr)
        eval_steps.append(
            {
                "member_present": bool(r["member_present"]),
                "det_score": score,
                "dynamic_thr": float(dynamic_thr),
                "pred_present": bool(pred_present),
            }
        )
        if (not neg_only) or (not pred_present):
            online_history.append(score)

    thr_last = float(online_thresholds[-1]) if online_thresholds else float("nan")
    thr_mean = float(np.mean(online_thresholds)) if online_thresholds else float("nan")
    if eval_steps:
        presence_acc = float(
            np.mean([s["pred_present"] == s["member_present"] for s in eval_steps])
        )
    else:
        presence_acc = float("nan")
    return {
        "thr_last": thr_last,
        "thr_mean": thr_mean,
        "thresholds": online_thresholds,
        "eval_steps": eval_steps,
        "presence_acc": presence_acc,
    }


def plot_neg_only_window_ablation(
    windows,
    thr_neg_only_on,
    thr_neg_only_off,
    acc_neg_only_on,
    acc_neg_only_off,
    out_path: str,
    y_metric: str,
):
    """Dual y-axis: left = threshold; right = online presence accuracy."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Ablation] matplotlib not installed; skip neg_only window plot.")
        return False

    y_metric = (y_metric or "mean").strip().lower()
    if y_metric not in ("mean", "last"):
        y_metric = "mean"

    c_on, c_off = "#2ca02c", "#d62728"
    xlab = "History window size"
    ylab_left = "Threshold" if y_metric == "mean" else "Last dynamic threshold"
    ylab_right = "ASR"
    

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax2 = ax.twinx()

    # Solid = threshold; markers o/s distinguish strategy (Neg-only / Full).
    ax.plot(
        windows,
        thr_neg_only_on,
        linestyle="-",
        marker="o",
        linewidth=2,
        markersize=7,
        color=c_on,
        label="Threshold | Neg-only",
    )
    ax.plot(
        windows,
        thr_neg_only_off,
        linestyle="-",
        marker="s",
        linewidth=2,
        markersize=7,
        color=c_off,
        label="Threshold | Full update",
    )
    # Dashed = online accuracy; looser dash pattern + long legend handles so dashes read in legend.
    dash_pattern = (0, (6, 4))
    ax2.plot(
        windows,
        acc_neg_only_on,
        linestyle=dash_pattern,
        marker="^",
        linewidth=2,
        markersize=7,
        color=c_on,
        alpha=0.95,
        label="Accuracy | Neg-only",
    )
    ax2.plot(
        windows,
        acc_neg_only_off,
        linestyle=dash_pattern,
        marker="v",
        linewidth=2,
        markersize=7,
        color=c_off,
        alpha=0.95,
        label="Accuracy | Full update",
    )

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab_left, color="#333333")
    ax2.set_ylabel(ylab_right, color="#555555")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="y", labelcolor="#333333")
    ax2.tick_params(axis="y", labelcolor="#555555")
    ax2.set_ylim(0.0, 1.0)

    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        lab1 + lab2,
        loc="lower right",
        fontsize=7.5,
        framealpha=0.92,
        handlelength=8,
        handletextpad=0.55,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[Ablation] Saved plot: {out_path}")
    return True


def save_neg_only_window_ablation_npz(
    out_path: str,
    windows: list,
    thr_neg_only_on: list,
    thr_neg_only_off: list,
    acc_neg_only_on: list,
    acc_neg_only_off: list,
    y_metric: str,
) -> bool:
    """Save neg-only window ablation data for replotting via plot_neg_only_window_ablation_from_npz.py."""
    out_path = (out_path or "").strip()
    if not out_path:
        return False
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        windows=np.asarray(windows, dtype=np.int32),
        thr_neg_only_on=np.asarray(thr_neg_only_on, dtype=np.float64),
        thr_neg_only_off=np.asarray(thr_neg_only_off, dtype=np.float64),
        acc_neg_only_on=np.asarray(acc_neg_only_on, dtype=np.float64),
        acc_neg_only_off=np.asarray(acc_neg_only_off, dtype=np.float64),
        y_metric=np.array([str(y_metric)], dtype=object),
    )
    print(f"[Ablation] Saved data: {out_path}")
    return True


def simulate_round1_trial_record(
    model,
    client_loaders,
    injected_target_batch,
    cfg,
    trial_seed: int,
    member_present: bool,
    true_member: Optional[int],
    inject_step: int,
):
    member_present = bool(member_present and (injected_target_batch is not None))
    if not member_present:
        true_member = None

    theta0 = copy.deepcopy(model)
    g_uploads = []
    for cid in range(cfg.num_clients):
        mloc = copy.deepcopy(theta0)
        this_inject_step = inject_step if (member_present and cid == true_member) else None
        g_sum = client_update_with_inject_step(
            mloc,
            client_loaders[cid],
            injected_target_batch,
            cfg,
            seed=trial_seed + cid * 100,
            inject_at_step=this_inject_step,
        )
        g_uploads.append(g_sum)

    scores = [float(g.float().norm().item()) for g in g_uploads]
    det_score = compute_det_score(scores, cfg.server_det_score_mode)
    pred_argmax = int(np.argmax(scores))
    return {
        "member_present": member_present,
        "true_member": true_member,
        "scores": scores,
        "det_score": float(det_score),
        "argmax_idx": pred_argmax,
    }

def compute_true_gradient(model, batch):
    """Only trainable parameters connected to loss may have gradients; under PEFT some leaves may be None, so allow_unused is required."""
    params = get_trainable_params(model)
    if not params:
        dev = next(model.parameters()).device
        return torch.zeros(0, device=dev, dtype=torch.float32)
    loss = compute_loss(model, batch)
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
    parts = [g.detach().reshape(-1).float() for g in grads if g is not None]
    if not parts:
        dev = params[0].device
        return torch.zeros(0, device=dev, dtype=torch.float32)
    return torch.cat(parts)


def select_target_sample(model, samples, cfg):
    if len(samples) == 0:
        print("Warning: attack pool is empty, cannot select target sample.")
        return None

    best_norm = -1.0
    best_batch = None
    for i in tqdm(range(len(samples)), desc="Select target sample"):
        batch = samples[i]
        g = compute_true_gradient(model, batch)
        norm = g.norm().item()
        if norm > best_norm:
            best_norm = norm
            best_batch = batch

    print(f"Target sample gradient norm: {best_norm:.6f}")
    return best_batch


def create_dataloader(texts, tokenizer, cfg, shuffle=True):
    dataset = AlpacaDataset(texts, tokenizer, cfg.max_length)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=True)


def _adv_init_meta(cfg):
    if not cfg.adv_init_use:
        return {"use": False}
    return {
        "use": True,
        "steps": cfg.adv_init_steps,
        "lr": cfg.adv_init_lr,
        "w_target": cfg.adv_init_w_target,
        "w_anchor": cfg.adv_init_w_anchor,
        "anchor_power": cfg.adv_init_anchor_power,
        "w_anchor_max": cfg.adv_init_w_anchor_max,
        "anchors_per_client": cfg.adv_init_anchors_per_client,
        "anchor_subset_size": cfg.adv_init_anchor_subset_size,
        "anchor_source": cfg.adv_init_anchor_source,
        "gradient_checkpointing": cfg.adv_init_gradient_checkpointing,
        "clip_grad_norm": cfg.adv_init_clip_grad_norm,
    }


def _build_step45_meta(cfg, attack_texts):
    attack_sig = hashlib.sha1(
        json.dumps(
            {"texts": attack_texts},
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "lora_target_modules": list(cfg.lora_target_modules),
        "num_clients": cfg.num_clients,
        "fl_samples": cfg.fl_samples,
        "attack_samples": cfg.attack_samples,
        "local_steps": cfg.local_steps,
        "grad_stat_mode": "sum",
        "server_attack": "round1_g_sum_l2",
        "zo_eps": cfg.zo_eps,
        "lr": cfg.lr,
        "attack_sig": attack_sig,
        "adv_init": _adv_init_meta(cfg),
    }


def _target_batch_match(a, b):
    if a is None or b is None:
        return a is None and b is None
    try:
        return torch.equal(a["input_ids"].cpu(), b["input_ids"].cpu()) and torch.equal(
            a["labels"].cpu(), b["labels"].cpu()
        )
    except Exception:
        return False


def try_load_adv_init_bundle(model, cfg, cache_meta_45, target_batch):
    if not (cfg.adv_init_use and cfg.adv_init_bundle_use):
        return False
    path = cfg.adv_init_bundle_path
    if not os.path.isfile(path):
        return False
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[adv-init cache] Read failed, will retrain: {e}")
        return False
    if not isinstance(obj, dict) or obj.get("meta") != cache_meta_45:
        print("[adv-init cache] meta mismatch, will retrain.")
        return False
    tb = obj.get("target_batch")
    if not _target_batch_match(tb, target_batch):
        print("[adv-init cache] target sample differs from current, will retrain.")
        return False
    sd = obj.get("state_dict")
    if not isinstance(sd, dict):
        return False
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        if _is_cuda_oom_error(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[adv-init cache] OOM when loading into model, retraining/skipping instead: {e}")
            return False
        raise
    print(f"[adv-init cache] Loaded model and target: {path}")
    return True


def save_adv_init_bundle(model, cfg, cache_meta_45, target_batch):
    os.makedirs(os.path.dirname(cfg.adv_init_bundle_path) or ".", exist_ok=True)
    torch.save(
        {
            "meta": cache_meta_45,
            "state_dict": model.state_dict(),
            "target_batch": target_batch,
        },
        cfg.adv_init_bundle_path,
    )
    print(f"[adv-init cache] Saved: {cfg.adv_init_bundle_path}")

def main():
    cfg = Config()
    set_seed(cfg.seed)
    print(f"Device: {cfg.device}")
    print(f"Config: {cfg}")

    print("\n[1] Loading data...")
    all_texts = load_alpaca_data(cfg)
    fl_texts = all_texts[: cfg.fl_samples * cfg.num_clients]
    attack_texts = all_texts[
        cfg.fl_samples * cfg.num_clients : cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    ]

    client_data = []
    per_client = cfg.fl_samples
    for i in range(cfg.num_clients):
        start = i * per_client
        end = (i + 1) * per_client
        client_data.append(fl_texts[start:end])

    print(f"Num clients: {cfg.num_clients}, samples per client: {per_client}")
    print(f"Attack pool samples: {cfg.attack_samples}")

    print("\n[2] Loading model...")
    model, tokenizer = get_model_and_tokenizer(cfg)

    print("\n[3] Creating data loaders...")
    client_loaders = []
    for texts in client_data:
        client_loaders.append(create_dataloader(texts, tokenizer, cfg))

    attack_batches = []
    for i in range(len(attack_texts)):
        enc = tokenizer(
            attack_texts[i],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        attack_batches.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }
        )

    cache_meta_45 = _build_step45_meta(cfg, attack_texts)



    print("\n[4] Selecting target sample...")
    target_batch = None
    if os.path.exists(cfg.target_cache_path):
        try:
            target_obj = torch.load(cfg.target_cache_path, map_location="cpu")
            if isinstance(target_obj, dict) and target_obj.get("meta") == cache_meta_45:
                target_batch = target_obj.get("target_batch")
                print(f"Hit target sample cache: {cfg.target_cache_path}")
        except Exception as e:
            print(f"Failed to read target sample cache, will recompute: {e}")
    if target_batch is None:
        target_batch = select_target_sample(model, attack_batches, cfg)
        os.makedirs(os.path.dirname(cfg.target_cache_path) or ".", exist_ok=True)
        torch.save({"meta": cache_meta_45, "target_batch": target_batch}, cfg.target_cache_path)
        print(f"Saved target sample cache: {cfg.target_cache_path}")

    if cfg.adv_init_use:
        loaded = try_load_adv_init_bundle(model, cfg, cache_meta_45, target_batch)
        if not loaded:
            print("\n[4.6] Adversarial init (LoRA + CausalLM, amplify target gradient norm)...")
            src = (cfg.adv_init_anchor_source or "clients").strip().lower()
            if src == "attack_pool":
                anchor_batches = collect_anchor_batches_from_attack_pool(
                    attack_batches, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(f"[adv-init] Anchor source: attack pool ({len(anchor_batches)} batches total)")
            elif src == "clients":
                anchor_batches = collect_anchor_batches(
                    client_loaders, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(f"[adv-init] Anchor source: per-client training data ({len(anchor_batches)} batches total)")
            else:
                print(f"Warning: adv_init_anchor_source={cfg.adv_init_anchor_source!r} invalid, falling back to clients.")
                anchor_batches = collect_anchor_batches(
                    client_loaders, target_batch, cfg, seed=cfg.seed + 7001
                )
            adv_init_ok = True
            try:
                adversarial_sharpness_init(model, target_batch, anchor_batches, cfg)
            except Exception as e:
                if _is_cuda_oom_error(e):
                    adv_init_ok = False
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"[adv-init] OOM, skipping adversarial init and continuing: {e}")
                else:
                    raise
            if adv_init_ok:
                save_adv_init_bundle(model, cfg, cache_meta_45, target_batch)

    print(
        "\n[5] Round-1 federated simulation: probabilistic member injection + det_score + ROC-Youden threshold; metrics ASR/TPR/FPR/AUC…"
    )
    r1_rng = random.Random(cfg.seed + 1009)
    presence_acc = float("nan")
    roc_auc = float("nan")
    if cfg.local_steps < 1:
        print("Skipping [5]: local_steps must be >= 1.")
        r1_trials_eff = 0
    else:
        r1_trials_eff = cfg.round1_server_trials
        max_s = cfg.local_steps - 1
        eff_inject = int(np.clip(cfg.round1_inject_step, 0, max_s))
        if cfg.round1_inject_step != eff_inject:
            print(f"Warning: round1_inject_step={cfg.round1_inject_step} clamped to [0,{max_s}]: {eff_inject}")

        trial_records = []
        for trial in range(cfg.round1_server_trials):
            member_present = r1_rng.random() < cfg.member_present_prob
            true_member = r1_rng.randrange(cfg.num_clients) if member_present else None
            rec = simulate_round1_trial_record(
                model,
                client_loaders,
                target_batch,
                cfg,
                trial_seed=cfg.seed + 50000 + trial * 1000,
                member_present=member_present,
                true_member=true_member,
                inject_step=eff_inject,
            )
            trial_records.append(rec)
            print(
                f"Round1 trial {trial+1}: true_present={rec['member_present']}, "
                f"true_member={rec['true_member']}, det_score={rec['det_score']:.4f}, "
                f"||g_sum||={[round(s, 4) for s in rec['scores']]}"
            )

        n_tri = len(trial_records)
        calib_end = int(max(1, min(n_tri - 1 if n_tri > 1 else 1, n_tri * cfg.threshold_calib_fraction)))
        calib_records = trial_records[:calib_end]
        mode = (cfg.server_threshold_mode or "online_quantile").strip().lower()

        threshold = float(cfg.server_fixed_threshold)
        if mode == "online_quantile":
            test_records = trial_records
            clip_min, clip_max = get_online_clip_bounds(cfg, cfg.server_det_score_mode)
            print(
                f"[5] Online dynamic threshold: mode=online_quantile, warmup={cfg.online_warmup}, "
                f"window={cfg.online_window}, alpha={cfg.online_alpha}, "
                f"clip=[{clip_min}, {clip_max}], neg_only_update={cfg.online_update_with_neg_only}"
            )
            if bool(getattr(cfg, "neg_only_window_ablation_enable", False)):
                w_list = [int(w) for w in getattr(cfg, "neg_only_ablation_windows", (10, 20, 40))]
                ykey = str(getattr(cfg, "neg_only_ablation_y_metric", "mean") or "mean").strip().lower()
                if ykey not in ("mean", "last"):
                    ykey = "mean"
                thr_field = f"thr_{ykey}"  # keys thr_mean / thr_last
                thr_on, thr_off = [], []
                acc_on, acc_off = [], []
                for w in w_list:
                    out_on = replay_online_quantile_thresholds(
                        trial_records, cfg, window=w, neg_only=True
                    )
                    out_off = replay_online_quantile_thresholds(
                        trial_records, cfg, window=w, neg_only=False
                    )
                    thr_on.append(float(out_on[thr_field]))
                    thr_off.append(float(out_off[thr_field]))
                    acc_on.append(float(out_on["presence_acc"]))
                    acc_off.append(float(out_off["presence_acc"]))
                    a_on, a_off = out_on["presence_acc"], out_off["presence_acc"]
                    sa_on = f"{a_on:.4f}" if np.isfinite(a_on) else "nan"
                    sa_off = f"{a_off:.4f}" if np.isfinite(a_off) else "nan"
                    print(
                        f"[Ablation][window={w}] neg_only=True  thr_{ykey}={out_on[thr_field]:.6f} "
                        f"acc={sa_on} | neg_only=False thr_{ykey}={out_off[thr_field]:.6f} acc={sa_off}"
                    )
                plot_path = str(getattr(cfg, "neg_only_ablation_plot_path", "outputs/ablation.png"))
                npz_path = (getattr(cfg, "neg_only_ablation_npz_path", None) or "").strip()
                if not npz_path:
                    root, _ = os.path.splitext(plot_path)
                    npz_path = f"{root}.npz" if root else ""
                if npz_path:
                    save_neg_only_window_ablation_npz(
                        npz_path,
                        w_list,
                        thr_on,
                        thr_off,
                        acc_on,
                        acc_off,
                        ykey,
                    )
                plot_neg_only_window_ablation(
                    w_list,
                    thr_on,
                    thr_off,
                    acc_on,
                    acc_off,
                    plot_path,
                    ykey,
                )
        else:
            test_records = trial_records[calib_end:]
            if not test_records:
                print("[5] Warning: no test trials left after calibration; evaluating on all trials (calibration/test overlap, metrics may be optimistic).")
                test_records = list(trial_records)
                calib_records = trial_records
            if mode == "fixed":
                print(f"[5] Using fixed threshold: threshold={threshold}")
            elif mode == "roc_youden":
                y_true_calib = np.array([1 if r["member_present"] else 0 for r in calib_records], dtype=np.int32)
                y_score_calib = np.array([r["det_score"] for r in calib_records], dtype=np.float64)
                if len(np.unique(y_true_calib)) < 2:
                    print(f"[5] ROC-Youden calibration failed (calib has single class label), falling back to fixed threshold={threshold}")
                else:
                    try:
                        from sklearn.metrics import roc_curve
                    except ImportError:
                        print("[5] ROC-Youden calibration failed (sklearn not installed), falling back to fixed threshold")
                    else:
                        fpr_calib, tpr_calib, thr_calib = roc_curve(y_true_calib, y_score_calib)
                        valid = np.isfinite(thr_calib)
                        if not np.any(valid):
                            print("[5] ROC-Youden calibration failed (threshold non-finite), falling back to fixed threshold")
                        else:
                            youden = tpr_calib[valid] - fpr_calib[valid]
                            best_idx_local = int(np.argmax(youden))
                            threshold = float(thr_calib[valid][best_idx_local])
                            best_j = float(youden[best_idx_local])
                            best_tpr = float(tpr_calib[valid][best_idx_local])
                            best_fpr = float(fpr_calib[valid][best_idx_local])
                            print(
                                f"[5] ROC-Youden calibration (calib n={len(calib_records)}): "
                                f"best_J={best_j:.6f} (TPR={best_tpr:.4f}, FPR={best_fpr:.4f}) "
                                f"=> threshold={threshold:.6f}"
                            )
            else:
                raise ValueError(f"Unknown server_threshold_mode: {cfg.server_threshold_mode!r}")

        TP = TN = FP = FN = 0
        member_idx_correct = 0
        member_pos_trials = 0
        online_thresholds = []
        online_history = []
        warmup = int(max(0, cfg.online_warmup))
        window = int(max(1, cfg.online_window))
        alpha = float(np.clip(cfg.online_alpha, 1e-6, 1.0 - 1e-6))
        q = 1.0 - alpha
        min_thr, max_thr = get_online_clip_bounds(cfg, cfg.server_det_score_mode)
        evaluated_records = []
        online_hist_updates = 0
        online_init_threshold = None

        for idx, r in enumerate(test_records):
            if mode == "online_quantile":
                score = float(r["det_score"])
                if len(online_history) < warmup:
                    online_history.append(score)
                    online_hist_updates += 1
                    if idx < 5:
                        print(
                            f"[5][online] trial={idx+1}, det_score={score:.4f}, "
                            f"stage=warmup, skip_decision=True"
                        )
                    continue
                if online_init_threshold is None:
                    hist_init = list(online_history)
                    fallback_thr = float(np.quantile(np.asarray(hist_init, dtype=np.float64), q))
                    if bool(getattr(cfg, "online_init_with_warmup_low_cluster", True)):
                        init_thr, n_low, n_high, gap_max = warmup_low_cluster_max_threshold(
                            hist_init, fallback=fallback_thr
                        )
                    else:
                        init_thr, n_low, n_high, gap_max = fallback_thr, 0, 0, 0.0
                    online_init_threshold = min(max(float(init_thr), min_thr), max_thr)
                    print(
                        f"[5][online] warmup done: init_threshold={online_init_threshold:.6f}, "
                        f"split_low={n_low}, split_high={n_high}, max_gap={gap_max:.6f}"
                    )
                    dynamic_thr = online_init_threshold
                else:
                    hist = online_history[-window:]
                    dynamic_thr = float(np.quantile(np.asarray(hist, dtype=np.float64), q))
                    dynamic_thr = min(max(dynamic_thr, min_thr), max_thr)
                pred_present = score > dynamic_thr
                online_thresholds.append(dynamic_thr)
                if (not cfg.online_update_with_neg_only) or (not pred_present):
                    online_history.append(score)
                    online_hist_updates += 1
                evaluated_records.append(r)
                if idx < 5:
                    print(
                        f"[5][online] trial={idx+1}, det_score={score:.4f}, "
                        f"threshold={dynamic_thr:.4f}, pred={pred_present}"
                    )
            else:
                pred_present = r["det_score"] > threshold
                evaluated_records.append(r)

            true_present = bool(r["member_present"])
            if true_present:
                member_pos_trials += 1
            if pred_present and true_present:
                TP += 1
            elif (not pred_present) and (not true_present):
                TN += 1
            elif pred_present and (not true_present):
                FP += 1
            else:
                FN += 1

            if true_present and pred_present:
                if r["argmax_idx"] == r["true_member"]:
                    member_idx_correct += 1

        total = len(evaluated_records)
        presence_acc = (TP + TN) / total if total else float("nan")
        tpr = TP / (TP + FN) if (TP + FN) else float("nan")
        fpr = FP / (FP + TN) if (FP + TN) else float("nan")
        asr = member_idx_correct / member_pos_trials if member_pos_trials else float("nan")

        roc_auc, _, _ = compute_roc_auc_and_maybe_plot(
            evaluated_records,
            (cfg.roc_plot_path or "").strip(),
            cfg.model_name,
            cfg.roc_dataset_label,
            scores_cache_path=(cfg.roc_scores_cache_path or "").strip(),
        )

        print("\n" + "=" * 60)
        print(
            f"[5] Test trials={total} (calib n={len(calib_records)}), det_mode={cfg.server_det_score_mode!r}"
        )
        print(
            f"[5] presence_acc={presence_acc:.1%}, TPR={tpr:.1%}, FPR={fpr:.1%}, "
            f"ASR={asr:.1%} (among member trials: predicted member and client ID correct / member trial count)"
        )
        if np.isfinite(roc_auc):
            print(f"[5] ROC-AUC (test, det_score)={roc_auc:.4f}")
        if mode == "online_quantile":
            if online_thresholds:
                thr_last = float(online_thresholds[-1])
                thr_mean = float(np.mean(online_thresholds))
            else:
                thr_last = float("nan")
                thr_mean = float("nan")
            skipped = len(test_records) - len(evaluated_records)
            print(
                f"[5] decision threshold(online): last={thr_last:.6f}, mean={thr_mean:.6f}, "
                f"mode={cfg.server_det_score_mode!r}"
            )
            print(
                f"[5] online warmup skipped trials={skipped}, evaluated={len(evaluated_records)}, "
                f"history_updates={online_hist_updates}"
            )
        else:
            print(f"[5] decision threshold={threshold:.6f}, mode={cfg.server_det_score_mode!r}")
        print("=" * 60)

    return presence_acc


if __name__ == "__main__":
    main()
