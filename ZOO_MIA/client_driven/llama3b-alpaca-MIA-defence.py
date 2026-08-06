"""
FedMeZO membership inference with upload defences (Alpaca + Open-Llama-3B + PEFT LoRA).

Select defence via --defence:
  dp    : optional L2 clip + Gaussian noise on uploaded g_sum (sweep epsilon)
  spas  : random coordinate sparsification (sweep keep_ratio)
  topk  : magnitude top-k sparsification (sweep topk_ratio)

Examples:
  python llama3b-alpaca-MIA-defence.py --defence dp
  python llama3b-alpaca-MIA-defence.py --defence spas
  python llama3b-alpaca-MIA-defence.py --defence topk
"""
import argparse
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

    # LoRA defaults aligned with llm/README.md (mezo_testcase.yaml uses alpha=16, dropout=0.05; editable)
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

    # Binary membership: with this probability inject the target into one client (else no injection)
    member_present_prob: float = 0.5
    # det_score mode (same as distilbert-agnews-MIA-zhibiao)
    server_det_score_mode: str = "gap"
    # Threshold: online_quantile / roc_youden / fixed
    server_threshold_mode: str = "online_quantile"
    threshold_calib_fraction: float = 0.5
    server_fixed_threshold: float = 0.0
    online_warmup: int = 20
    online_window: int = 40
    online_alpha: float = 0.3
    online_min_threshold: float = 10000
    online_max_threshold: float = 1000000
    online_update_with_neg_only: bool = True
    online_auto_clip_by_mode: bool = True
    online_init_with_warmup_low_cluster: bool = True
    roc_plot_path: str = ""
    roc_scores_cache_path: str = "outputs/mia_roc_scores_llama3b_alpaca.npz"
    roc_dataset_label: str = "Alpaca (Llama3B)"
    # Upload defence: "dp" | "spas" | "topk" (set via --defence)
    defence: str = "dp"

    # DP: Gaussian noise on uploaded g_sum (optional L2 clip)
    client_upload_noise_sigma: float = 0.0
    defense_epsilons: tuple = (0.1, 0.5, 2.0, 10.0)
    defense_dp_delta: float = 1e-5
    # Default attack comparison: fixed sensitivity=1, no L2 clip, coordinate-wise Gaussian noise.
    # Set <=0 to auto-estimate ||g_sum|| quantile as L2 sensitivity (stronger DP / larger noise).
    defense_dp_sensitivity: float = 1.0
    defense_dp_sensitivity_quantile: float = 0.95
    defense_dp_sensitivity_probe_trials: int = 30
    defense_dp_clip_before_noise: bool = False
    defense_sensitivity_cache_path: str = "outputs/llama3b_dp_l2_sensitivity.json"

    # spas: random coordinate sparsification keep ratio
    client_upload_sparsity_keep_ratio: float = 1.0
    defense_keep_ratios: tuple = (0.2, 0.4, 0.6, 0.8, 1.0)

    # topk: magnitude top-k keep ratio
    client_upload_topk_ratio: float = 1.0
    defense_topk_ratios: tuple = (0.2, 0.4, 0.6, 0.8, 1.0)

    # Sweep metrics outputs (filled by apply_defence_output_paths)
    defense_metrics_path: str = "outputs/asr_vs_epsilon_metrics.json"
    defense_metrics_csv_path: str = "outputs/asr_vs_epsilon_metrics.csv"
    defense_metrics_plot_path: str = "outputs/asr_vs_epsilon_metrics.png"

    adv_init_use: bool = True
    adv_init_steps: int = 500
    # Large model + high-order grads are unstable: prefer 1e-4~5e-4; 1e-3 often blows up mid-run
    adv_init_lr: float = 1e-4
    adv_init_w_target: float = 1.5
    adv_init_w_anchor: float = 0.15
    adv_init_anchor_power: float = 2.0
    # The max term often adds spikes/instability; keep off first, enable gradually later
    adv_init_w_anchor_max: float = 0.1
    # Clip trainable-param grad L2 before each update; 0 disables clipping
    adv_init_clip_grad_norm: float = 0.2
    adv_init_anchors_per_client: int = 50
    # Per-step random anchor subset size (<= total anchors) to cut 2nd-order peak memory
    adv_init_anchor_subset_size: int = 4
    # Adv-init anchors: "clients"=per-client DataLoaders (default); "attack_pool"=attack_batches
    adv_init_anchor_source: str = "attack_pool"
    adv_init_log_every: int = 10
    adv_init_bundle_path: str = "outputs/llama3b_alpaca_mia_adv_init_bundle.pt"
    adv_init_bundle_use: bool = True  # True: skip adv init when cache hits
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
        raise RuntimeError(f"Invalid or empty Alpaca data: {cfg.data_path}")

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

    # bf16 is usually more stable than fp16 (esp. adv init with high-order / 2nd-order graphs).
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
    print(f"Trainable params (LoRA+score): {trainable_params}")
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
    # ZO needs only scalar loss; do not build autograd graphs on both forwards or (loss_pos-loss_neg) links two Llama graphs and OOMs.
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
    Sample anchor batches from the attack pool (skip items with the same input_ids as target).
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
        print("[Adv init] Attack-pool anchors: no candidates (pool may contain only the target).")
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
    """Best-effort enable/disable gradient checkpointing on PEFT-wrapped Llama; return True on success."""
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
        print(f"[Adv init] Failed to toggle gradient checkpointing (safe to ignore): {e}")
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
        print("[Adv init] No target sample; skip.")
        return
    params = get_trainable_params(model)
    if not params:
        print("[Adv init] No trainable params; skip.")
        return

    gc_on = False
    if getattr(cfg, "adv_init_gradient_checkpointing", False):
        gc_on = _set_adversarial_grad_checkpointing(model, True)
        if gc_on:
            print("[Adv init] Enabled gradient checkpointing (adv-init stage only).")

    model.train()
    opt = torch.optim.Adam(params, lr=cfg.adv_init_lr)
    n_anchor = max(1, len(anchor_batches))
    subset_k = max(
        1,
        min(int(getattr(cfg, "adv_init_anchor_subset_size", n_anchor)), n_anchor),
    )
    if subset_k < len(anchor_batches):
        print(
            f"[Adv init] Each step randomly uses {subset_k}/{len(anchor_batches)} anchors in the loss."
        )
    model.zero_grad(set_to_none=True)

    n_t0, mean_a0, max_a0, L0 = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
    print(
        f"[Adv init] start step 0/{cfg.adv_init_steps}: "
        f"||∇L_target||={n_t0:.6f}, anchor mean||∇L||={mean_a0:.6f}, max||∇L||={max_a0:.6f}, "
        f"L={L0:.6f} (=-w_t||g_tgt||+(w_a/n)Σ||g||^{cfg.adv_init_anchor_power:g}"
        f"+{'w_max·max||g||' if cfg.adv_init_w_anchor_max > 0 else '0'}) "
        f"(num_anchors={len(anchor_batches)})"
    )

    log_every = max(0, int(cfg.adv_init_log_every))
    anchor_power = float(cfg.adv_init_anchor_power)
    nan_happened = False
    for step in tqdm(range(cfg.adv_init_steps), desc="Adv init (LoRA+score)"):
        opt.zero_grad(set_to_none=True)

        n_t = _grad_norm_classifier(model, target_batch, create_graph=True)
        if not torch.isfinite(n_t).item():
            nan_happened = True
            print(f"[Adv init] Non-finite n_t={n_t}; early stop at step={step+1}.")
            break
        # Random anchor subset each step; one scalar loss + one backward
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
                    print(f"[Adv init] Non-finite anchor na={na}; early stop at step={step+1}.")
                    break
                anchor_norms.append(na)
                loss = loss + (cfg.adv_init_w_anchor / max(1, subset_k)) * na.pow(anchor_power)
            if nan_happened:
                break
            if cfg.adv_init_w_anchor_max > 0 and anchor_norms:
                loss = loss + cfg.adv_init_w_anchor_max * torch.stack(anchor_norms).max()

        if not torch.isfinite(loss).item():
            nan_happened = True
            print(f"[Adv init] Non-finite loss={loss}; early stop at step={step+1}.")
            break
        loss.backward()

        clip = float(getattr(cfg, "adv_init_clip_grad_norm", 0.0) or 0.0)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)

        # Check grads for NaN/Inf (extra cost only when anomalies appear)
        for prm in params:
            if prm.grad is not None and not torch.isfinite(prm.grad).all():
                nan_happened = True
                print("[Adv init] Param grads contain NaN/Inf; early stop.")
                break
        if nan_happened:
            break
        opt.step()

        done = step + 1
        if log_every > 0 and (done % log_every == 0 or done == cfg.adv_init_steps):
            nt, ma, mxa, Lm = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
            print(
                f"[Adv init]  step {done:4d}/{cfg.adv_init_steps}: "
                f"||∇L_target||={nt:.6f}, anchor mean||∇L||={ma:.6f}, max||∇L||={mxa:.6f}, L={Lm:.6f}"
            )

    if gc_on:
        _set_adversarial_grad_checkpointing(model, False)

    if nan_happened:
        print("[Adv init] NaN/Inf detected; stop adv init (keep current params).")
        return

    model.eval()
    model.zero_grad(set_to_none=True)
    n_tf, mean_af, max_af, Lf = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
    rel = (n_tf - n_t0) / (n_t0 + 1e-12) * 100.0
    print(
        f"[Adv init] done: ||∇L_target|| {n_t0:.6f} -> {n_tf:.6f} "
        f"(rel {rel:+.2f}%), anchor mean||∇L|| {mean_a0:.6f} -> {mean_af:.6f}, "
        f"max||∇L|| {max_a0:.6f} -> {max_af:.6f}, L {L0:.6f} -> {Lf:.6f}"
    )
    if n_tf < 0.05 * max(n_t0, 1e-8):
        print(
            "[Adv init] Warning: final ||∇L_target|| still much lower than start; try smaller adv_init_lr, "
            "tighter adv_init_clip_grad_norm, or temporarily set adv_init_w_anchor_max=0."
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
    """Compute presence det_score from per-client grad-norm scores (same as zhibiao)."""
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
    Split sorted warmup scores at the largest gap into two clusters; return max of the low cluster as init threshold.
    If too few samples or split fails, fall back to fallback.
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


def clip_upload_l2(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip the upload vector into an L2 ball; max_norm is the DP sensitivity bound C."""
    max_norm = float(max_norm)
    if max_norm <= 0.0:
        return vec
    norm = float(vec.float().norm().item())
    if norm <= max_norm or norm <= 0.0:
        return vec
    return vec * (max_norm / norm)



def sparsify_uploaded_gradient(grad_vec: torch.Tensor, keep_ratio: float, rand_seed: int) -> torch.Tensor:
    r = float(keep_ratio)
    if r >= 1.0:
        return grad_vec
    if r <= 0.0:
        return torch.zeros_like(grad_vec)
    n = int(grad_vec.numel())
    if n == 0:
        return grad_vec
    k = int(np.ceil(n * r))
    k = max(1, min(n, k))
    if k >= n:
        return grad_vec
    flat = grad_vec.reshape(-1)
    if flat.is_cuda:
        gen = torch.Generator(device=flat.device)
        gen.manual_seed(int(rand_seed))
    else:
        gen = torch.Generator().manual_seed(int(rand_seed))
    idx = torch.randperm(n, generator=gen, device=flat.device)[:k]
    out = torch.zeros_like(flat)
    out[idx] = flat[idx]
    return out.view_as(grad_vec)


def topk_uploaded_gradient(grad_vec: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    r = float(keep_ratio)
    if r >= 1.0:
        return grad_vec
    if r <= 0.0:
        return torch.zeros_like(grad_vec)
    n = int(grad_vec.numel())
    if n == 0:
        return grad_vec
    k = int(np.ceil(n * r))
    k = max(1, min(n, k))
    if k >= n:
        return grad_vec
    flat = grad_vec.reshape(-1)
    _, idx = torch.topk(flat.abs(), k=k, largest=True, sorted=False)
    out = torch.zeros_like(flat)
    out[idx] = flat[idx]
    return out.view_as(grad_vec)


def apply_upload_defence(g_sum: torch.Tensor, cfg, *, cid: int, trial_seed: int) -> torch.Tensor:
    """Apply the selected upload defence to one client's g_sum before server scoring."""
    mode = str(getattr(cfg, "defence", "dp") or "dp").strip().lower()
    if mode in ("dp", "none", ""):
        clip_norm = float(getattr(cfg, "defense_upload_l2_clip", 0.0) or 0.0)
        if clip_norm > 0.0 and bool(getattr(cfg, "defense_dp_clip_before_noise", True)):
            g_sum = clip_upload_l2(g_sum, clip_norm)
        sigma = float(getattr(cfg, "client_upload_noise_sigma", 0.0))
        if sigma > 0.0:
            noise_gen = torch.Generator(device=g_sum.device) if g_sum.is_cuda else torch.Generator()
            noise_gen.manual_seed(trial_seed + cid * 100 + 77)
            g_sum = g_sum + torch.randn(
                g_sum.shape, generator=noise_gen, dtype=g_sum.dtype, device=g_sum.device
            ) * sigma
        return g_sum
    if mode == "spas":
        return sparsify_uploaded_gradient(
            g_sum,
            float(getattr(cfg, "client_upload_sparsity_keep_ratio", 1.0)),
            rand_seed=trial_seed + cid * 100 + 23,
        )
    if mode == "topk":
        return topk_uploaded_gradient(
            g_sum,
            float(getattr(cfg, "client_upload_topk_ratio", 1.0)),
        )
    raise ValueError(f"Unknown defence={mode!r}; expected dp/spas/topk")


def apply_defence_output_paths(cfg) -> None:
    """Set default metrics output paths for the selected defence."""
    mode = str(getattr(cfg, "defence", "dp") or "dp").strip().lower()
    defaults = {
        "dp": (
            "outputs/asr_vs_epsilon_metrics.json",
            "outputs/asr_vs_epsilon_metrics.csv",
            "outputs/asr_vs_epsilon_metrics.png",
        ),
        "spas": (
            "outputs/asr_vs_sparsity_metrics.json",
            "outputs/asr_vs_sparsity_metrics.csv",
            "outputs/asr_vs_sparsity_metrics.png",
        ),
        "topk": (
            "outputs/asr_vs_topk_metrics.json",
            "outputs/asr_vs_topk_metrics.csv",
            "outputs/asr_vs_topk_metrics.png",
        ),
    }
    if mode not in defaults:
        return
    jp, cp, pp = defaults[mode]
    cfg.defense_metrics_path = jp
    cfg.defense_metrics_csv_path = cp
    cfg.defense_metrics_plot_path = pp


def _defense_sensitivity_cache_meta(cfg):
    max_s = max(0, cfg.local_steps - 1)
    eff_inject = int(np.clip(cfg.round1_inject_step, 0, max_s))
    return {
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "num_clients": cfg.num_clients,
        "local_steps": cfg.local_steps,
        "round1_inject_step": eff_inject,
        "member_present_prob": cfg.member_present_prob,
        "probe_trials": int(cfg.defense_dp_sensitivity_probe_trials),
        "quantile": float(cfg.defense_dp_sensitivity_quantile),
    }


def collect_client_upload_l2_norms(
    model,
    client_loaders,
    target_batch,
    cfg,
    *,
    num_trials: int,
    seed_offset: int = 0,
):
    """Probe L2 norms of noiseless client uploads g_sum (mixed member/non-member)."""
    if cfg.local_steps < 1:
        return []

    max_s = cfg.local_steps - 1
    eff_inject = int(np.clip(cfg.round1_inject_step, 0, max_s))
    r1_rng = random.Random(cfg.seed + 9109 + seed_offset)
    norms = []

    for trial in range(int(num_trials)):
        member_present = r1_rng.random() < cfg.member_present_prob
        true_member = r1_rng.randrange(cfg.num_clients) if member_present else None
        trial_seed = cfg.seed + seed_offset + 91000 + trial * 1000
        theta0 = copy.deepcopy(model)
        for cid in range(cfg.num_clients):
            mloc = copy.deepcopy(theta0)
            this_inject_step = (
                eff_inject if (member_present and cid == true_member) else None
            )
            g_sum = client_update_with_inject_step(
                mloc,
                client_loaders[cid],
                target_batch,
                cfg,
                seed=trial_seed + cid * 100,
                inject_at_step=this_inject_step,
            )
            norms.append(float(g_sum.float().norm().item()))
    return norms


def estimate_client_upload_l2_sensitivity(
    model,
    client_loaders,
    target_batch,
    cfg,
    *,
    force_reestimate: bool = False,
):
    """
    Estimate L2 sensitivity C from the quantile of ||g_sum|| on probe trials.
    C is both the upload clip threshold and the Gaussian-mechanism sensitivity.
    """
    fixed = float(getattr(cfg, "defense_dp_sensitivity", 0.0) or 0.0)
    if fixed > 0.0 and not force_reestimate:
        return {
            "l2_sensitivity": fixed,
            "source": "fixed",
            "probe_norms": [],
            "norm_mean": float("nan"),
            "norm_max": float("nan"),
            "norm_quantile": float("nan"),
        }

    cache_path = (getattr(cfg, "defense_sensitivity_cache_path", "") or "").strip()
    meta = _defense_sensitivity_cache_meta(cfg)
    if cache_path and not force_reestimate and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("meta") == meta:
                c = float(cached["l2_sensitivity"])
                print(f"[DP] Hit L2 sensitivity cache: C={c:.6f} ({cache_path})")
                return {
                    "l2_sensitivity": c,
                    "source": "cache",
                    "probe_norms": cached.get("probe_norms", []),
                    "norm_mean": float(cached.get("norm_mean", float("nan"))),
                    "norm_max": float(cached.get("norm_max", float("nan"))),
                    "norm_quantile": float(cached.get("norm_quantile", float("nan"))),
                }
        except Exception as e:
            print(f"[DP] Failed to read L2 sensitivity cache; re-probing: {e}")

    probe_trials = int(max(1, getattr(cfg, "defense_dp_sensitivity_probe_trials", 30)))
    q = float(np.clip(getattr(cfg, "defense_dp_sensitivity_quantile", 0.95), 0.5, 1.0))
    print(f"[DP] Probing client-upload L2 sensitivity: probe_trials={probe_trials}, quantile={q:.2f}")
    norms = collect_client_upload_l2_norms(
        model,
        client_loaders,
        target_batch,
        cfg,
        num_trials=probe_trials,
    )
    if not norms:
        raise RuntimeError("Failed to probe any client upload gradient norms.")

    arr = np.asarray(norms, dtype=np.float64)
    c = float(np.quantile(arr, q))
    c = max(c, 1e-8)
    stats = {
        "l2_sensitivity": c,
        "source": "probe",
        "probe_norms": norms,
        "norm_mean": float(np.mean(arr)),
        "norm_max": float(np.max(arr)),
        "norm_quantile": c,
    }
    print(
        f"[DP] Probe done: ||g_sum|| mean={stats['norm_mean']:.4f}, "
        f"max={stats['norm_max']:.4f}, q{q:.0f}={c:.4f} => C={c:.4f}"
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": meta,
                    "l2_sensitivity": c,
                    "norm_mean": stats["norm_mean"],
                    "norm_max": stats["norm_max"],
                    "norm_quantile": stats["norm_quantile"],
                    "probe_norms": norms,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[DP] Saved L2 sensitivity cache: {cache_path}")
    return stats


def resolve_defense_dp_settings(model, client_loaders, target_batch, cfg):
    """Resolve L2 sensitivity C and upload clip threshold; write back to cfg for simulate."""
    sens_info = estimate_client_upload_l2_sensitivity(
        model, client_loaders, target_batch, cfg
    )
    c = float(sens_info["l2_sensitivity"])
    cfg.defense_upload_l2_clip = c if bool(getattr(cfg, "defense_dp_clip_before_noise", True)) else 0.0
    cfg.defense_dp_sensitivity_resolved = c
    return sens_info


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
        g_sum = apply_upload_defence(g_sum, cfg, cid=cid, trial_seed=trial_seed)
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
    """Only trainable params linked to loss may have grads; under PEFT some leaves can be None — use allow_unused."""
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
        print("Warning: attack pool empty; cannot select a target sample.")
        return None

    best_norm = -1.0
    best_batch = None
    for i in tqdm(range(len(samples)), desc="Selecting target"):
        batch = samples[i]
        g = compute_true_gradient(model, batch)
        norm = g.norm().item()
        if norm > best_norm:
            best_norm = norm
            best_batch = batch

    print(f"Target sample grad norm: {best_norm:.6f}")
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
        print(f"[Adv-init cache] Load failed; will retrain: {e}")
        return False
    if not isinstance(obj, dict) or obj.get("meta") != cache_meta_45:
        print("[Adv-init cache] meta mismatch; will retrain.")
        return False
    tb = obj.get("target_batch")
    if not _target_batch_match(tb, target_batch):
        print("[Adv-init cache] target sample mismatch; will retrain.")
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
            print(f"[Adv-init cache] OOM while loading into model; retrain/skip: {e}")
            return False
        raise
    print(f"[Adv-init cache] Loaded model and target: {path}")
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
    print(f"[Adv-init cache] Saved: {cfg.adv_init_bundle_path}")


def run_round1_eval(model, client_loaders, target_batch, cfg, *, seed_offset=0, tag="5", save_roc=True):
    mode = str(getattr(cfg, "defence", "dp") or "dp").strip().lower()
    if mode == "spas":
        defence_info = f"keep_ratio={float(getattr(cfg, 'client_upload_sparsity_keep_ratio', 1.0))}"
    elif mode == "topk":
        defence_info = f"topk_ratio={float(getattr(cfg, 'client_upload_topk_ratio', 1.0))}"
    else:
        defence_info = f"sigma={float(getattr(cfg, 'client_upload_noise_sigma', 0.0))}"
    print(
        f"\n[{tag}] Round-1 FL sim + membership detection; "
        f"ASR/TPR/FPR/AUC... (defence={mode}, {defence_info})"
    )
    r1_rng = random.Random(cfg.seed + 1009 + seed_offset)
    presence_acc = float("nan")
    roc_auc = float("nan")
    asr = float("nan")
    if cfg.local_steps < 1:
        print(f"Skip [{tag}]: local_steps must be >= 1.")
        return {
            "presence_acc": presence_acc,
            "roc_auc": roc_auc,
            "asr": asr,
            "tpr": float("nan"),
            "fpr": float("nan"),
        }

    max_s = cfg.local_steps - 1
    eff_inject = int(np.clip(cfg.round1_inject_step, 0, max_s))
    if cfg.round1_inject_step != eff_inject:
        print(f"Warning: round1_inject_step={cfg.round1_inject_step} clipped to [0,{max_s}]: {eff_inject}")

    trial_records = []
    for trial in range(cfg.round1_server_trials):
        member_present = r1_rng.random() < cfg.member_present_prob
        true_member = r1_rng.randrange(cfg.num_clients) if member_present else None
        rec = simulate_round1_trial_record(
            model,
            client_loaders,
            target_batch,
            cfg,
            trial_seed=cfg.seed + seed_offset + 50000 + trial * 1000,
            member_present=member_present,
            true_member=true_member,
            inject_step=eff_inject,
        )
        trial_records.append(rec)
        print(
            f"[{tag}] trial {trial+1}: true_present={rec['member_present']}, "
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
            f"[{tag}] Online dynamic threshold: mode=online_quantile, warmup={cfg.online_warmup}, "
            f"window={cfg.online_window}, alpha={cfg.online_alpha}, "
            f"clip=[{clip_min}, {clip_max}], neg_only_update={cfg.online_update_with_neg_only}"
        )
    else:
        test_records = trial_records[calib_end:]
        if not test_records:
            print(f"[{tag}] Warning: no leftover test trials after calib; evaluating on all trials (optimistic).")
            test_records = list(trial_records)
            calib_records = trial_records
        if mode == "fixed":
            print(f"[{tag}] Using fixed threshold: threshold={threshold}")
        elif mode == "roc_youden":
            y_true_calib = np.array([1 if r["member_present"] else 0 for r in calib_records], dtype=np.int32)
            y_score_calib = np.array([r["det_score"] for r in calib_records], dtype=np.float64)
            if len(np.unique(y_true_calib)) < 2:
                print(f"[{tag}] ROC-Youden calib failed (single-class calib); fallback fixed threshold={threshold}")
            else:
                try:
                    from sklearn.metrics import roc_curve
                except ImportError:
                    print(f"[{tag}] ROC-Youden calib failed (sklearn missing); fallback fixed threshold")
                else:
                    fpr_calib, tpr_calib, thr_calib = roc_curve(y_true_calib, y_score_calib)
                    valid = np.isfinite(thr_calib)
                    if not np.any(valid):
                        print(f"[{tag}] ROC-Youden calib failed (non-finite threshold); fallback fixed threshold")
                    else:
                        youden = tpr_calib[valid] - fpr_calib[valid]
                        best_idx_local = int(np.argmax(youden))
                        threshold = float(thr_calib[valid][best_idx_local])
                        best_j = float(youden[best_idx_local])
                        best_tpr = float(tpr_calib[valid][best_idx_local])
                        best_fpr = float(fpr_calib[valid][best_idx_local])
                        print(
                            f"[{tag}] ROC-Youden calib (calib n={len(calib_records)}): "
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
                        f"[{tag}][online] trial={idx+1}, det_score={score:.4f}, "
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
                    f"[{tag}][online] warmup done: init_threshold={online_init_threshold:.6f}, "
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
                    f"[{tag}][online] trial={idx+1}, det_score={score:.4f}, "
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

    roc_plot_path = (cfg.roc_plot_path or "").strip() if save_roc else ""
    roc_scores_cache_path = (cfg.roc_scores_cache_path or "").strip() if save_roc else ""
    roc_auc, _, _ = compute_roc_auc_and_maybe_plot(
        evaluated_records,
        roc_plot_path,
        cfg.model_name,
        cfg.roc_dataset_label,
        scores_cache_path=roc_scores_cache_path,
    )

    print("\n" + "=" * 60)
    print(
        f"[{tag}] test trials={total} (calib n={len(calib_records)}), det_mode={cfg.server_det_score_mode!r}"
    )
    print(
        f"[{tag}] presence_acc={presence_acc:.1%}, TPR={tpr:.1%}, FPR={fpr:.1%}, "
        f"ASR={asr:.1%} (among member-present trials: predicted present and correct client / #member trials)"
    )
    if np.isfinite(roc_auc):
        print(f"[{tag}] ROC-AUC (test, det_score)={roc_auc:.4f}")
    if mode == "online_quantile":
        if online_thresholds:
            thr_last = float(online_thresholds[-1])
            thr_mean = float(np.mean(online_thresholds))
        else:
            thr_last = float("nan")
            thr_mean = float("nan")
        skipped = len(test_records) - len(evaluated_records)
        print(
            f"[{tag}] decision threshold(online): last={thr_last:.6f}, mean={thr_mean:.6f}, "
            f"mode={cfg.server_det_score_mode!r}"
        )
        print(
            f"[{tag}] online warmup skipped trials={skipped}, evaluated={len(evaluated_records)}, "
            f"history_updates={online_hist_updates}"
        )
    else:
        print(f"[{tag}] decision threshold={threshold:.6f}, mode={cfg.server_det_score_mode!r}")
    print("=" * 60)
    return {
        "presence_acc": float(presence_acc),
        "roc_auc": float(roc_auc),
        "asr": float(asr),
        "tpr": float(tpr),
        "fpr": float(fpr),
    }


def _safe_json_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _plot_defense_epsilon_results(rows, cfg):
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Defence exp] matplotlib unavailable; skip plot: {e}")
        return

    plot_path = (getattr(cfg, "defense_metrics_plot_path", "") or "").strip()
    if not plot_path:
        return

    epsilons = np.asarray([r["epsilon"] for r in rows], dtype=np.float64)
    asrs = np.asarray([r["asr"] for r in rows], dtype=np.float64)
    aucs = np.asarray([r["roc_auc"] for r in rows], dtype=np.float64)

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6.8, 4.8))
    ax1.plot(epsilons, asrs, marker="s", linewidth=2, color="#d62728", label="ASR")
    ax1.set_xscale("log")
    ax1.set_xlabel("DP epsilon")
    ax1.set_ylabel("ASR", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(epsilons, aucs, marker="o", linewidth=2, color="#1f77b4", label="ROC-AUC")
    ax2.set_ylabel("ROC-AUC", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(-0.02, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Defence exp] Saved ASR/AUC plot: {plot_path}")


def run_defense_epsilon_experiment(
    model,
    client_loaders,
    target_batch,
    cfg,
    *,
    baseline_metrics=None,
):
    epsilons = [float(x) for x in cfg.defense_epsilons]
    rows = []
    original_sigma = float(getattr(cfg, "client_upload_noise_sigma", 0.0))
    original_clip = float(getattr(cfg, "defense_upload_l2_clip", 0.0) or 0.0)

    sens_info = resolve_defense_dp_settings(model, client_loaders, target_batch, cfg)
    l2_sensitivity = float(sens_info["l2_sensitivity"])
    trainable_dim = int(
        sum(p.numel() for p in get_trainable_params(model))
    )
    expected_noise_factor = float(np.sqrt(max(1, trainable_dim)))
    print(
        f"[6] DP calibration: L2 sensitivity C={l2_sensitivity:.6f} ({sens_info['source']}), "
        f"clip_before_noise={bool(getattr(cfg, 'defense_dp_clip_before_noise', False))}, "
        f"trainable_dim={trainable_dim}"
    )

    if baseline_metrics is not None:
        rows.append(
            {
                "epsilon": None,
                "sigma": 0.0,
                "upload_l2_sensitivity": l2_sensitivity,
                "expected_noise_l2": 0.0,
                "noise_to_signal_ratio": 0.0,
                "asr": float(baseline_metrics.get("asr", float("nan"))),
                "roc_auc": float(baseline_metrics.get("roc_auc", float("nan"))),
                "presence_acc": float(baseline_metrics.get("presence_acc", float("nan"))),
                "tpr": float(baseline_metrics.get("tpr", float("nan"))),
                "fpr": float(baseline_metrics.get("fpr", float("nan"))),
            }
        )
        print(
            f"[6] baseline (no DP): ASR={rows[-1]['asr']:.1%}, "
            f"ROC-AUC={rows[-1]['roc_auc']:.4f}"
        )

    try:
        for epsilon in epsilons:
            sigma = float(
                epsilon_to_gaussian_sigma(
                    epsilon, cfg.defense_dp_delta, l2_sensitivity
                )
            )
            expected_noise_l2 = sigma * expected_noise_factor
            cfg.client_upload_noise_sigma = sigma
            metrics = run_round1_eval(
                model,
                client_loaders,
                target_batch,
                cfg,
                seed_offset=0,
                tag=f"6 epsilon={epsilon}",
                save_roc=False,
            )
            row = {
                "epsilon": float(epsilon),
                "sigma": float(sigma),
                "upload_l2_sensitivity": l2_sensitivity,
                "expected_noise_l2": float(expected_noise_l2),
                "asr": float(metrics["asr"]),
                "roc_auc": float(metrics["roc_auc"]),
                "presence_acc": float(metrics["presence_acc"]),
                "tpr": float(metrics["tpr"]),
                "fpr": float(metrics["fpr"]),
            }
            if sens_info.get("source") != "fixed":
                row["noise_to_signal_ratio"] = float(
                    expected_noise_l2 / max(l2_sensitivity, 1e-12)
                )
            rows.append(row)
            if sens_info.get("source") == "fixed":
                print(
                    f"[6] epsilon={epsilon:.4g}, sigma={sigma:.6f}, "
                    f"E||noise||≈{expected_noise_l2:.2f}, "
                    f"ASR={row['asr']:.1%}, ROC-AUC={row['roc_auc']:.4f}"
                )
            else:
                print(
                    f"[6] epsilon={epsilon:.4g}, sigma={sigma:.6f}, "
                    f"E||noise||≈{expected_noise_l2:.2f}, "
                    f"noise/signal≈{row['noise_to_signal_ratio']:.3f}, "
                    f"ASR={row['asr']:.1%}, ROC-AUC={row['roc_auc']:.4f}"
                )
    finally:
        cfg.client_upload_noise_sigma = original_sigma
        cfg.defense_upload_l2_clip = original_clip

    dp_rows = [r for r in rows if r.get("epsilon") is not None]
    metrics_obj = {
        "epsilon": [r["epsilon"] for r in dp_rows],
        "sigma": [_safe_json_float(r["sigma"]) for r in dp_rows],
        "asr_measured": [_safe_json_float(r["asr"]) for r in dp_rows],
        "roc_auc_measured": [_safe_json_float(r["roc_auc"]) for r in dp_rows],
        "presence_acc_measured": [_safe_json_float(r["presence_acc"]) for r in dp_rows],
        "tpr_measured": [_safe_json_float(r["tpr"]) for r in dp_rows],
        "fpr_measured": [_safe_json_float(r["fpr"]) for r in dp_rows],
        "dp_delta": float(cfg.defense_dp_delta),
        "dp_l2_sensitivity": _safe_json_float(l2_sensitivity),
        "dp_sensitivity_source": sens_info.get("source"),
        "dp_sensitivity_quantile": float(cfg.defense_dp_sensitivity_quantile),
        "dp_sensitivity_probe_trials": int(cfg.defense_dp_sensitivity_probe_trials),
        "dp_clip_before_noise": bool(cfg.defense_dp_clip_before_noise),
        "upload_norm_mean": _safe_json_float(sens_info.get("norm_mean")),
        "upload_norm_max": _safe_json_float(sens_info.get("norm_max")),
        "trainable_param_dim": trainable_dim,
        "rows": [
            {
                k: (
                    _safe_json_float(v)
                    if isinstance(v, (float, np.floating))
                    else v
                )
                for k, v in row.items()
            }
            for row in rows
        ],
    }

    metrics_path = (cfg.defense_metrics_path or "").strip()
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"[Defence exp] Saved metrics to: {metrics_path}")

    csv_path = (getattr(cfg, "defense_metrics_csv_path", "") or "").strip()
    if csv_path and rows:
        import csv

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: (
                            ""
                            if isinstance(v, (float, np.floating)) and not np.isfinite(v)
                            else v
                        )
                        for k, v in row.items()
                    }
                )
        print(f"[Defence exp] Saved CSV: {csv_path}")

    _plot_defense_epsilon_results(dp_rows, cfg)
    return metrics_obj



def _plot_defense_sparsity_results(rows, cfg):
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Defence exp] matplotlib unavailable; skip plot: {e}")
        return

    plot_path = (getattr(cfg, "defense_metrics_plot_path", "") or "").strip()
    if not plot_path:
        return

    ratios = np.asarray([r["keep_ratio"] for r in rows], dtype=np.float64)
    asrs = np.asarray([r["asr"] for r in rows], dtype=np.float64)
    aucs = np.asarray([r["roc_auc"] for r in rows], dtype=np.float64)

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6.8, 4.8))
    ax1.plot(ratios, asrs, marker="s", linewidth=2, color="#d62728", label="ASR")
    ax1.set_xlabel("keep ratio")
    ax1.set_ylabel("ASR", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(ratios, aucs, marker="o", linewidth=2, color="#1f77b4", label="ROC-AUC")
    ax2.set_ylabel("ROC-AUC", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(-0.02, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Defence exp] Saved ASR/AUC plot: {plot_path}")

def run_defense_sparsity_experiment(
    model,
    client_loaders,
    target_batch,
    cfg,
    *,
    baseline_metrics=None,
):
    keep_ratios = [float(x) for x in cfg.defense_keep_ratios]
    rows = []
    original_ratio = float(getattr(cfg, "client_upload_sparsity_keep_ratio", 1.0))

    try:
        for ratio in keep_ratios:
            if (
                baseline_metrics is not None
                and abs(ratio - 1.0) < 1e-12
            ):
                row = {
                    "keep_ratio": float(ratio),
                    "asr": float(baseline_metrics.get("asr", float("nan"))),
                    "roc_auc": float(baseline_metrics.get("roc_auc", float("nan"))),
                    "presence_acc": float(baseline_metrics.get("presence_acc", float("nan"))),
                    "tpr": float(baseline_metrics.get("tpr", float("nan"))),
                    "fpr": float(baseline_metrics.get("fpr", float("nan"))),
                }
                rows.append(row)
                print(
                    f"[6] keep_ratio=1.0 reuse [5] baseline: "
                    f"ASR={row['asr']:.1%}, ROC-AUC={row['roc_auc']:.4f}"
                )
                continue

            cfg.client_upload_sparsity_keep_ratio = ratio
            metrics = run_round1_eval(
                model,
                client_loaders,
                target_batch,
                cfg,
                seed_offset=0,
                tag=f"6 keep={ratio}",
                save_roc=False,
            )
            row = {
                "keep_ratio": float(ratio),
                "asr": float(metrics["asr"]),
                "roc_auc": float(metrics["roc_auc"]),
                "presence_acc": float(metrics["presence_acc"]),
                "tpr": float(metrics["tpr"]),
                "fpr": float(metrics["fpr"]),
            }
            rows.append(row)
            print(
                f"[6] keep_ratio={ratio:.1f}, ASR={row['asr']:.1%}, "
                f"ROC-AUC={row['roc_auc']:.4f}"
            )
    finally:
        cfg.client_upload_sparsity_keep_ratio = original_ratio

    metrics_obj = {
        "keep_ratio": [r["keep_ratio"] for r in rows],
        "asr_measured": [_safe_json_float(r["asr"]) for r in rows],
        "roc_auc_measured": [_safe_json_float(r["roc_auc"]) for r in rows],
        "presence_acc_measured": [_safe_json_float(r["presence_acc"]) for r in rows],
        "tpr_measured": [_safe_json_float(r["tpr"]) for r in rows],
        "fpr_measured": [_safe_json_float(r["fpr"]) for r in rows],
        "rows": [
            {
                k: _safe_json_float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in row.items()
            }
            for row in rows
        ],
    }

    metrics_path = (cfg.defense_metrics_path or "").strip()
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"[Defence exp] Saved metrics to: {metrics_path}")

    csv_path = (getattr(cfg, "defense_metrics_csv_path", "") or "").strip()
    if csv_path and rows:
        import csv

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: (
                            ""
                            if isinstance(v, (float, np.floating)) and not np.isfinite(v)
                            else v
                        )
                        for k, v in row.items()
                    }
                )
        print(f"[Defence exp] Saved CSV: {csv_path}")

    _plot_defense_sparsity_results(rows, cfg)
    return metrics_obj

def _plot_defense_topk_results(rows, cfg):
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Defence exp] matplotlib unavailable; skip plot: {e}")
        return

    plot_path = (getattr(cfg, "defense_metrics_plot_path", "") or "").strip()
    if not plot_path:
        return

    ratios = np.asarray([r["topk_ratio"] for r in rows], dtype=np.float64)
    asrs = np.asarray([r["asr"] for r in rows], dtype=np.float64)
    aucs = np.asarray([r["roc_auc"] for r in rows], dtype=np.float64)

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6.8, 4.8))
    ax1.plot(ratios, asrs, marker="s", linewidth=2, color="#d62728", label="ASR")
    ax1.set_xlabel("top-k ratio")
    ax1.set_ylabel("ASR", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(ratios, aucs, marker="o", linewidth=2, color="#1f77b4", label="ROC-AUC")
    ax2.set_ylabel("ROC-AUC", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(-0.02, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Defence exp] Saved ASR/AUC plot: {plot_path}")

def run_defense_topk_experiment(
    model,
    client_loaders,
    target_batch,
    cfg,
    *,
    baseline_metrics=None,
):
    topk_ratios = [float(x) for x in cfg.defense_topk_ratios]
    rows = []
    original_ratio = float(getattr(cfg, "client_upload_topk_ratio", 1.0))

    try:
        for ratio in topk_ratios:
            if (
                baseline_metrics is not None
                and abs(ratio - 1.0) < 1e-12
            ):
                row = {
                    "topk_ratio": float(ratio),
                    "asr": float(baseline_metrics.get("asr", float("nan"))),
                    "roc_auc": float(baseline_metrics.get("roc_auc", float("nan"))),
                    "presence_acc": float(baseline_metrics.get("presence_acc", float("nan"))),
                    "tpr": float(baseline_metrics.get("tpr", float("nan"))),
                    "fpr": float(baseline_metrics.get("fpr", float("nan"))),
                }
                rows.append(row)
                print(
                    f"[6] topk_ratio=1.0 reuse [5] baseline: "
                    f"ASR={row['asr']:.1%}, ROC-AUC={row['roc_auc']:.4f}"
                )
                continue

            cfg.client_upload_topk_ratio = ratio
            metrics = run_round1_eval(
                model,
                client_loaders,
                target_batch,
                cfg,
                seed_offset=0,
                tag=f"6 topk={ratio}",
                save_roc=False,
            )
            row = {
                "topk_ratio": float(ratio),
                "asr": float(metrics["asr"]),
                "roc_auc": float(metrics["roc_auc"]),
                "presence_acc": float(metrics["presence_acc"]),
                "tpr": float(metrics["tpr"]),
                "fpr": float(metrics["fpr"]),
            }
            rows.append(row)
            print(
                f"[6] topk_ratio={ratio:.1f}, ASR={row['asr']:.1%}, "
                f"ROC-AUC={row['roc_auc']:.4f}"
            )
    finally:
        cfg.client_upload_topk_ratio = original_ratio

    metrics_obj = {
        "topk_ratio": [r["topk_ratio"] for r in rows],
        "asr_measured": [_safe_json_float(r["asr"]) for r in rows],
        "roc_auc_measured": [_safe_json_float(r["roc_auc"]) for r in rows],
        "presence_acc_measured": [_safe_json_float(r["presence_acc"]) for r in rows],
        "tpr_measured": [_safe_json_float(r["tpr"]) for r in rows],
        "fpr_measured": [_safe_json_float(r["fpr"]) for r in rows],
        "rows": [
            {
                k: _safe_json_float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in row.items()
            }
            for row in rows
        ],
    }

    metrics_path = (cfg.defense_metrics_path or "").strip()
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"[Defence exp] Saved metrics to: {metrics_path}")

    csv_path = (getattr(cfg, "defense_metrics_csv_path", "") or "").strip()
    if csv_path and rows:
        import csv

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: (
                            ""
                            if isinstance(v, (float, np.floating)) and not np.isfinite(v)
                            else v
                        )
                        for k, v in row.items()
                    }
                )
        print(f"[Defence exp] Saved CSV: {csv_path}")

    _plot_defense_topk_results(rows, cfg)
    return metrics_obj

def epsilon_to_gaussian_sigma(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    eps = float(epsilon)
    dlt = float(delta)
    sens = float(sensitivity)
    if eps <= 0:
        raise ValueError("epsilon must be > 0.")
    if not (0 < dlt < 1):
        raise ValueError("delta must be in (0, 1).")
    if sens <= 0:
        raise ValueError("sensitivity must be > 0.")
    return sens * np.sqrt(2.0 * np.log(1.25 / dlt)) / eps


def parse_args():
    p = argparse.ArgumentParser(
        description="Llama3B Alpaca MIA with upload defence (dp / spas / topk)"
    )
    p.add_argument(
        "--defence",
        type=str,
        default="dp",
        choices=["dp", "spas", "topk"],
        help="Upload defence: dp=Gaussian DP noise, spas=random sparsify, topk=magnitude top-k",
    )
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0 / cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data_path", type=str, default="")
    p.add_argument("--model_name", type=str, default="")
    p.add_argument("--num_clients", type=int, default=None)
    p.add_argument("--round1_server_trials", type=int, default=None)
    # DP overrides
    p.add_argument(
        "--dp_epsilons",
        type=float,
        nargs="+",
        default=None,
        help="Epsilon list for DP sweep",
    )
    p.add_argument("--dp_delta", type=float, default=None)
    p.add_argument("--dp_sensitivity", type=float, default=None)
    p.add_argument("--dp_clip_before_noise", action="store_true", default=False)
    p.add_argument("--no_dp_clip_before_noise", action="store_true", default=False)
    # spas / topk overrides
    p.add_argument(
        "--keep_ratios",
        type=float,
        nargs="+",
        default=None,
        help="keep_ratio list for spas sweep",
    )
    p.add_argument(
        "--topk_ratios",
        type=float,
        nargs="+",
        default=None,
        help="topk_ratio list for topk sweep",
    )
    return p.parse_args()


def apply_args_to_cfg(cfg: Config, args) -> Config:
    cfg.defence = str(args.defence).strip().lower()
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.data_path:
        cfg.data_path = args.data_path
    if args.model_name:
        cfg.model_name = args.model_name
    if args.num_clients is not None:
        cfg.num_clients = int(args.num_clients)
    if args.round1_server_trials is not None:
        cfg.round1_server_trials = int(args.round1_server_trials)
    if args.dp_epsilons is not None:
        cfg.defense_epsilons = tuple(float(x) for x in args.dp_epsilons)
    if args.dp_delta is not None:
        cfg.defense_dp_delta = float(args.dp_delta)
    if args.dp_sensitivity is not None:
        cfg.defense_dp_sensitivity = float(args.dp_sensitivity)
    if args.no_dp_clip_before_noise:
        cfg.defense_dp_clip_before_noise = False
    elif args.dp_clip_before_noise:
        cfg.defense_dp_clip_before_noise = True
    if args.keep_ratios is not None:
        cfg.defense_keep_ratios = tuple(float(x) for x in args.keep_ratios)
    if args.topk_ratios is not None:
        cfg.defense_topk_ratios = tuple(float(x) for x in args.topk_ratios)
    apply_defence_output_paths(cfg)
    return cfg


def main():
    args = parse_args()
    cfg = apply_args_to_cfg(Config(), args)
    set_seed(cfg.seed)
    print(f"device: {cfg.device}")
    print(f"defence: {cfg.defence}")
    print(f"config: {cfg}")

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

    print(f"clients: {cfg.num_clients}, samples/client: {per_client}")
    print(f"attack pool: {cfg.attack_samples}")

    print("\n[2] Loading model...")
    model, tokenizer = get_model_and_tokenizer(cfg)

    print("\n[3] Creating dataloaders...")
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
                print(f"Hit target cache: {cfg.target_cache_path}")
        except Exception as e:
            print(f"Failed to load target cache, recomputing: {e}")
    if target_batch is None:
        target_batch = select_target_sample(model, attack_batches, cfg)
        os.makedirs(os.path.dirname(cfg.target_cache_path) or ".", exist_ok=True)
        torch.save({"meta": cache_meta_45, "target_batch": target_batch}, cfg.target_cache_path)
        print(f"Saved target cache: {cfg.target_cache_path}")

    if cfg.adv_init_use:
        loaded = try_load_adv_init_bundle(model, cfg, cache_meta_45, target_batch)
        if not loaded:
            print("\n[4.6] Adversarial init (LoRA + CausalLM)...")
            src = (cfg.adv_init_anchor_source or "clients").strip().lower()
            if src == "attack_pool":
                anchor_batches = collect_anchor_batches_from_attack_pool(
                    attack_batches, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(f"[Adv init] anchors from attack pool ({len(anchor_batches)} batches)")
            elif src == "clients":
                anchor_batches = collect_anchor_batches(
                    client_loaders, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(f"[Adv init] anchors from clients ({len(anchor_batches)} batches)")
            else:
                print(
                    f"Warning: adv_init_anchor_source={cfg.adv_init_anchor_source!r} invalid; using clients."
                )
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
                    print(f"[Adv init] OOM, skip and continue: {e}")
                else:
                    raise
            if adv_init_ok:
                save_adv_init_bundle(model, cfg, cache_meta_45, target_batch)

    # Baseline [5] with defence disabled
    original_sigma = float(getattr(cfg, "client_upload_noise_sigma", 0.0))
    original_spas = float(getattr(cfg, "client_upload_sparsity_keep_ratio", 1.0))
    original_topk = float(getattr(cfg, "client_upload_topk_ratio", 1.0))
    cfg.client_upload_noise_sigma = 0.0
    cfg.client_upload_sparsity_keep_ratio = 1.0
    cfg.client_upload_topk_ratio = 1.0
    base_metrics = run_round1_eval(model, client_loaders, target_batch, cfg, tag="5", save_roc=True)
    cfg.client_upload_noise_sigma = original_sigma
    cfg.client_upload_sparsity_keep_ratio = original_spas
    cfg.client_upload_topk_ratio = original_topk

    mode = cfg.defence
    if mode == "dp":
        print("\n[6] DP defence sweep: noise on uploaded g_sum vs epsilon.")
        defense_metrics = run_defense_epsilon_experiment(
            model, client_loaders, target_batch, cfg, baseline_metrics=base_metrics
        )
        print("[6] done:")
        for row in defense_metrics.get("rows", []):
            if row.get("epsilon") is None:
                print(f"  baseline (no DP): ASR={row['asr']:.1%}, ROC-AUC={row['roc_auc']:.4f}")
            else:
                print(
                    f"  epsilon={row['epsilon']:.4g}: ASR={row['asr']:.1%}, "
                    f"ROC-AUC={row['roc_auc']:.4f}"
                )
    elif mode == "spas":
        print("\n[6] Random sparsification sweep: keep_ratio vs ASR/AUC.")
        defense_metrics = run_defense_sparsity_experiment(
            model, client_loaders, target_batch, cfg, baseline_metrics=base_metrics
        )
        print("[6] done:")
        for row in defense_metrics.get("rows", []):
            print(
                f"  keep_ratio={row['keep_ratio']:.1f}: "
                f"ASR={row['asr']:.1%}, ROC-AUC={row['roc_auc']:.4f}"
            )
    elif mode == "topk":
        print("\n[6] Magnitude top-k sweep: topk_ratio vs ASR/AUC.")
        defense_metrics = run_defense_topk_experiment(
            model, client_loaders, target_batch, cfg, baseline_metrics=base_metrics
        )
        print("[6] done:")
        for row in defense_metrics.get("rows", []):
            print(
                f"  topk_ratio={row['topk_ratio']:.1f}: "
                f"ASR={row['asr']:.1%}, ROC-AUC={row['roc_auc']:.4f}"
            )
    else:
        raise ValueError(f"Unknown defence={mode!r}")
    return base_metrics["presence_acc"]


if __name__ == "__main__":
    main()
