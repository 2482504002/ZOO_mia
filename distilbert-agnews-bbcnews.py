"""
FedMeZO membership inference experiment (full metrics):
- Client training & target samples: AG News
- Adversarial init anchors & aux calibration pool: BBC News (same-domain out-of-source, excluding entertainment)
Standalone: python federatedscope/distilbert-agnews-MIA-zhibiao-bbcnews.py
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
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mia_roc_plotting import (
    compute_roc_auc_and_maybe_plot,
    compute_roc_auc_arrays,
    test_records_to_y_arrays,
)


# BBC aux category mapping, aligned with AG News label semantics:
# 0=World, 1=Sports, 2=Business, 3=Sci/Tech.
# BBC has no category named World; politics is the closest proxy for World.
BBC_AUX_LABEL_MAP = {
    "politics": 0,
    "sport": 1,
    "business": 2,
    "tech": 3,
}

AGNEWS_CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


@dataclass
class Config:
    seed: int = 42
    data_root: str = "../datasets/agnews"
    aux_data_root: str = "/home/zhike/JWH/data/bbc-news"
    aux_exclude_categories: tuple = ("entertainment",)
    model_name: str = "/home/zhike/JWH/model/distilbert-base-uncased/"
    max_length: int = 64
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"

    num_clients: int = 2
    fl_samples: int = 2000
    local_steps: int = 10
    batch_size: int = 1
    lr: float = 1e-5
    zo_eps: float = 1e-3

    attack_samples: int = 500
    target_pool_samples: int = 500
    aux_size_sweep: tuple = (100, 200, 300, 400, 500)
    target_cache_path: str = "outputs/toy_mlp_target_cache_bbc_aux_zhibiao.pt"
    # Split BBC aux attack_pool into disjoint init/calib subsets:
    # - init: adversarial init anchors (attack_pool mode)
    # - calib: threshold calibration (threshold_calib_source="aux_attack_pool")
    aux_attack_pool_calib_fraction: float = 0.3

    round1_server_trials: int = 500
    round1_inject_step: int = 0

    # Round1: binary detection of member presence (single member only)
    member_present_prob: float = 0.5  # probability of member injection per trial

    # Server threshold detection: compute det_score per client, compare to threshold
    # Decision score:
    # - "gap": max_score - second_max_score
    # - "ratio": max_score / mean(scores)
    # - "max": max_score
    # - "norm_mean": mean(scores)
    # - "zmax": (max - mean) / std
    # - "top1_top2": top1 / top2
    # - "iqr_outlier": (max - Q3) / IQR
    # - "gini": score imbalance (Gini)
    # - "entropy": normalized entropy of scores
    server_det_score_mode: str = "gap"

    # Threshold selection:
    # - "fixed": use server_fixed_threshold
    # - "calibrate_on_neg": calibrate threshold from quantile of first threshold_calib_fraction non-member trials
    # - "roc_youden": on first threshold_calib_fraction trials, pick threshold at ROC point maximizing (TPR-FPR)
    # - "online_quantile": online dynamic threshold (historical det_score only, no ground-truth labels)
    server_threshold_mode: str = "online_quantile"
    # Threshold calibration data source:
    # - "trial_prefix": calibrate on round1 trial prefix (original logic)
    # - "aux_attack_pool": run separate trials with attack_pool sample injection for calibration
    threshold_calib_source: str = "aux_attack_pool"
    server_fixed_threshold: float = 0.0
    desired_fpr: float = 0.05  # under calibrate_on_neg: target approximate false positive rate
    threshold_calib_fraction: float = 0.5
    threshold_calib_trials: int = 50
    online_warmup: int = 30
    online_window: int = 80
    online_alpha: float = 0.12
    online_min_threshold: float = 10000
    online_max_threshold: float = 1e12
    online_update_with_neg_only: bool = True
    online_auto_clip_by_mode: bool = True
    compare_metric_modes_enable: bool = True

    roc_plot_path: str = ""
    roc_scores_cache_path: str = "outputs/mia_roc_scores_distilbert_agnews_bbc_aux.npz"
    roc_dataset_label: str = "AG News target + BBC aux"
    gap_plot_path: str = "outputs/gap-client_norm_strip_distilbert_agnews_bbc_aux.pdf"
    client_norm_plot_path: str = "outputs/client_norm_strip_distilbert_agnews_bbc_aux.pdf"
    client_norm_plot_log_x: bool = True
    extra_metric_plot_enable: bool = True
    extra_metric_plot_dir: str = "outputs/det_metric_strips_distilbert_agnews_bbc_aux"
    extra_metric_plot_log_x: bool = True
    metric_panel_plot_enable: bool = True
    metric_panel_plot_path: str = "outputs/det_metric_panel_distilbert_agnews_bbc_aux.pdf"
    metric_panel_plot_log_x: bool = True

    adv_init_use: bool = True
    adv_init_steps: int =  200
    adv_init_lr: float = 1e-3
    adv_init_w_target: float = 1.0
    adv_init_w_anchor: float = 0.4
    adv_init_anchor_power: float = 2.0
    adv_init_w_anchor_max: float = 0.12
    adv_init_anchors_per_client: int = 50
    # Adversarial init anchor source: "clients"=per-client DataLoader (default); "attack_pool"=attack pool attack_batches
    adv_init_anchor_source: str = "attack_pool"
    adv_init_log_every: int = 10
    adv_init_bundle_path: str = "outputs/toy_mlp_adv_init_bundle_bbc_aux_zhibiao.pt"
    adv_init_bundle_use: bool = True  # True: skip adversarial init if cache hit


class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_pick(texts, labels, target_n, seed):
    rng = random.Random(seed)
    label_to_idx = {}
    for i, y in enumerate(labels):
        label_to_idx.setdefault(int(y), []).append(i)
    if not label_to_idx:
        return [], []

    for y in label_to_idx:
        rng.shuffle(label_to_idx[y])

    classes = sorted(label_to_idx.keys())
    n_class = len(classes)
    base = target_n // n_class
    rem = target_n % n_class

    take_per_class = {y: min(base, len(label_to_idx[y])) for y in classes}
    need_extra = rem + (target_n - sum(take_per_class.values()) - rem)
    while need_extra > 0:
        progressed = False
        for y in classes:
            if need_extra <= 0:
                break
            if take_per_class[y] < len(label_to_idx[y]):
                take_per_class[y] += 1
                need_extra -= 1
                progressed = True
        if not progressed:
            break

    picked = []
    for y in classes:
        picked.extend(label_to_idx[y][: take_per_class[y]])
    rng.shuffle(picked)
    return [texts[i] for i in picked], [int(labels[i]) for i in picked]


def load_agnews_data(cfg, target_n=None):
    """Load AG News (client training segment, or contiguous segment including target holdout)."""
    if target_n is None:
        target_n = cfg.fl_samples * cfg.num_clients

    local_data_dir = os.path.join(cfg.data_root, "data")
    if os.path.isdir(local_data_dir):
        parquet_files = sorted(
            [
                os.path.join(local_data_dir, f)
                for f in os.listdir(local_data_dir)
                if f.endswith(".parquet")
            ]
        )
        if parquet_files:
            frames = [pd.read_parquet(fp) for fp in parquet_files]
            df = pd.concat(frames, ignore_index=True)
            if "text" in df.columns and "label" in df.columns:
                texts = df["text"].astype(str).tolist()
                labels = (
                    pd.to_numeric(df["label"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .tolist()
                )
                n = min(target_n, len(texts))
                if n < target_n:
                    print(f"Warning: local AG News has only {n}  samples, fewer than required  {target_n}  samples.")
                return balanced_pick(texts, labels, n, cfg.seed)
            print(f"Warning: local AG News missing text/label columns, actual columns: {list(df.columns)}")

    try:
        from datasets import load_dataset

        dataset = load_dataset("ag_news")
        train_data = dataset["train"]
        n = min(target_n, len(train_data))
        train_texts = [item["text"] for item in train_data.select(range(n))]
        train_labels = [item["label"] for item in train_data.select(range(n))]
        return balanced_pick(train_texts, train_labels, n, cfg.seed)
    except Exception:
        print("Could not load AG News from local or datasets, using synthetic data...")
        texts = [f"This is a sample news article number {i}" for i in range(target_n)]
        labels = [i % 4 for i in range(target_n)]
        return texts, labels


def load_bbc_aux_data(cfg, target_n=None):
    """Load out-of-source aux pool from BBC News, excluding entertainment category."""
    if target_n is None:
        target_n = cfg.attack_samples

    exclude = {c.strip().lower() for c in cfg.aux_exclude_categories}
    root = cfg.aux_data_root.rstrip(os.sep)
    texts, labels = [], []

    for fname in ("train.jsonl", "test.jsonl"):
        path = os.path.join(root, fname)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                cat = (item.get("label_text") or item.get("category") or "").strip().lower()
                if not cat or cat in exclude:
                    continue
                if cat not in BBC_AUX_LABEL_MAP:
                    print(f"Warning: skipping unknown BBC category {cat!r}")
                    continue
                texts.append(str(item["text"]))
                labels.append(BBC_AUX_LABEL_MAP[cat])

    if not texts:
        csv_path = os.path.join(root, "bbc-text.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            if "text" in df.columns and "category" in df.columns:
                for _, row in df.iterrows():
                    cat = str(row["category"]).strip().lower()
                    if cat in exclude or cat not in BBC_AUX_LABEL_MAP:
                        continue
                    texts.append(str(row["text"]))
                    labels.append(BBC_AUX_LABEL_MAP[cat])

    if not texts:
        raise RuntimeError(
            f"Could not load BBC News aux data from {root} (excluded {sorted(exclude)})."
        )

    n_pick = min(target_n, len(texts))
    if n_pick < target_n:
        print(f"Warning: BBC News has only {len(texts)} samples, fewer than required aux pool {target_n} samples.")

    picked_texts, picked_labels = balanced_pick(texts, labels, n_pick, cfg.seed + 9001)
    kept_cats = {k: v for k, v in BBC_AUX_LABEL_MAP.items() if k not in exclude}
    print(
        f"Aux set BBC News: path={root}, excluded categories={sorted(exclude)}, "
        f"kept categories={kept_cats}, sampled={len(picked_texts)} samples"
    )
    return picked_texts, picked_labels


def get_model_and_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=4,
        ignore_mismatched_sizes=True,
    ).to(cfg.device)

    for name, param in model.named_parameters():
        if "classifier" not in name and "score" not in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
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
    outputs = model(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
    )
    return F.cross_entropy(outputs.logits, batch["label"].to(model.device))


def mezo_step(model, batch, params, theta, z, cfg):
    theta_orig = theta.clone()
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
    flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
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
                "label": batch["label"],
            }
            if tid is not None and b["input_ids"].shape == tid.shape:
                if torch.equal(b["input_ids"].cpu(), tid):
                    continue
            anchors.append(b)
            count += 1
    return anchors


def collect_anchor_batches_from_attack_pool(attack_batches, target_batch, cfg, seed):
    """
    Sample anchor batches from attack pool list (skip items with same input_ids as target).
    Total count matches client mode: num_clients * adv_init_anchors_per_client.
    """
    rng = random.Random(seed)
    tid = target_batch["input_ids"].detach().cpu() if target_batch is not None else None
    candidates = []
    for b in attack_batches:
        bdict = {
            "input_ids": b["input_ids"],
            "attention_mask": b["attention_mask"],
            "label": b["label"],
        }
        if tid is not None and bdict["input_ids"].shape == tid.shape:
            if torch.equal(bdict["input_ids"].cpu(), tid):
                continue
        candidates.append(bdict)

    total_want = cfg.num_clients * cfg.adv_init_anchors_per_client
    if not candidates:
        print("[Adversarial init] Attack pool anchors: no candidates available (may contain only target).")
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
        print("[Adversarial init] No target sample, skipping.")
        return
    params = get_trainable_params(model)
    if not params:
        print("[Adversarial init] No trainable parameters, skipping.")
        return

    model.train()
    opt = torch.optim.Adam(params, lr=cfg.adv_init_lr)
    n_anchor = max(1, len(anchor_batches))
    model.zero_grad(set_to_none=True)

    n_t0, mean_a0, max_a0, L0 = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
    print(
        f"[Adversarial init] Start step 0/{cfg.adv_init_steps}: "
        f"||∇L_target||={n_t0:.6f}, anchor mean||∇L||={mean_a0:.6f}, max||∇L||={max_a0:.6f}, "
        f"L={L0:.6f} (=-w_t||g_tgt||+(w_a/n)Σ||g||^{cfg.adv_init_anchor_power:g}"
        f"+{'w_max·max||g||' if cfg.adv_init_w_anchor_max > 0 else '0'}) "
        f"(num anchors={len(anchor_batches)})"
    )

    log_every = max(0, int(cfg.adv_init_log_every))
    p = float(cfg.adv_init_anchor_power)
    for step in tqdm(range(cfg.adv_init_steps), desc="Adversarial init (classifier head)"):
        opt.zero_grad(set_to_none=True)
        n_t = _grad_norm_classifier(model, target_batch)
        loss = -cfg.adv_init_w_target * n_t
        anchor_norms = []
        for ab in anchor_batches:
            na = _grad_norm_classifier(model, ab)
            anchor_norms.append(na)
            loss = loss + (cfg.adv_init_w_anchor / n_anchor) * na.pow(p)
        if cfg.adv_init_w_anchor_max > 0 and anchor_norms:
            loss = loss + cfg.adv_init_w_anchor_max * torch.stack(anchor_norms).max()
        loss.backward()
        opt.step()

        done = step + 1
        if log_every > 0 and (done % log_every == 0 or done == cfg.adv_init_steps):
            nt, ma, mxa, Lm = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
            print(
                f"[Adversarial init]  step {done:4d}/{cfg.adv_init_steps}: "
                f"||∇L_target||={nt:.6f}, anchor mean||∇L||={ma:.6f}, max||∇L||={mxa:.6f}, L={Lm:.6f}"
            )

    model.eval()
    model.zero_grad(set_to_none=True)
    n_tf, mean_af, max_af, Lf = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
    rel = (n_tf - n_t0) / (n_t0 + 1e-12) * 100.0
    print(
        f"[Adversarial init] Done: ||∇L_target|| {n_t0:.6f} -> {n_tf:.6f} "
        f"(relative {rel:+.2f}%), anchor mean||∇L|| {mean_a0:.6f} -> {mean_af:.6f}, "
        f"max||∇L|| {max_a0:.6f} -> {max_af:.6f}, L {L0:.6f} -> {Lf:.6f}"
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
    """Given per-client scores, compute monotonic score for member presence detection."""
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


def server_decide_member_presence(scores, cfg, threshold: float):
    """Return (pred_member_present: bool, pred_member_idx or None)."""
    det_score = compute_det_score(scores, cfg.server_det_score_mode)
    pred_member_present = det_score > float(threshold)
    if pred_member_present:
        pred_member_idx = int(np.argmax(scores))
    else:
        pred_member_idx = None
    return pred_member_present, pred_member_idx, det_score


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


def _batch_int_label(batch):
    return int(batch["label"].reshape(-1)[0].item())


def _group_target_batches_by_label(batches):
    grouped = {c: [] for c in AGNEWS_CLASS_NAMES}
    for b in batches:
        grouped.setdefault(_batch_int_label(b), []).append(b)
    return grouped


def _run_adv_init_for_target(model, target_batch, attack_init_batches, client_loaders, cfg):
    if not cfg.adv_init_use:
        return
    src = (cfg.adv_init_anchor_source or "clients").strip().lower()
    if src == "attack_pool":
        anchor_batches = collect_anchor_batches_from_attack_pool(
            attack_init_batches, target_batch, cfg, seed=cfg.seed + 7001
        )
    elif src == "clients":
        anchor_batches = collect_anchor_batches(
            client_loaders, target_batch, cfg, seed=cfg.seed + 7001
        )
    else:
        anchor_batches = collect_anchor_batches(
            client_loaders, target_batch, cfg, seed=cfg.seed + 7001
        )
    adversarial_sharpness_init(model, target_batch, anchor_batches, cfg)


def run_presence_detection_eval(
    model,
    client_loaders,
    target_batch,
    attack_calib_batches,
    cfg,
    *,
    trial_rng_seed: int,
    log_each_trial: bool = False,
    roc_plot_path: str = "",
    roc_scores_cache_path: str = "",
    roc_dataset_label: str = "",
    do_plots: bool = False,
):
    """Run Round1 trials, calibrate threshold, return evaluated_records and metrics."""
    r1_rng = random.Random(trial_rng_seed)
    max_s = cfg.local_steps - 1
    eff_inject = int(np.clip(cfg.round1_inject_step, 0, max_s))

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
        if log_each_trial:
            print(
                f"Round1 trial {trial+1}: true_present={rec['member_present']}, "
                f"true_member={rec['true_member']}, det_score={rec['det_score']:.4f}, "
                f"scores={[round(s, 4) for s in rec['scores']]}"
            )

    threshold = float("nan")
    calib_source = "trial_prefix"
    calib_end = 0
    calib_records = []

    if cfg.server_threshold_mode == "online_quantile":
        calib_source = "online_stream"
        clip_min, clip_max = get_online_clip_bounds(cfg, cfg.server_det_score_mode)
        if log_each_trial:
            print(
                f"[5] Online dynamic threshold: mode=online_quantile, warmup={cfg.online_warmup}, "
                f"window={cfg.online_window}, alpha={cfg.online_alpha}, "
                f"clip=[{clip_min}, {clip_max}], "
                f"neg_only_update={cfg.online_update_with_neg_only}"
            )
    else:
        calib_source = (cfg.threshold_calib_source or "trial_prefix").strip().lower()
        calib_end = int(
            max(1, min(cfg.round1_server_trials, cfg.round1_server_trials * cfg.threshold_calib_fraction))
        )
        if calib_source == "trial_prefix":
            calib_records = trial_records[:calib_end]
            if log_each_trial:
                print(f"[5] Threshold calibration source: trial_prefix (n={len(calib_records)})")
        elif calib_source == "aux_attack_pool":
            aux_pool = []
            tgt_ids = target_batch["input_ids"].detach().cpu() if target_batch is not None else None
            for b in attack_calib_batches:
                if tgt_ids is not None and b["input_ids"].shape == tgt_ids.shape:
                    if torch.equal(b["input_ids"].detach().cpu(), tgt_ids):
                        continue
                aux_pool.append(b)
            if not aux_pool:
                if log_each_trial:
                    print("[5] Warning: aux_attack_pool is empty, falling back to trial_prefix calibration.")
                calib_records = trial_records[:calib_end]
                calib_source = "trial_prefix"
            else:
                calib_n = int(max(1, cfg.threshold_calib_trials))
                calib_rng = random.Random(cfg.seed + 11009)
                calib_records = []
                for c in range(calib_n):
                    c_member_present = calib_rng.random() < cfg.member_present_prob
                    c_true_member = calib_rng.randrange(cfg.num_clients) if c_member_present else None
                    injected_batch = aux_pool[calib_rng.randrange(len(aux_pool))]
                    calib_records.append(
                        simulate_round1_trial_record(
                            model,
                            client_loaders,
                            injected_batch,
                            cfg,
                            trial_seed=cfg.seed + 150000 + c * 1000,
                            member_present=c_member_present,
                            true_member=c_true_member,
                            inject_step=eff_inject,
                        )
                    )
                calib_end = len(calib_records)
                if log_each_trial:
                    print(
                        f"[5] Threshold calibration source: aux_attack_pool "
                        f"(calib_trials={calib_end}, aux_candidates={len(aux_pool)})"
                    )
        else:
            raise ValueError(f"Unknown threshold_calib_source: {cfg.threshold_calib_source!r}")

        if cfg.server_threshold_mode == "fixed":
            threshold = float(cfg.server_fixed_threshold)
            if log_each_trial:
                print(f"[5] Using fixed threshold: threshold={threshold}")
        elif cfg.server_threshold_mode == "calibrate_on_neg":
            neg_det_scores = [r["det_score"] for r in calib_records if not r["member_present"]]
            if len(neg_det_scores) < 1:
                threshold = float(cfg.server_fixed_threshold)
                if log_each_trial:
                    print(f"[5] Calibration failed (no negative trials), falling back to fixed threshold={threshold}")
            else:
                q = 1.0 - float(cfg.desired_fpr)
                threshold = float(np.quantile(neg_det_scores, q))
                if log_each_trial:
                    print(
                        f"[5] Calibrated threshold: calib_end={calib_end}, neg={len(neg_det_scores)}, "
                        f"desired_fpr={cfg.desired_fpr} => threshold={threshold:.6f}"
                    )
        elif cfg.server_threshold_mode == "roc_youden":
            y_true_calib = np.array([1 if r["member_present"] else 0 for r in calib_records], dtype=np.int32)
            y_score_calib = np.array([r["det_score"] for r in calib_records], dtype=np.float64)
            if len(np.unique(y_true_calib)) < 2:
                threshold = float(cfg.server_fixed_threshold)
            else:
                try:
                    from sklearn.metrics import roc_curve

                    fpr_calib, tpr_calib, thr_calib = roc_curve(y_true_calib, y_score_calib)
                    valid = np.isfinite(thr_calib)
                    if np.any(valid):
                        youden = tpr_calib[valid] - fpr_calib[valid]
                        threshold = float(thr_calib[valid][int(np.argmax(youden))])
                    else:
                        threshold = float(cfg.server_fixed_threshold)
                except ImportError:
                    threshold = float(cfg.server_fixed_threshold)
        elif cfg.server_threshold_mode != "online_quantile":
            raise ValueError(f"Unknown server_threshold_mode: {cfg.server_threshold_mode!r}")

    if cfg.server_threshold_mode == "online_quantile":
        test_records = trial_records
    elif calib_source == "trial_prefix":
        test_records = trial_records[calib_end:]
    else:
        test_records = trial_records

    TP = TN = FP = FN = 0
    member_idx_correct = 0
    member_idx_pred_cnt = 0
    online_thresholds = []
    online_history = []
    warmup = int(max(0, cfg.online_warmup))
    window = int(max(1, cfg.online_window))
    alpha = float(np.clip(cfg.online_alpha, 1e-6, 1.0 - 1e-6))
    q_online = 1.0 - alpha
    min_thr, max_thr = get_online_clip_bounds(cfg, cfg.server_det_score_mode)
    evaluated_records = []

    for idx, r in enumerate(test_records):
        if cfg.server_threshold_mode == "online_quantile":
            score = float(r["det_score"])
            if len(online_history) < warmup:
                online_history.append(score)
                continue
            hist = online_history[-window:]
            dynamic_thr = float(np.quantile(np.asarray(hist, dtype=np.float64), q_online))
            dynamic_thr = min(max(dynamic_thr, min_thr), max_thr)
            pred_present = score > dynamic_thr
            online_thresholds.append(dynamic_thr)
            if (not cfg.online_update_with_neg_only) or (not pred_present):
                online_history.append(score)
            evaluated_records.append(r)
        else:
            pred_present = r["det_score"] > threshold
            evaluated_records.append(r)

        true_present = bool(r["member_present"])
        if pred_present and true_present:
            TP += 1
        elif (not pred_present) and (not true_present):
            TN += 1
        elif pred_present and (not true_present):
            FP += 1
        else:
            FN += 1
        if true_present and pred_present:
            member_idx_pred_cnt += 1
            if r["argmax_idx"] == r["true_member"]:
                member_idx_correct += 1

    total = len(evaluated_records)
    presence_acc = (TP + TN) / total if total else float("nan")
    tpr = TP / (TP + FN) if (TP + FN) else float("nan")
    fpr = FP / (FP + TN) if (FP + TN) else float("nan")
    miss_rate = FN / (TP + FN) if (TP + FN) else float("nan")
    precision = TP / (TP + FP) if (TP + FP) else float("nan")
    member_idx_acc = member_idx_correct / member_idx_pred_cnt if member_idx_pred_cnt else float("nan")

    roc_auc = float("nan")
    if evaluated_records:
        y_true, y_score = test_records_to_y_arrays(evaluated_records)
        if y_true is not None and len(np.unique(y_true)) >= 2:
            try:
                roc_auc, _, _ = compute_roc_auc_arrays(y_true, y_score)
            except ImportError:
                roc_auc = float("nan")
        if do_plots:
            compute_roc_auc_and_maybe_plot(
                evaluated_records,
                roc_plot_path,
                cfg.model_name,
                roc_dataset_label,
                scores_cache_path=roc_scores_cache_path,
            )

    return {
        "evaluated_records": evaluated_records,
        "roc_auc": roc_auc,
        "presence_acc": presence_acc,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "miss_rate": miss_rate,
        "member_idx_acc": member_idx_acc,
        "member_idx_correct": member_idx_correct,
        "member_idx_pred_cnt": member_idx_pred_cnt,
        "threshold": threshold,
        "total": total,
        "test_records": test_records,
        "online_thresholds": online_thresholds,
    }


def report_per_class_target_auc(
    model_baseline,
    target_candidate_batches,
    attack_init_batches,
    attack_calib_batches,
    client_loaders,
    cfg,
    overall_auc: float = float("nan"),
):
    """For each AG News target class, select target, run adversarial init, and report ROC-AUC."""
    grouped = _group_target_batches_by_label(target_candidate_batches)
    rows = []
    print("\n[6] Per-class ROC-AUC evaluation by AG News target class (BBC out-of-source aux)…")

    for cls in sorted(AGNEWS_CLASS_NAMES.keys()):
        name = AGNEWS_CLASS_NAMES[cls]
        pool = grouped.get(cls) or []
        if not pool:
            print(f"  Class {cls} ({name}): candidate pool empty, skipping")
            rows.append(
                {
                    "class_id": cls,
                    "class_name": name,
                    "pool_size": 0,
                    "roc_auc": None,
                    "presence_acc": None,
                    "test_trials": 0,
                }
            )
            continue

        print(f"\n  --- Class {cls} ({name}), candidates {len(pool)} samples ---")
        model_cls = copy.deepcopy(model_baseline)
        target_cls = select_target_sample(model_cls, pool, cfg)
        if cfg.adv_init_use:
            _run_adv_init_for_target(
                model_cls, target_cls, attack_init_batches, client_loaders, cfg
            )

        res = run_presence_detection_eval(
            model_cls,
            client_loaders,
            target_cls,
            attack_calib_batches,
            cfg,
            trial_rng_seed=cfg.seed + 1009 + cls * 100000,
            log_each_trial=False,
            do_plots=False,
        )
        auc = float(res["roc_auc"])
        rows.append(
            {
                "class_id": cls,
                "class_name": name,
                "pool_size": len(pool),
                "roc_auc": None if not np.isfinite(auc) else round(auc, 6),
                "presence_acc": None
                if not np.isfinite(res["presence_acc"])
                else round(float(res["presence_acc"]), 6),
                "test_trials": int(res["total"]),
                "target_label": _batch_int_label(target_cls),
            }
        )
        print(
            f"  Class {cls} ({name}): AUC={auc:.4f}, "
            f"presence_acc={res['presence_acc']:.1%}, test_trials={res['total']}"
        )

    valid_aucs = [r["roc_auc"] for r in rows if r["roc_auc"] is not None]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    aucs_only = [r["roc_auc"] for r in rows if r["roc_auc"] is not None]
    if len(aucs_only) >= 2 and np.isfinite(overall_auc):
        spread = max(aucs_only) - min(aucs_only)
    else:
        spread = float("nan")

    print("\n" + "=" * 60)
    print("[6] ========== Per-class ROC-AUC summary (AG News target × BBC aux) ==========")
    print(f"{'Class':<12} {'ID':>3} {'Candidates':>6} {'AUC':>8} {'presence_acc':>14}")
    print("-" * 60)
    for r in rows:
        auc_s = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "N/A"
        acc_s = (
            f"{r['presence_acc']:.1%}"
            if r["presence_acc"] is not None
            else "N/A"
        )
        print(
            f"{r['class_name']:<12} {r['class_id']:>3} {r['pool_size']:>6} "
            f"{auc_s:>8} {acc_s:>14}"
        )
    print("-" * 60)
    if np.isfinite(overall_auc):
        print(f"Overall AUC ([5] global target): {overall_auc:.4f}")
    if np.isfinite(mean_auc):
        print(f"Mean AUC across four classes: {mean_auc:.4f}")
    if np.isfinite(spread):
        print(f"Four-class AUC spread (max-min): {spread:.4f}")
        worst = min((r for r in rows if r["roc_auc"] is not None), key=lambda x: x["roc_auc"])
        best = max((r for r in rows if r["roc_auc"] is not None), key=lambda x: x["roc_auc"])
        print(
            f"Lowest: {worst['class_name']} (AUC={worst['roc_auc']:.4f}); "
            f"Highest: {best['class_name']} (AUC={best['roc_auc']:.4f})"
        )
    print("=" * 60)

    save_path = (cfg.per_class_auc_save_path or "").strip()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        payload = {
            "aux_dataset": "bbc-news",
            "target_dataset": "agnews",
            "overall_auc": None if not np.isfinite(overall_auc) else round(overall_auc, 6),
            "mean_per_class_auc": None if not np.isfinite(mean_auc) else round(mean_auc, 6),
            "per_class": rows,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[6] Per-class AUC saved: {save_path}")

    return rows


def compute_true_gradient(model, batch):
    model.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    grad = []
    for p in get_trainable_params(model):
        grad.append(p.grad.detach().flatten())
    return torch.cat(grad).float()


def plot_gap_strip(test_records, out_path: str, *, log_x: bool = True) -> bool:
    out_path = (out_path or "").strip()
    if not out_path or not test_records:
        return False

    pos_gap = []
    neg_gap = []
    for r in test_records:
        g = float(r.get("det_score", 0.0))
        if bool(r.get("member_present")):
            pos_gap.append(g)
        else:
            neg_gap.append(g)

    if (len(pos_gap) + len(neg_gap)) == 0:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[5] GAP distribution plot: matplotlib not installed, skipping plot.")
        return False

    rng = np.random.default_rng(20260413)
    y_pos = np.ones(len(pos_gap), dtype=np.float64) + rng.uniform(-0.08, 0.08, size=len(pos_gap))
    y_neg = np.zeros(len(neg_gap), dtype=np.float64) + rng.uniform(-0.08, 0.08, size=len(neg_gap))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    if neg_gap:
        ax.scatter(neg_gap, y_neg, s=12, c="#1f77b4", alpha=0.65, label="No member")
    if pos_gap:
        ax.scatter(pos_gap, y_pos, s=12, c="#d62728", alpha=0.65, label="Member present")
    if log_x:
        ax.set_xscale("log")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No member", "Member present"])
    ax.set_xlabel("Trial GAP Score (max - second_max)")
    ax.set_title("GAP Distribution by Trial Label")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[5] GAP distribution plot saved: {out_path}")
    return True


def plot_client_norm_strip(test_records, out_path: str, *, log_x: bool = True) -> bool:
    out_path = (out_path or "").strip()
    if not out_path or not test_records:
        return False

    pos_scores = []
    neg_scores = []
    for r in test_records:
        scores = r.get("scores") or []
        if bool(r.get("member_present")):
            pos_scores.extend(float(s) for s in scores)
        else:
            neg_scores.extend(float(s) for s in scores)

    if (len(pos_scores) + len(neg_scores)) == 0:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[5] Client norm plot: matplotlib not installed, skipping plot.")
        return False

    rng = np.random.default_rng(20260413)
    y_pos = np.ones(len(pos_scores), dtype=np.float64) + rng.uniform(-0.08, 0.08, size=len(pos_scores))
    y_neg = np.zeros(len(neg_scores), dtype=np.float64) + rng.uniform(-0.08, 0.08, size=len(neg_scores))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    if neg_scores:
        ax.scatter(neg_scores, y_neg, s=12, c="#1f77b4", alpha=0.65, label="No member")
    if pos_scores:
        ax.scatter(pos_scores, y_pos, s=12, c="#d62728", alpha=0.65, label="Member present")
    if log_x:
        ax.set_xscale("log")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No member", "Member present"])
    ax.set_xlabel("Client Gradient Norm Score")
    ax.set_title("Client Norm Distribution by Trial Label")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[5] Client norm distribution plot saved: {out_path}")
    return True


def _plot_binary_strip(pos_vals, neg_vals, out_path: str, title: str, xlabel: str, log_x: bool) -> bool:
    if (len(pos_vals) + len(neg_vals)) == 0:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    vals = [float(v) for v in (pos_vals + neg_vals) if np.isfinite(v)]
    if not vals:
        return False
    use_log = bool(log_x and min(vals) > 0.0)

    rng = np.random.default_rng(20260413)
    y_pos = np.ones(len(pos_vals), dtype=np.float64) + rng.uniform(-0.08, 0.08, size=len(pos_vals))
    y_neg = np.zeros(len(neg_vals), dtype=np.float64) + rng.uniform(-0.08, 0.08, size=len(neg_vals))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    if neg_vals:
        ax.scatter(neg_vals, y_neg, s=12, c="#1f77b4", alpha=0.65, label="No member")
    if pos_vals:
        ax.scatter(pos_vals, y_pos, s=12, c="#d62728", alpha=0.65, label="Member present")
    if use_log:
        ax.set_xscale("log")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No member", "Member present"])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_extra_metric_strips(test_records, out_dir: str, *, log_x: bool = True) -> int:
    out_dir = (out_dir or "").strip()
    if not out_dir or not test_records:
        return 0
    metric_defs = [
        ("ratio", "Ratio Distribution", "Trial Ratio Score (max / mean)"),
        ("zmax", "Zmax Distribution", "Trial Zmax Score ((max-mean)/std)"),
        ("top1_top2", "Top1/Top2 Distribution", "Trial Top1/Top2 Score"),
        ("iqr_outlier", "IQR Outlier Distribution", "Trial IQR Outlier Score ((max-Q3)/IQR)"),
        ("gini", "Gini Distribution", "Trial Gini Score"),
        ("entropy", "Entropy Distribution", "Trial Entropy Score"),
    ]
    saved = 0
    os.makedirs(out_dir, exist_ok=True)
    for metric, title, xlabel in metric_defs:
        pos_vals = []
        neg_vals = []
        for r in test_records:
            scores = r.get("scores") or []
            v = compute_det_score(scores, metric)
            if bool(r.get("member_present")):
                pos_vals.append(float(v))
            else:
                neg_vals.append(float(v))
        out_path = os.path.join(out_dir, f"{metric}_strip.pdf")
        ok = _plot_binary_strip(pos_vals, neg_vals, out_path, title, xlabel, log_x=log_x)
        if ok:
            saved += 1
            print(f"[5] Metric distribution plot saved ({metric}): {out_path}")
    return saved


def _metric_values_for_records(test_records, metric: str):
    pos_vals = []
    neg_vals = []
    if metric == "norm":
        for r in test_records:
            scores = r.get("scores") or []
            if bool(r.get("member_present")):
                pos_vals.extend(float(s) for s in scores)
            else:
                neg_vals.extend(float(s) for s in scores)
        return pos_vals, neg_vals

    for r in test_records:
        scores = r.get("scores") or []
        v = compute_det_score(scores, metric)
        if bool(r.get("member_present")):
            pos_vals.append(float(v))
        else:
            neg_vals.append(float(v))
    return pos_vals, neg_vals


def plot_metric_panel(test_records, out_path: str, *, log_x: bool = True) -> bool:
    out_path = (out_path or "").strip()
    if not out_path or not test_records:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[5] Six-metric panel plot: matplotlib not installed, skipping plot.")
        return False

    metric_defs = [
        ("norm", "Norm", "Client Gradient Norm"),
        ("gap", "GAP", "GAP (max - second_max)"),
        ("ratio", "Ratio", "Ratio (max / mean)"),
        ("top1_top2", "Top1/Top2", "Top1/Top2"),
    ]

    fig, axes = plt.subplots(len(metric_defs), 1, figsize=(9, 5))
    if len(metric_defs) == 1:
        axes = [axes]
    rng = np.random.default_rng(20260413)

    for i, (metric, title, xlabel) in enumerate(metric_defs):
        ax = axes[i]
        pos_vals, neg_vals = _metric_values_for_records(test_records, metric)
        vals = [float(v) for v in (pos_vals + neg_vals) if np.isfinite(v)]
        if not vals:
            ax.set_title(f"{title} (empty)")
            ax.set_yticks([])
            continue
        use_log = bool(log_x and min(vals) > 0.0)
        y_pos = rng.uniform(-0.08, 0.08, size=len(pos_vals))
        y_neg = rng.uniform(-0.08, 0.08, size=len(neg_vals))

        if neg_vals:
            ax.scatter(neg_vals, y_neg, s=11, c="#1f77b4", alpha=0.65, label="No member")
        if pos_vals:
            ax.scatter(pos_vals, y_pos, s=11, c="#d62728", alpha=0.65, label="Member present")

        if use_log:
            ax.set_xscale("log")
        ax.set_ylim(-0.12, 0.12)
        ax.set_yticks([])
        ax.set_ylabel(title, rotation=0, labelpad=30, va="center")
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)

        if i == 0:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)

    fig.suptitle("Detection Metric Distributions (Mixed Red/Blue Strips)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[5] Six-metric panel plot saved: {out_path}")
    return True


def evaluate_online_mode_from_records(test_records, mode: str, cfg):
    if not test_records:
        return None
    warmup = int(max(0, cfg.online_warmup))
    window = int(max(1, cfg.online_window))
    alpha = float(np.clip(cfg.online_alpha, 1e-6, 1.0 - 1e-6))
    q = 1.0 - alpha
    min_thr, max_thr = get_online_clip_bounds(cfg, mode)

    stream = []
    if mode == "norm_all":
        for r in test_records:
            scores = r.get("scores") or []
            true_present = bool(r.get("member_present"))
            for s in scores:
                stream.append((float(s), true_present))
    else:
        for r in test_records:
            score = float(compute_det_score((r.get("scores") or []), mode))
            true_present = bool(r.get("member_present"))
            stream.append((score, true_present))

    hist = []
    thr_list = []
    TP = TN = FP = FN = 0
    eval_n = 0
    for score, true_present in stream:
        if len(hist) < warmup:
            hist.append(score)
            continue
        base = hist[-window:]
        thr = float(np.quantile(np.asarray(base, dtype=np.float64), q))
        thr = min(max(thr, min_thr), max_thr)
        pred = (score > thr)
        if pred and true_present:
            TP += 1
        elif (not pred) and (not true_present):
            TN += 1
        elif pred and (not true_present):
            FP += 1
        else:
            FN += 1
        eval_n += 1
        thr_list.append(thr)
        if (not cfg.online_update_with_neg_only) or (not pred):
            hist.append(score)

    acc = (TP + TN) / eval_n if eval_n else float("nan")
    precision = TP / (TP + FP) if (TP + FP) else float("nan")
    tpr = TP / (TP + FN) if (TP + FN) else float("nan")
    fpr = FP / (FP + TN) if (FP + TN) else float("nan")
    miss_rate = FN / (TP + FN) if (TP + FN) else float("nan")
    thr_mean = float(np.mean(thr_list)) if thr_list else float("nan")
    thr_last = float(thr_list[-1]) if thr_list else float("nan")
    return {
        "mode": mode,
        "eval_n": eval_n,
        "acc": acc,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "miss_rate": miss_rate,
        "thr_mean": thr_mean,
        "thr_last": thr_last,
    }


def select_target_sample(model, samples, cfg):
    if len(samples) == 0:
        print("Warning: attack pool is empty, cannot select target sample.")
        return None

    best_norm = -1.0
    best_batch = None
    for i in tqdm(range(len(samples)), desc="Selecting target sample"):
        batch = samples[i]
        g = compute_true_gradient(model, batch)
        norm = g.norm().item()
        if norm > best_norm:
            best_norm = norm
            best_batch = batch

    print(f"Target sample gradient norm: {best_norm:.6f}")
    return best_batch


def create_attack_batches(texts, labels, tokenizer, cfg):
    batches = []
    for i in range(len(texts)):
        enc = tokenizer(
            texts[i],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        batches.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": torch.tensor([labels[i]], dtype=torch.long),
            }
        )
    return batches


def create_dataloader(texts, labels, tokenizer, cfg, shuffle=True):
    dataset = AGNewsDataset(texts, labels, tokenizer, cfg.max_length)
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
        "anchor_source": cfg.adv_init_anchor_source,
    }


def _build_step45_meta(cfg, attack_texts, attack_labels, target_texts, target_labels):
    attack_sig = hashlib.sha1(
        json.dumps(
            {"texts": attack_texts, "labels": attack_labels},
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    target_sig = hashlib.sha1(
        json.dumps(
            {"texts": target_texts, "labels": target_labels},
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "num_clients": cfg.num_clients,
        "fl_samples": cfg.fl_samples,
        "attack_samples": cfg.attack_samples,
        "target_dataset": "agnews",
        "aux_dataset": "bbc-news",
        "aux_data_root": cfg.aux_data_root,
        "aux_exclude_categories": list(cfg.aux_exclude_categories),
        "local_steps": cfg.local_steps,
        "grad_stat_mode": "sum",
        "server_attack": "round1_g_sum_l2",
        "zo_eps": cfg.zo_eps,
        "lr": cfg.lr,
        "target_sig": target_sig,
        "attack_sig": attack_sig,
        "aux_attack_pool_calib_fraction": cfg.aux_attack_pool_calib_fraction,
        "adv_init": _adv_init_meta(cfg),
    }


def _target_batch_match(a, b):
    if a is None or b is None:
        return a is None and b is None
    try:
        return torch.equal(a["input_ids"].cpu(), b["input_ids"].cpu()) and torch.equal(
            a["label"].cpu(), b["label"].cpu()
        )
    except Exception:
        return False


def _batch_in_pool(batch, pool):
    if batch is None:
        return False
    for b in pool:
        if _target_batch_match(batch, b):
            return True
    return False


def try_load_adv_init_bundle(model, cfg, cache_meta_45, target_batch):
    if not (cfg.adv_init_use and cfg.adv_init_bundle_use):
        return False
    path = cfg.adv_init_bundle_path
    if not os.path.isfile(path):
        return False
    try:
        obj = torch.load(path, map_location=cfg.device)
    except Exception as e:
        print(f"[Adversarial init cache] Read failed, will retrain: {e}")
        return False
    if not isinstance(obj, dict) or obj.get("meta") != cache_meta_45:
        print("[Adversarial init cache] meta mismatch, will retrain.")
        return False
    tb = obj.get("target_batch")
    if not _target_batch_match(tb, target_batch):
        print("[Adversarial init cache] target sample differs from current, will retrain.")
        return False
    sd = obj.get("state_dict")
    if not isinstance(sd, dict):
        return False
    model.load_state_dict(sd, strict=True)
    print(f"[Adversarial init cache] Loaded model and target: {path}")
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
    print(f"[Adversarial init cache] Saved: {cfg.adv_init_bundle_path}")


def _build_target_select_meta(cfg, target_texts, target_labels):
    target_sig = hashlib.sha1(
        json.dumps(
            {"texts": target_texts, "labels": target_labels},
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "target_pool_samples": cfg.target_pool_samples,
        "target_dataset": "agnews",
        "target_sig": target_sig,
    }


def split_aux_attack_batches(attack_batches, cfg, split_seed):
    split_rng = random.Random(split_seed)
    attack_indices = list(range(len(attack_batches)))
    split_rng.shuffle(attack_indices)
    if len(attack_indices) >= 2:
        calib_n = int(round(len(attack_indices) * float(cfg.aux_attack_pool_calib_fraction)))
        calib_n = max(1, min(len(attack_indices) - 1, calib_n))
    else:
        calib_n = 0
    calib_idx = set(attack_indices[:calib_n])
    attack_init_batches = [
        attack_batches[i] for i in range(len(attack_batches)) if i not in calib_idx
    ]
    attack_calib_batches = [
        attack_batches[i] for i in range(len(attack_batches)) if i in calib_idx
    ]
    return attack_init_batches, attack_calib_batches


def run_adv_init_on_model(model, target_batch, attack_init_batches, client_loaders, cfg):
    if not cfg.adv_init_use:
        return
    src = (cfg.adv_init_anchor_source or "clients").strip().lower()
    if src == "attack_pool":
        anchor_batches = collect_anchor_batches_from_attack_pool(
            attack_init_batches, target_batch, cfg, seed=cfg.seed + 7001
        )
        print(f"[Adversarial init] Anchor source: BBC aux pool ( {len(anchor_batches)} batches)")
    elif src == "clients":
        anchor_batches = collect_anchor_batches(
            client_loaders, target_batch, cfg, seed=cfg.seed + 7001
        )
        print(f"[Adversarial init] Anchor source: per-client training data ( {len(anchor_batches)} batches)")
    else:
        anchor_batches = collect_anchor_batches(
            client_loaders, target_batch, cfg, seed=cfg.seed + 7001
        )
    adversarial_sharpness_init(model, target_batch, anchor_batches, cfg)


def main():
    cfg = Config()
    set_seed(cfg.seed)
    print(f"Device: {cfg.device}")
    print(f"Config: {cfg}")

    print("\n[1] Loading data...")
    print("[1a] Client training set: AG News")
    target_pool_n = int(cfg.target_pool_samples)
    total_ag = cfg.fl_samples * cfg.num_clients + target_pool_n
    ag_texts, ag_labels = load_agnews_data(cfg, target_n=total_ag)
    fl_off = cfg.fl_samples * cfg.num_clients
    fl_texts, fl_labels = ag_texts[:fl_off], ag_labels[:fl_off]
    target_texts, target_labels = ag_texts[fl_off:], ag_labels[fl_off:]
    print("[1b] Target candidate pool: AG News holdout (non-overlapping with clients)")
    print(f"     Target candidate samples: {len(target_texts)}")

    client_data = []
    per_client = cfg.fl_samples
    for i in range(cfg.num_clients):
        start = i * per_client
        end = (i + 1) * per_client
        client_data.append((fl_texts[start:end], fl_labels[start:end]))

    print(f"Number of clients: {cfg.num_clients}, samples per client: {per_client} (AG News)")

    print("\n[2] Loading model...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    model_baseline = copy.deepcopy(model)

    print("\n[3] Creating data loaders...")
    client_loaders = []
    for texts, labels in client_data:
        client_loaders.append(create_dataloader(texts, labels, tokenizer, cfg))

    target_candidate_batches = create_attack_batches(
        target_texts, target_labels, tokenizer, cfg
    )

    target_meta = _build_target_select_meta(cfg, target_texts, target_labels)
    print("\n[4] Selecting target sample (AG News, shared across aux sizes)...")
    target_batch = None
    if os.path.exists(cfg.target_cache_path):
        try:
            target_obj = torch.load(cfg.target_cache_path, map_location="cpu")
            if isinstance(target_obj, dict) and target_obj.get("meta") == target_meta:
                target_batch = target_obj.get("target_batch")
                if _batch_in_pool(target_batch, target_candidate_batches):
                    print(f"Target sample cache hit: {cfg.target_cache_path}")
                else:
                    print("[4] Target cache not in AG News candidate pool, forcing reselection.")
                    target_batch = None
        except Exception as e:
            print(f"Failed to read target sample cache, will recompute: {e}")
    if target_batch is None:
        target_batch = select_target_sample(model_baseline, target_candidate_batches, cfg)
        os.makedirs(os.path.dirname(cfg.target_cache_path) or ".", exist_ok=True)
        torch.save({"meta": target_meta, "target_batch": target_batch}, cfg.target_cache_path)
        print(f"Saved target sample cache: {cfg.target_cache_path}")
    tgt_cls = _batch_int_label(target_batch)
    print(
        f"[4] Global target class: {tgt_cls} ({AGNEWS_CLASS_NAMES.get(tgt_cls, '?')})"
    )

    aux_sizes = [int(n) for n in cfg.aux_size_sweep]
    print(f"\n[5] Aux set size sweep: {aux_sizes}")
    auc_list = []
    sweep_details = []

    if cfg.local_steps < 1:
        print("Skipping [5]: local_steps must be >= 1.")
        print(f"[5] AUC list: {auc_list}")
        return float("nan")

    max_s = cfg.local_steps - 1
    eff_inject = int(np.clip(cfg.round1_inject_step, 0, max_s))
    if cfg.round1_inject_step != eff_inject:
        print(f"Warning: round1_inject_step={cfg.round1_inject_step} clipped to [0,{max_s}]: {eff_inject}")

    for idx, aux_n in enumerate(aux_sizes):
        print("\n" + "=" * 60)
        print(f"[5] Aux set size = {aux_n} ({idx + 1}/{len(aux_sizes)})")
        print("=" * 60)

        print(f"[5.{idx+1}a] Loading BBC aux pool (n={aux_n})...")
        aux_texts, aux_labels = load_bbc_aux_data(cfg, target_n=aux_n)
        attack_batches = create_attack_batches(aux_texts, aux_labels, tokenizer, cfg)
        attack_init_batches, attack_calib_batches = split_aux_attack_batches(
            attack_batches, cfg, split_seed=cfg.seed + 9011 + aux_n
        )
        print(
            f"[5.{idx+1}b] BBC aux pool split: init={len(attack_init_batches)}, "
            f"calib={len(attack_calib_batches)}, total={len(attack_batches)}"
        )

        model = copy.deepcopy(model_baseline)
        if cfg.adv_init_use:
            print(f"\n[5.{idx+1}c] Adversarial init (aux_size={aux_n})...")
            run_adv_init_on_model(
                model, target_batch, attack_init_batches, client_loaders, cfg
            )

        is_last = idx == len(aux_sizes) - 1
        eval_res = run_presence_detection_eval(
            model,
            client_loaders,
            target_batch,
            attack_calib_batches,
            cfg,
            trial_rng_seed=cfg.seed + 1009 + aux_n,
            log_each_trial=False,
            roc_plot_path=(cfg.roc_plot_path or "").strip() if is_last else "",
            roc_scores_cache_path=(cfg.roc_scores_cache_path or "").strip() if is_last else "",
            roc_dataset_label=f"{cfg.roc_dataset_label} (aux={aux_n})",
            do_plots=is_last,
        )
        roc_auc = float(eval_res["roc_auc"])
        auc_list.append(roc_auc)
        sweep_details.append(
            {
                "aux_size": aux_n,
                "roc_auc": None if not np.isfinite(roc_auc) else round(roc_auc, 6),
                "presence_acc": round(float(eval_res["presence_acc"]), 6)
                if np.isfinite(eval_res["presence_acc"])
                else None,
                "test_trials": int(eval_res["total"]),
            }
        )
        auc_str = f"{roc_auc:.4f}" if np.isfinite(roc_auc) else "nan"
        print(
            f"[5.{idx+1}d] aux_size={aux_n}: AUC={auc_str}, "
            f"presence_acc={eval_res['presence_acc']:.1%}, test_trials={eval_res['total']}"
        )

        if is_last:
            evaluated_records = eval_res["evaluated_records"]
            test_records = eval_res["test_records"]
            plot_gap_strip(
                evaluated_records,
                (cfg.gap_plot_path or "").strip(),
                log_x=bool(cfg.client_norm_plot_log_x),
            )
            plot_client_norm_strip(
                evaluated_records,
                (cfg.client_norm_plot_path or "").strip(),
                log_x=bool(cfg.client_norm_plot_log_x),
            )
            if cfg.extra_metric_plot_enable:
                saved_n = plot_extra_metric_strips(
                    evaluated_records,
                    (cfg.extra_metric_plot_dir or "").strip(),
                    log_x=bool(cfg.extra_metric_plot_log_x),
                )
                print(f"[5] Extra metric plot count: {saved_n}")
            if cfg.metric_panel_plot_enable:
                plot_metric_panel(
                    evaluated_records,
                    (cfg.metric_panel_plot_path or "").strip(),
                    log_x=bool(cfg.metric_panel_plot_log_x),
                )
            if cfg.server_threshold_mode == "online_quantile" and cfg.compare_metric_modes_enable:
                compare_modes = [
                    ("norm_all", "norm(all)"),
                    ("gap", "gap"),
                    ("ratio", "ratio"),
                    ("top1_top2", "top1/top2"),
                ]
                print("[5] Four-metric comparison (last aux_size tier, same online threshold policy):")
                for m, m_name in compare_modes:
                    cm = evaluate_online_mode_from_records(test_records, m, cfg)
                    if cm is None:
                        continue
                    cm_min, cm_max = get_online_clip_bounds(cfg, m)
                    print(
                        f"    - {m_name:10s}: acc={cm['acc']:.1%}, precision={cm['precision']:.1%}, "
                        f"TPR={cm['tpr']:.1%}, FPR={cm['fpr']:.1%}, miss={cm['miss_rate']:.1%}, "
                        f"thr_mean={cm['thr_mean']:.3f}, clip=[{cm_min:.3g},{cm_max:.3g}]"
                    )

    print("\n" + "=" * 60)
    print("[5] ========== Aux set size sweep results ==========")
    print(f"{'aux_size':>10} {'AUC':>10}")
    print("-" * 24)
    for aux_n, auc in zip(aux_sizes, auc_list):
        auc_s = f"{auc:.4f}" if np.isfinite(auc) else "nan"
        print(f"{aux_n:>10} {auc_s:>10}")
    print("-" * 24)
    formatted_aucs = [
        round(a, 4) if np.isfinite(a) else None for a in auc_list
    ]
    print(f"aux_sizes = {aux_sizes}")
    print(f"auc_list  = {formatted_aucs}")
    print("=" * 60)

    sweep_path = "outputs/aux_size_sweep_auc_bbc.json"
    os.makedirs(os.path.dirname(sweep_path) or ".", exist_ok=True)
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "aux_sizes": aux_sizes,
                "auc_list": formatted_aucs,
                "details": sweep_details,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[5] Sweep results saved: {sweep_path}")

    return auc_list[-1] if auc_list else float("nan")


if __name__ == "__main__":
    main()
