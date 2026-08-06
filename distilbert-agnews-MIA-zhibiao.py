"""
FedMeZO 玩具成员推理实验（AG News + DistilBERT 分类头）。
原 zoo-toy-mlp-attack-fast.py 备份：本文件可独立运行: python federatedscope/1.py
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
from mia_roc_plotting import compute_roc_auc_and_maybe_plot


@dataclass
class Config:
    seed: int = 42
    data_root: str = "../datasets/agnews"
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
    target_cache_path: str = "outputs/toy_mlp_target_cache.pt"
    # 将 attack_pool 强制划分为互斥 init/calib 子集：
    # - init: 目标样本选择 + 对抗初始化（attack_pool 模式）
    # - calib: 阈值标定（threshold_calib_source="aux_attack_pool"）
    aux_attack_pool_calib_fraction: float = 0.3

    round1_server_trials: int = 500
    round1_inject_step: int = 0

    # Round1：成员“是否存在”做二元检测（不考虑多个成员）
    member_present_prob: float = 0.5  # 每个 trial 有成员注入的概率

    # 服务器阈值判别：对每个客户端得到一个分数 det_score，再与 threshold 比较
    # 决策分数：
    # - "gap": max_score - second_max_score
    # - "ratio": max_score / mean(scores)
    # - "max": max_score
    # - "norm_mean": mean(scores)
    # - "zmax": (max - mean) / std
    # - "top1_top2": top1 / top2
    # - "iqr_outlier": (max - Q3) / IQR
    # - "gini": 分数不均衡度
    # - "entropy": 分数归一化熵
    server_det_score_mode: str = "gap"

    # 阈值选择：
    # - "fixed": 使用 server_fixed_threshold
    # - "calibrate_on_neg": 用前 threshold_calib_fraction 的“无成员 trial”分位数标定阈值
    # - "roc_youden": 在前 threshold_calib_fraction 的 trial 上，取 ROC 中 (TPR-FPR) 最大点对应阈值
    # - "online_quantile": 在线动态阈值（仅使用历史 det_score，不使用真值标签）
    server_threshold_mode: str = "online_quantile"
    # 阈值标定数据来源：
    # - "trial_prefix": 用 round1 trial 前缀做标定（原逻辑）
    # - "aux_attack_pool": 额外跑一组独立 trial，用 attack_pool 样本注入做标定
    threshold_calib_source: str = "aux_attack_pool"
    server_fixed_threshold: float = 0.0
    desired_fpr: float = 0.05  # calibrate_on_neg 模式下：希望 false positive rate 近似到此值
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
    roc_scores_cache_path: str = "outputs/mia_roc_scores_distilbert_agnews.npz"
    roc_dataset_label: str = "AG News"
    gap_plot_path: str = "outputs/gap-client_norm_strip_distilbert_agnews.pdf"
    client_norm_plot_path: str = "outputs/client_norm_strip_distilbert_agnews.pdf"
    client_norm_plot_log_x: bool = True
    extra_metric_plot_enable: bool = True
    extra_metric_plot_dir: str = "outputs/det_metric_strips_distilbert_agnews"
    extra_metric_plot_log_x: bool = True
    metric_panel_plot_enable: bool = True
    metric_panel_plot_path: str = "outputs/det_metric_panel_distilbert_agnews.pdf"
    metric_panel_plot_log_x: bool = True

    adv_init_use: bool = True
    adv_init_steps: int =  200
    adv_init_lr: float = 1e-3
    adv_init_w_target: float = 1.0
    adv_init_w_anchor: float = 0.4
    adv_init_anchor_power: float = 2.0
    adv_init_w_anchor_max: float = 0.12
    adv_init_anchors_per_client: int = 50
    # 对抗初始化锚点来源: "clients"=各客户端 DataLoader（默认）; "attack_pool"=攻击池 attack_batches
    adv_init_anchor_source: str = "attack_pool"
    adv_init_log_every: int = 10
    adv_init_bundle_path: str = "outputs/toy_mlp_adv_init_bundle.pt"
    adv_init_bundle_use: bool = True  # True: 命中缓存则跳过对抗初始化


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


def load_agnews_data(cfg):
    total_needed = cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    rng = random.Random(cfg.seed)

    def balanced_pick(texts, labels, target_n):
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
                n = min(total_needed, len(texts))
                if n < total_needed:
                    print(f"警告: 本地AG News仅有 {n} 条，少于所需 {total_needed} 条。")
                return balanced_pick(texts, labels, n)
            print(f"警告: 本地AG News缺少 text/label 列，实际列: {list(df.columns)}")

    try:
        from datasets import load_dataset

        dataset = load_dataset("ag_news")
        train_data = dataset["train"]
        n = min(total_needed, len(train_data))
        train_texts = [item["text"] for item in train_data.select(range(n))]
        train_labels = [item["label"] for item in train_data.select(range(n))]
        return balanced_pick(train_texts, train_labels, n)
    except Exception:
        print("无法从本地或datasets加载AG News，使用合成数据...")
        texts = [f"This is a sample news article number {i}" for i in range(total_needed)]
        labels = [i % 4 for i in range(total_needed)]
        return texts, labels


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
    print(f"可训练参数数量: {trainable_params}")
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
    从攻击池列表中采样锚点 batch（跳过与 target 相同 input_ids 的项）。
    总个数与客户端模式一致: num_clients * adv_init_anchors_per_client。
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
        print("[对抗初始化] 攻击池锚点: 无可用候选（可能仅含目标）。")
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
        print("[对抗初始化] 无目标样本，跳过。")
        return
    params = get_trainable_params(model)
    if not params:
        print("[对抗初始化] 无可训练参数，跳过。")
        return

    model.train()
    opt = torch.optim.Adam(params, lr=cfg.adv_init_lr)
    n_anchor = max(1, len(anchor_batches))
    model.zero_grad(set_to_none=True)

    n_t0, mean_a0, max_a0, L0 = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
    print(
        f"[对抗初始化] 起始 step 0/{cfg.adv_init_steps}: "
        f"||∇L_target||={n_t0:.6f}, 锚点 mean||∇L||={mean_a0:.6f}, max||∇L||={max_a0:.6f}, "
        f"L={L0:.6f} (=-w_t||g_tgt||+(w_a/n)Σ||g||^{cfg.adv_init_anchor_power:g}"
        f"+{'w_max·max||g||' if cfg.adv_init_w_anchor_max > 0 else '0'}) "
        f"(锚点个数={len(anchor_batches)})"
    )

    log_every = max(0, int(cfg.adv_init_log_every))
    p = float(cfg.adv_init_anchor_power)
    for step in tqdm(range(cfg.adv_init_steps), desc="对抗初始化(分类头)"):
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
                f"[对抗初始化]  step {done:4d}/{cfg.adv_init_steps}: "
                f"||∇L_target||={nt:.6f}, 锚点 mean||∇L||={ma:.6f}, max||∇L||={mxa:.6f}, L={Lm:.6f}"
            )

    model.eval()
    model.zero_grad(set_to_none=True)
    n_tf, mean_af, max_af, Lf = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
    rel = (n_tf - n_t0) / (n_t0 + 1e-12) * 100.0
    print(
        f"[对抗初始化] 结束: ||∇L_target|| {n_t0:.6f} -> {n_tf:.6f} "
        f"(相对 {rel:+.2f}%), 锚点 mean||∇L|| {mean_a0:.6f} -> {mean_af:.6f}, "
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
    """给定各客户端的 score，计算用于“有无成员”的单调分数。"""
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
    raise ValueError(f"未知 server_det_score_mode: {mode!r}")


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
    """返回 (pred_member_present: bool, pred_member_idx or None)。"""
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
        print("[5] GAP 分布图: 未安装 matplotlib，跳过绘图。")
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
    print(f"[5] GAP 分布图已保存: {out_path}")
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
        print("[5] 客户端范数图: 未安装 matplotlib，跳过绘图。")
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
    print(f"[5] 客户端范数分布图已保存: {out_path}")
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
            print(f"[5] 指标分布图已保存({metric}): {out_path}")
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


def _gaussian_kde_1d(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Scott-rule Gaussian KDE (numpy-only; no scipy required)."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    g = np.asarray(grid, dtype=np.float64).ravel()
    n = int(x.size)
    if n == 0 or g.size == 0:
        return np.zeros_like(g, dtype=np.float64)
    if n == 1:
        bw = max(1e-3, abs(float(x[0])) * 1e-2 + 1e-6)
    else:
        std = float(np.std(x, ddof=1))
        if not np.isfinite(std) or std < 1e-12:
            bw = max(1e-3, abs(float(np.mean(x))) * 1e-2 + 1e-6)
        else:
            bw = std * (n ** (-1.0 / 5.0))
            bw = max(bw, 1e-12)
    z = (g[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))
    return dens.astype(np.float64, copy=False)


def _plot_member_nonmember_density(ax, pos_vals, neg_vals, *, use_log: bool) -> bool:
    """Draw overlapping member / non-member density curves on ``ax``."""

    def _clean(vals, require_positive: bool) -> np.ndarray:
        arr = np.asarray(
            [float(v) for v in vals if np.isfinite(v)],
            dtype=np.float64,
        )
        if require_positive:
            arr = arr[arr > 0.0]
        return arr

    pos = _clean(pos_vals, use_log)
    neg = _clean(neg_vals, use_log)
    if pos.size + neg.size == 0:
        return False
    allv = np.concatenate([a for a in (pos, neg) if a.size > 0])
    lo = float(np.min(allv))
    hi = float(np.max(allv))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return False
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1e-6)
        lo, hi = lo - pad, hi + pad

    n_grid = 256
    if use_log:
        log_lo, log_hi = np.log(lo), np.log(hi)
        span = max(log_hi - log_lo, 1e-12)
        pad = 0.08 * span
        log_grid = np.linspace(log_lo - pad, log_hi + pad, n_grid)
        grid = np.exp(log_grid)

        def _dens(samples: np.ndarray):
            if samples.size == 0:
                return None
            return _gaussian_kde_1d(np.log(samples), log_grid)
    else:
        span = max(hi - lo, 1e-12)
        pad = 0.08 * span
        grid = np.linspace(lo - pad, hi + pad, n_grid)

        def _dens(samples: np.ndarray):
            if samples.size == 0:
                return None
            return _gaussian_kde_1d(samples, grid)

    plotted = False
    d_neg = _dens(neg)
    if d_neg is not None:
        ax.plot(grid, d_neg, color="#1f77b4", lw=2.0, label="No member")
        ax.fill_between(grid, d_neg, color="#1f77b4", alpha=0.18)
        plotted = True
    d_pos = _dens(pos)
    if d_pos is not None:
        ax.plot(grid, d_pos, color="#d62728", lw=2.0, label="Member present")
        ax.fill_between(grid, d_pos, color="#d62728", alpha=0.18)
        plotted = True

    if use_log:
        ax.set_xscale("log")
    ax.set_ylim(bottom=0.0)
    return plotted


def plot_metric_panel(test_records, out_path: str, *, log_x: bool = True) -> bool:
    out_path = (out_path or "").strip()
    if not out_path or not test_records:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[5] Metric panel: matplotlib not installed; skipping plot.")
        return False

    # 2x2 panel; force_log=None means auto (log if span is large).
    metric_defs = [
        ("norm", "Client Gradient Norm", None),
        ("gap", "GAP (max - second_max)", None),
        ("ratio", "Ratio (max / mean)", False),  # typically ~1–2; keep linear like reference
        ("top1_top2", "Top1/Top2", None),
    ]
    panel_tags = ("(a)", "(b)", "(c)", "(d)")
    label_fs = 14
    tick_fs = 12

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes_flat = np.asarray(axes).ravel()

    for i, (metric, title, force_log) in enumerate(metric_defs):
        ax = axes_flat[i]
        tag = panel_tags[i]
        caption = f"{tag} {title}"
        pos_vals, neg_vals = _metric_values_for_records(test_records, metric)
        vals = [float(v) for v in (pos_vals + neg_vals) if np.isfinite(v)]
        if not vals:
            ax.set_xlabel(f"{caption} (empty)", fontsize=label_fs, fontweight="bold")
            ax.set_ylabel("Density", fontsize=label_fs, fontweight="bold")
            ax.tick_params(axis="both", labelsize=tick_fs)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
            continue
        vmin = float(min(vals))
        vmax = float(max(vals))
        if force_log is None:
            use_log = bool(log_x and vmin > 0.0 and (vmax / max(vmin, 1e-12) >= 10.0))
        else:
            use_log = bool(force_log and vmin > 0.0)
        ok = _plot_member_nonmember_density(ax, pos_vals, neg_vals, use_log=use_log)
        if not ok:
            ax.set_xlabel(f"{caption} (empty)", fontsize=label_fs, fontweight="bold")
            ax.set_ylabel("Density", fontsize=label_fs, fontweight="bold")
            ax.tick_params(axis="both", labelsize=tick_fs)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
            continue
        ax.set_title("")  # caption goes below the axes
        ax.set_xlabel(caption, fontsize=label_fs, fontweight="bold")
        ax.set_ylabel("Density", fontsize=label_fs, fontweight="bold")
        ax.tick_params(axis="both", labelsize=tick_fs)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        if i == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=11)

    for j in range(len(metric_defs), axes_flat.size):
        axes_flat[j].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[5] Metric density panel saved: {out_path}")
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
        print("警告: 攻击池为空，无法选择目标样本。")
        return None

    best_norm = -1.0
    best_batch = None
    for i in tqdm(range(len(samples)), desc="选择目标样本"):
        batch = samples[i]
        g = compute_true_gradient(model, batch)
        norm = g.norm().item()
        if norm > best_norm:
            best_norm = norm
            best_batch = batch

    print(f"目标样本梯度范数: {best_norm:.6f}")
    return best_batch


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


def _build_step45_meta(cfg, attack_texts, attack_labels):
    attack_sig = hashlib.sha1(
        json.dumps(
            {"texts": attack_texts, "labels": attack_labels},
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
        "local_steps": cfg.local_steps,
        "grad_stat_mode": "sum",
        "server_attack": "round1_g_sum_l2",
        "zo_eps": cfg.zo_eps,
        "lr": cfg.lr,
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
        print(f"[对抗初始化缓存] 读取失败，将重新训练: {e}")
        return False
    if not isinstance(obj, dict) or obj.get("meta") != cache_meta_45:
        print("[对抗初始化缓存] meta 不匹配，将重新训练。")
        return False
    tb = obj.get("target_batch")
    if not _target_batch_match(tb, target_batch):
        print("[对抗初始化缓存] 目标样本与当前不一致，将重新训练。")
        return False
    sd = obj.get("state_dict")
    if not isinstance(sd, dict):
        return False
    model.load_state_dict(sd, strict=True)
    print(f"[对抗初始化缓存] 已加载模型与目标: {path}")
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
    print(f"[对抗初始化缓存] 已保存: {cfg.adv_init_bundle_path}")


def main():
    cfg = Config()
    set_seed(cfg.seed)
    print(f"设备: {cfg.device}")
    print(f"配置: {cfg}")

    print("\n[1] 加载数据...")
    all_texts, all_labels = load_agnews_data(cfg)
    fl_texts = all_texts[: cfg.fl_samples * cfg.num_clients]
    fl_labels = all_labels[: cfg.fl_samples * cfg.num_clients]
    attack_texts = all_texts[
        cfg.fl_samples * cfg.num_clients : cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    ]
    attack_labels = all_labels[
        cfg.fl_samples * cfg.num_clients : cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    ]

    client_data = []
    per_client = cfg.fl_samples
    for i in range(cfg.num_clients):
        start = i * per_client
        end = (i + 1) * per_client
        client_data.append((fl_texts[start:end], fl_labels[start:end]))

    print(f"客户端数: {cfg.num_clients}, 每客户端样本: {per_client}")
    print(f"攻击池样本: {cfg.attack_samples}")

    print("\n[2] 加载模型...")
    model, tokenizer = get_model_and_tokenizer(cfg)

    print("\n[3] 创建数据加载器...")
    client_loaders = []
    for texts, labels in client_data:
        client_loaders.append(create_dataloader(texts, labels, tokenizer, cfg))

    attack_batches = []
    for i in range(len(attack_texts)):
        enc = tokenizer(
            attack_texts[i],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        attack_batches.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": torch.tensor([attack_labels[i]], dtype=torch.long),
            }
        )

    split_rng = random.Random(cfg.seed + 9011)
    attack_indices = list(range(len(attack_batches)))
    split_rng.shuffle(attack_indices)
    if len(attack_indices) >= 2:
        calib_n = int(round(len(attack_indices) * float(cfg.aux_attack_pool_calib_fraction)))
        calib_n = max(1, min(len(attack_indices) - 1, calib_n))
    else:
        calib_n = 0
    calib_idx = set(attack_indices[:calib_n])
    attack_init_batches = [attack_batches[i] for i in range(len(attack_batches)) if i not in calib_idx]
    attack_calib_batches = [attack_batches[i] for i in range(len(attack_batches)) if i in calib_idx]
    print(
        f"[3] attack_pool 划分完成: init={len(attack_init_batches)}, "
        f"calib={len(attack_calib_batches)}, total={len(attack_batches)}"
    )

    cache_meta_45 = _build_step45_meta(cfg, attack_texts, attack_labels)

    print("\n[4] 选择目标样本...")
    target_batch = None
    if os.path.exists(cfg.target_cache_path):
        try:
            target_obj = torch.load(cfg.target_cache_path, map_location="cpu")
            if isinstance(target_obj, dict) and target_obj.get("meta") == cache_meta_45:
                target_batch = target_obj.get("target_batch")
                if _batch_in_pool(target_batch, attack_init_batches):
                    print(f"命中目标样本缓存: {cfg.target_cache_path}")
                else:
                    print("[4] 目标缓存不在 attack_init 子集，强制重选。")
                    target_batch = None
        except Exception as e:
            print(f"读取目标样本缓存失败，将重新计算: {e}")
    if target_batch is None:
        target_source = attack_init_batches if attack_init_batches else attack_batches
        if not attack_init_batches:
            print("[4] 警告: attack_init 为空，目标样本回退到整池选择。")
        target_batch = select_target_sample(model, target_source, cfg)
        os.makedirs(os.path.dirname(cfg.target_cache_path) or ".", exist_ok=True)
        torch.save({"meta": cache_meta_45, "target_batch": target_batch}, cfg.target_cache_path)
        print(f"已保存目标样本缓存: {cfg.target_cache_path}")

    if cfg.adv_init_use:
        loaded = try_load_adv_init_bundle(model, cfg, cache_meta_45, target_batch)
        if not loaded:
            print("\n[4.6] 对抗初始化（仅分类头，放大目标梯度范数）...")
            src = (cfg.adv_init_anchor_source or "clients").strip().lower()
            if src == "attack_pool":
                anchor_batches = collect_anchor_batches_from_attack_pool(
                    attack_init_batches, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(f"[对抗初始化] 锚点来源: 攻击池（共 {len(anchor_batches)} 个 batch）")
            elif src == "clients":
                anchor_batches = collect_anchor_batches(
                    client_loaders, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(f"[对抗初始化] 锚点来源: 各客户端训练数据（共 {len(anchor_batches)} 个 batch）")
            else:
                print(f"警告: adv_init_anchor_source={cfg.adv_init_anchor_source!r} 无效，改用 clients。")
                anchor_batches = collect_anchor_batches(
                    client_loaders, target_batch, cfg, seed=cfg.seed + 7001
                )
            adversarial_sharpness_init(model, target_batch, anchor_batches, cfg)
            save_adv_init_bundle(model, cfg, cache_meta_45, target_batch)

    print("\n[5] 第一轮联邦模拟：阈值检测“有无成员”（不考虑多个成员）…")
    r1_rng = random.Random(cfg.seed + 1009)
    presence_acc = float("nan")
    roc_auc = float("nan")
    if cfg.local_steps < 1:
        print("跳过 [5]：local_steps 需 >= 1。")
        r1_trials_eff = 0
    else:
        r1_trials_eff = cfg.round1_server_trials
        max_s = cfg.local_steps - 1
        eff_inject = int(np.clip(cfg.round1_inject_step, 0, max_s))
        if cfg.round1_inject_step != eff_inject:
            print(f"警告: round1_inject_step={cfg.round1_inject_step} 已夹到 [0,{max_s}]: {eff_inject}")

        # 先计算每个 trial 的 det_score（不立即判决），最后再用阈值策略得到 pred 并算指标
        trial_records = []
        for trial in range(cfg.round1_server_trials):
            member_present = (r1_rng.random() < cfg.member_present_prob)
            if member_present:
                true_member = r1_rng.randrange(cfg.num_clients)
            else:
                true_member = None
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
                f"scores={[round(s, 4) for s in rec['scores']]}"
            )

        if cfg.server_threshold_mode == "online_quantile":
            threshold = float("nan")
            calib_source = "online_stream"
            calib_end = 0
            calib_records = []
            clip_min, clip_max = get_online_clip_bounds(cfg, cfg.server_det_score_mode)
            print(
                f"[5] 在线动态阈值: mode=online_quantile, warmup={cfg.online_warmup}, "
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
                print(f"[5] 阈值标定来源: trial_prefix (n={len(calib_records)})")
            elif calib_source == "aux_attack_pool":
                aux_pool = []
                tgt_ids = target_batch["input_ids"].detach().cpu() if target_batch is not None else None
                for b in attack_calib_batches:
                    if tgt_ids is not None and b["input_ids"].shape == tgt_ids.shape:
                        if torch.equal(b["input_ids"].detach().cpu(), tgt_ids):
                            continue
                    aux_pool.append(b)
                if not aux_pool:
                    print("[5] 警告: aux_attack_pool 为空，回退到 trial_prefix 标定。")
                    calib_records = trial_records[:calib_end]
                    calib_source = "trial_prefix"
                else:
                    calib_n = int(max(1, cfg.threshold_calib_trials))
                    calib_rng = random.Random(cfg.seed + 11009)
                    calib_records = []
                    for c in range(calib_n):
                        c_member_present = (calib_rng.random() < cfg.member_present_prob)
                        c_true_member = calib_rng.randrange(cfg.num_clients) if c_member_present else None
                        injected_batch = aux_pool[calib_rng.randrange(len(aux_pool))]
                        c_rec = simulate_round1_trial_record(
                            model,
                            client_loaders,
                            injected_batch,
                            cfg,
                            trial_seed=cfg.seed + 150000 + c * 1000,
                            member_present=c_member_present,
                            true_member=c_true_member,
                            inject_step=eff_inject,
                        )
                        calib_records.append(c_rec)
                    calib_end = len(calib_records)
                    print(
                        f"[5] 阈值标定来源: aux_attack_pool "
                        f"(calib_trials={calib_end}, aux_candidates={len(aux_pool)})"
                    )
            else:
                raise ValueError(f"未知 threshold_calib_source: {cfg.threshold_calib_source!r}")

        if cfg.server_threshold_mode == "fixed":
            threshold = float(cfg.server_fixed_threshold)
            print(f"[5] 使用 fixed 阈值: threshold={threshold}")
        elif cfg.server_threshold_mode == "calibrate_on_neg":
            neg_det_scores = [
                r["det_score"]
                for r in calib_records
                if (r["member_present"] is False)
            ]
            if len(neg_det_scores) < 1:
                threshold = float(cfg.server_fixed_threshold)
                print(
                    f"[5] 标定失败（无 negative trial），回退 fixed threshold={threshold}"
                )
            else:
                q = 1.0 - float(cfg.desired_fpr)
                threshold = float(np.quantile(neg_det_scores, q))
                print(
                    f"[5] 标定阈值：calib_end={calib_end}, neg={len(neg_det_scores)}, "
                    f"desired_fpr={cfg.desired_fpr} => threshold={threshold:.6f}"
                )
        elif cfg.server_threshold_mode == "roc_youden":
            y_true_calib = np.array([1 if r["member_present"] else 0 for r in calib_records], dtype=np.int32)
            y_score_calib = np.array([r["det_score"] for r in calib_records], dtype=np.float64)
            if len(np.unique(y_true_calib)) < 2:
                threshold = float(cfg.server_fixed_threshold)
                print(
                    f"[5] ROC-Youden 标定失败（calib 仅单类标签），回退 fixed threshold={threshold}"
                )
            else:
                try:
                    from sklearn.metrics import roc_curve
                except ImportError:
                    threshold = float(cfg.server_fixed_threshold)
                    print(
                        f"[5] ROC-Youden 标定失败（未安装 sklearn），回退 fixed threshold={threshold}"
                    )
                else:
                    fpr_calib, tpr_calib, thr_calib = roc_curve(y_true_calib, y_score_calib)
                    valid = np.isfinite(thr_calib)
                    if not np.any(valid):
                        threshold = float(cfg.server_fixed_threshold)
                        print(
                            f"[5] ROC-Youden 标定失败（阈值全为非有限值），回退 fixed threshold={threshold}"
                        )
                    else:
                        youden = tpr_calib[valid] - fpr_calib[valid]
                        best_idx_local = int(np.argmax(youden))
                        threshold = float(thr_calib[valid][best_idx_local])
                        best_j = float(youden[best_idx_local])
                        best_tpr = float(tpr_calib[valid][best_idx_local])
                        best_fpr = float(fpr_calib[valid][best_idx_local])
                        print(
                            f"[5] ROC-Youden 标定阈值：calib_end={calib_end}, "
                            f"best_j={best_j:.6f} (TPR={best_tpr:.4f}, FPR={best_fpr:.4f}) "
                            f"=> threshold={threshold:.6f}"
                        )
        elif cfg.server_threshold_mode == "online_quantile":
            # 在线模式的阈值在预测阶段逐轮更新，这里不做静态阈值计算。
            pass
        else:
            raise ValueError(f"未知 server_threshold_mode: {cfg.server_threshold_mode!r}")

        # 评价：
        # - trial_prefix: 前缀参与标定，测试只看后缀
        # - aux_attack_pool: 标定与评估完全独立，测试使用全部 trial_records
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
        q = 1.0 - alpha
        min_thr, max_thr = get_online_clip_bounds(cfg, cfg.server_det_score_mode)
        evaluated_records = []
        online_hist_updates = 0

        for idx, r in enumerate(test_records):
            if cfg.server_threshold_mode == "online_quantile":
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

                if len(online_history) >= warmup:
                    hist = online_history[-window:]
                    dynamic_thr = float(np.quantile(np.asarray(hist, dtype=np.float64), q))
                    dynamic_thr = min(max(dynamic_thr, min_thr), max_thr)
                else:
                    dynamic_thr = min(max(float(cfg.server_fixed_threshold), min_thr), max_thr)
                pred_present = (score > dynamic_thr)
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
                pred_present = (r["det_score"] > threshold)
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

        roc_auc, _, _ = compute_roc_auc_and_maybe_plot(
            evaluated_records,
            (cfg.roc_plot_path or "").strip(),
            cfg.model_name,
            cfg.roc_dataset_label,
            scores_cache_path=(cfg.roc_scores_cache_path or "").strip(),
        )
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
            print(f"[5] 额外指标图输出数量: {saved_n}")
        if cfg.metric_panel_plot_enable:
            plot_metric_panel(
                evaluated_records,
                (cfg.metric_panel_plot_path or "").strip(),
                log_x=bool(cfg.metric_panel_plot_log_x),
            )

        print("\n" + "=" * 60)
        print(
            f"[5] 阈值检测指标（test trials: {total}）\n"
            f"presence_acc={presence_acc:.1%}, precision={precision:.1%}, TPR={tpr:.1%}, FPR={fpr:.1%}, "
            f"miss_rate={miss_rate:.1%}"
        )
        if np.isfinite(roc_auc):
            print(f"[5] ROC-AUC（test, det_score）={roc_auc:.4f}")
        print(
            f"[5] 成员存在条件下的成员指认准确率：{member_idx_correct}/{member_idx_pred_cnt} = {member_idx_acc:.1%}"
        )
        if cfg.server_threshold_mode == "online_quantile":
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
            if cfg.compare_metric_modes_enable:
                compare_modes = [
                    ("norm_all", "norm(all)"),
                    ("gap", "gap"),
                    ("ratio", "ratio"),
                    ("top1_top2", "top1/top2"),
                ]
                print("[5] 四指标对比（同一在线阈值策略）:")
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
        else:
            print(f"[5] decision threshold={threshold:.6f}, mode={cfg.server_det_score_mode!r}")
        print("=" * 60)

    return presence_acc


if __name__ == "__main__":
    main()
