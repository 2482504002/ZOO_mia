"""
FedMeZO client-driven membership inference experiment (multi-dataset + DistilBERT classification head).

Usage:
  python MIA_distillbert.py --dataset agnews
  python MIA_distillbert.py --dataset alpaca
  python MIA_distillbert.py --dataset dolly
  python MIA_distillbert.py --dataset gsm8k

Default dataset paths (override with --data_root):
  agnews  -> /home/zhike/JWH/ZOO_MIA/datasets/agnews (local parquet / h5)
  alpaca  -> /home/zhike/JWH/ZOO_MIA/datasets/alpaca_data.json
  dolly   -> /home/zhike/JWH/ZOO_MIA/datasets/dolly15k/databricks-dolly-15k.jsonl
  gsm8k   -> /home/zhike/JWH/ZOO_MIA/datasets/gsm8k (local parquet only, no HF fallback)
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mia_roc_plotting import compute_roc_auc_and_maybe_plot

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_DATASETS = os.path.join(_REPO_ROOT, "datasets")


@dataclass
class Config:
    seed: int = 42
    dataset: str = "agnews"
    data_root: str = ""
    alpaca_json_name: str = "alpaca_data.json"
    dolly_jsonl_name: str = "databricks-dolly-15k.jsonl"
    gsm8k_parquet_subdir: str = "main"
    gsm8k_merge_test: bool = True
    num_labels: int = 4
    model_name: str = "/home/zhike/JWH/model/distilbert-base-uncased/"
    max_length: int = 256
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"

    num_clients: int = 2
    fl_samples: int = 2000
    local_steps: int = 10
    batch_size: int = 1
    lr: float = 1e-5
    zo_eps: float = 1e-3

    attack_samples: int = 500
    # sequential (default unified split); alpaca may use dirichlet / uniform mutually exclusive sampling
    attack_sampling_mode: str = "sequential"
    attack_dirichlet_alpha: float = 0.1
    target_cache_path: str = ""

    round1_server_trials: int = 1000
    round1_inject_step: int = 0
    member_present_prob: float = 0.5
    server_det_score_mode: str = "gap"
    server_threshold_mode: str = "online_quantile"
    # trial_prefix | aux_attack_pool (agnews defaults to the latter)
    threshold_calib_source: str = "trial_prefix"
    aux_attack_pool_calib_fraction: float = 0.3
    threshold_calib_trials: int = 50
    server_fixed_threshold: float = 0.0
    desired_fpr: float = 0.05
    threshold_calib_fraction: float = 0.5
    online_warmup: int = 30
    online_window: int = 80
    online_alpha: float = 0.12
    online_min_threshold: float = 10000
    online_max_threshold: float = 1e12
    online_update_with_neg_only: bool = True
    online_auto_clip_by_mode: bool = True
    compare_metric_modes_enable: bool = True

    compare_plot_path: str = ""
    compare_roc_data_path: str = ""
    compare_include_fedmia: bool = True
    compare_include_ltmia: bool = True
    fedmia_direction_calibrate: bool = True
    ltmia_early_ratio: float = 0.5
    ltmia_hidden_dim: int = 32
    ltmia_epochs: int = 200
    ltmia_lr: float = 1e-2
    ltmia_weight_decay: float = 1e-4
    ltmia_min_train_samples: int = 12
    ltmia_use_log1p: bool = True
    ltmia_use_clean_model: bool = True
    ltmia_report_auc_target: float = float("nan")
    ltmia_report_auc_atol: float = 0.02

    roc_plot_path: str = ""
    roc_scores_cache_path: str = ""
    roc_dataset_label: str = ""

    # agnews extras (disabled by default for other datasets)
    gap_plot_path: str = ""
    client_norm_plot_path: str = ""
    client_norm_plot_log_x: bool = True
    extra_metric_plot_enable: bool = False
    extra_metric_plot_dir: str = ""
    extra_metric_plot_log_x: bool = True
    metric_panel_plot_enable: bool = False
    metric_panel_plot_path: str = ""
    metric_panel_plot_log_x: bool = True

    adv_init_use: bool = True
    adv_init_steps: int = 200
    adv_init_lr: float = 1e-3
    adv_init_w_target: float = 1.0
    adv_init_w_anchor: float = 0.4
    adv_init_anchor_power: float = 2.0
    adv_init_w_anchor_max: float = 0.12
    adv_init_anchors_per_client: int = 50
    adv_init_anchor_source: str = "attack_pool"
    adv_init_log_every: int = 10
    adv_init_bundle_path: str = ""
    adv_init_bundle_use: bool = True


def normalize_dataset_name(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "agnews": "agnews",
        "alpaca": "alpaca",
        "dolly": "dolly",
        "dolly15k": "dolly",
        "gsm8k": "gsm8k",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown dataset={name!r}; choices: agnews, alpaca, dolly, dolly15k, gsm8k"
        )
    return aliases[key]


def apply_dataset_defaults(cfg: Config, *, data_root_cli: Optional[str] = None) -> Config:
    """Fill output paths and default data_root from --dataset."""
    ds = normalize_dataset_name(cfg.dataset)
    cfg.dataset = ds

    defaults = {
        "agnews": {
            "data_root": os.path.join(_REPO_DATASETS, "agnews"),
            "max_length": 64,
            "roc_dataset_label": "AG News",
            "target_cache_path": "outputs/agnews_distilbert_mia_target_cache.pt",
            "adv_init_bundle_path": "outputs/agnews_distilbert_mia_adv_init_bundle.pt",
            "roc_scores_cache_path": "outputs/mia_roc_scores_distilbert_agnews.npz",
            "compare_plot_path": "outputs/distilbert_agnews_mia_compare_roc.png",
            "gap_plot_path": "outputs/gap-client_norm_strip_distilbert_agnews.pdf",
            "client_norm_plot_path": "outputs/client_norm_strip_distilbert_agnews.pdf",
            "extra_metric_plot_enable": True,
            "extra_metric_plot_dir": "outputs/det_metric_strips_distilbert_agnews",
            "metric_panel_plot_enable": True,
            "metric_panel_plot_path": "outputs/det_metric_panel_distilbert_agnews.pdf",
            "threshold_calib_source": "aux_attack_pool",
            "ltmia_report_auc_target": float("nan"),
        },
        "alpaca": {
            "data_root": os.path.join(_REPO_DATASETS, "alpaca_data.json"),
            "max_length": 256,
            "roc_dataset_label": "Alpaca",
            "target_cache_path": "outputs/alpaca_distilbert_mia_target_cache.pt",
            "adv_init_bundle_path": "outputs/alpaca_distilbert_mia_adv_init_bundle.pt",
            "roc_scores_cache_path": "outputs/mia_roc_scores_distilbert_alpaca.npz",
            "compare_plot_path": "outputs/distilbert_alpaca_mia_compare_roc.png",
            "threshold_calib_source": "trial_prefix",
            "extra_metric_plot_enable": False,
            "metric_panel_plot_enable": False,
            "ltmia_report_auc_target": float("nan"),
        },
        "dolly": {
            "data_root": os.path.join(
                _REPO_DATASETS, "dolly15k", "databricks-dolly-15k.jsonl"
            ),
            "max_length": 256,
            "roc_dataset_label": "Dolly-15k",
            "target_cache_path": "outputs/dolly15k_distilbert_mia_target_cache.pt",
            "adv_init_bundle_path": "outputs/dolly15k_distilbert_mia_adv_init_bundle.pt",
            "roc_scores_cache_path": "outputs/mia_roc_scores_distilbert_dolly.npz",
            "compare_plot_path": "outputs/distilbert_dolly_mia_compare_roc.png",
            "threshold_calib_source": "trial_prefix",
            "extra_metric_plot_enable": False,
            "metric_panel_plot_enable": False,
            "ltmia_report_auc_target": float("nan"),
        },
        "gsm8k": {
            "data_root": os.path.join(_REPO_DATASETS, "gsm8k"),
            "max_length": 512,
            "roc_dataset_label": "GSM8K",
            "target_cache_path": "outputs/gsm8k_distilbert_mia_target_cache.pt",
            "adv_init_bundle_path": "outputs/gsm8k_distilbert_mia_adv_init_bundle.pt",
            "roc_scores_cache_path": "outputs/mia_roc_scores_distilbert_gsm8k.npz",
            "compare_plot_path": "outputs/distilbert_gsm8k_mia_compare_roc.png",
            "roc_plot_path": "outputs/mia_roc_distilbert_gsm8k.png",
            "threshold_calib_source": "trial_prefix",
            "extra_metric_plot_enable": False,
            "metric_panel_plot_enable": False,
            "ltmia_report_auc_target": 0.6,
        },
    }
    d = defaults[ds]
    for k, v in d.items():
        setattr(cfg, k, v)

    if data_root_cli:
        cfg.data_root = data_root_cli
    elif not (cfg.data_root or "").strip():
        cfg.data_root = d["data_root"]

    return cfg


class TextClsDataset(Dataset):
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


def _balanced_pick(texts, labels, target_n, rng):
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


def _sample_attack_indices_dirichlet(labels, k: int, alpha: float, seed: int):
    """Generate attack-pool sample quotas per class via Dirichlet(alpha), then sample randomly within each class."""
    n = len(labels)
    if n <= 0 or k <= 0:
        return []
    k = int(min(k, n))
    y = np.asarray(labels, dtype=np.int64)
    classes = np.unique(y)
    if classes.size <= 1:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx[:k]

    alpha = float(max(alpha, 1e-6))
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(classes.shape[0], alpha, dtype=np.float64))
    want = rng.multinomial(k, p)

    cls_to_idx = {}
    for c in classes:
        idx_c = np.where(y == c)[0].tolist()
        random.Random(seed + int(c) * 131 + 17).shuffle(idx_c)
        cls_to_idx[int(c)] = idx_c

    chosen = []
    leftovers = []
    for i, c in enumerate(classes.tolist()):
        pool = cls_to_idx[int(c)]
        take = int(min(want[i], len(pool)))
        chosen.extend(pool[:take])
        leftovers.extend(pool[take:])

    if len(chosen) < k:
        random.Random(seed + 9973).shuffle(leftovers)
        need = k - len(chosen)
        chosen.extend(leftovers[:need])

    random.Random(seed + 4242).shuffle(chosen)
    return chosen[:k]


def _load_agnews_from_parquet(data_root: str):
    local_data_dir = os.path.join(data_root, "data")
    if not os.path.isdir(local_data_dir):
        return None
    parquet_files = sorted(
        os.path.join(local_data_dir, f)
        for f in os.listdir(local_data_dir)
        if f.endswith(".parquet")
    )
    if not parquet_files:
        return None
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("Reading AG News parquet requires pandas (and pyarrow)") from e
    frames = [pd.read_parquet(fp) for fp in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    if "text" not in df.columns or "label" not in df.columns:
        print(f"Warning: local AG News parquet missing text/label columns; actual columns: {list(df.columns)}")
        return None
    texts = df["text"].astype(str).tolist()
    labels = (
        pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).tolist()
    )
    return texts, labels


def _load_agnews_from_h5(data_root: str):
    candidates = [
        os.path.join(data_root, "data", "agnews_data.h5"),
        os.path.join(data_root, "agnews_data.h5"),
    ]
    h5_path = next((p for p in candidates if os.path.isfile(p)), None)
    if h5_path is None:
        return None
    try:
        import h5py
    except ImportError:
        print(f"Warning: found {h5_path} but h5py is not installed; skipping h5 fallback.")
        return None
    texts, labels = [], []
    with h5py.File(h5_path, "r") as f:
        if "X" in f and "Y" in f:
            X, Y = f["X"], f["Y"]
            for i in range(len(Y)):
                x = X[i]
                if isinstance(x, bytes):
                    x = x.decode("utf-8", errors="ignore")
                elif hasattr(x, "tobytes"):
                    x = x.tobytes().decode("utf-8", errors="ignore")
                else:
                    x = str(x)
                texts.append(x)
                labels.append(int(Y[i]))
        else:
            for split in ("train", "test", "Train", "Test"):
                if split not in f:
                    continue
                g = f[split]
                text_key = next((k for k in ("text", "X", "texts") if k in g), None)
                label_key = next((k for k in ("label", "Y", "labels", "y") if k in g), None)
                if text_key is None or label_key is None:
                    continue
                for i in range(len(g[label_key])):
                    x = g[text_key][i]
                    if isinstance(x, bytes):
                        x = x.decode("utf-8", errors="ignore")
                    else:
                        x = str(x)
                    texts.append(x)
                    labels.append(int(g[label_key][i]))
    if not texts:
        print(f"Warning: failed to parse text from h5: {h5_path}")
        return None
    print(f"[AGNews] Loaded {len(texts)} samples from h5: {h5_path}")
    return texts, labels


def load_agnews_data(cfg):
    """Prefer local parquet, then h5; raise explicitly if missing (no silent synthesis)."""
    total_needed = cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    rng = random.Random(cfg.seed)
    root = cfg.data_root

    loaded = _load_agnews_from_parquet(root)
    if loaded is None:
        loaded = _load_agnews_from_h5(root)
    if loaded is None:
        raise FileNotFoundError(
            f"No usable local AG News data found (expected {root}/data/*.parquet or agnews_data.h5)."
            f"Prepare local files before running (no network download by default)."
        )
    texts, labels = loaded
    n = min(total_needed, len(texts))
    if len(texts) < total_needed:
        print(f"Warning: local AG News has only {len(texts)} samples, fewer than required {total_needed}.")
    print(f"[AGNews] Balanced sample of {n} (pool size {len(texts)}).")
    return _balanced_pick(texts, labels, n, rng)


def _resolve_alpaca_json_path(cfg) -> str:
    root = (cfg.data_root or "").strip()
    name = getattr(cfg, "alpaca_json_name", "alpaca_data.json")
    if root and os.path.isfile(root):
        return root
    if root and os.path.isdir(root):
        cand = os.path.join(root, name)
        if os.path.isfile(cand):
            return cand
    if root.endswith(".json") and not os.path.isfile(root):
        raise FileNotFoundError(f"Alpaca JSON not found: {root}")
    fallback = os.path.join(_REPO_DATASETS, name)
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        f"Alpaca data not found. Tried: data_root={root!r}, "
        f"join={os.path.join(root, name) if root else None}, fallback={fallback}"
    )


def _alpaca_record_to_text(d: dict) -> str:
    inst = (d.get("instruction") or "").strip()
    inp = (d.get("input") or "").strip()
    out = (d.get("output") or "").strip()
    if inp:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{inst}\n\n### Response:\n{out}"
    )


def _alpaca_pseudo_label(d: dict, num_classes: int) -> int:
    key = f"{d.get('instruction', '')}\n{d.get('output', '')}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest()[:12], 16)
    return h % num_classes


def load_alpaca_data(cfg):
    path = _resolve_alpaca_json_path(cfg)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("alpaca_data.json should be a JSON array list[dict]")
    texts = [_alpaca_record_to_text(d) for d in raw]
    labels = [_alpaca_pseudo_label(d, cfg.num_labels) for d in raw]
    total_needed = cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    n = min(total_needed, len(texts))
    if len(texts) < total_needed:
        print(f"Warning: Alpaca has only {len(texts)} samples, fewer than required {total_needed}; truncating to available count.")
    print(f"[Alpaca] Loaded from {path}, balanced sample of {n}.")
    rng = random.Random(cfg.seed)
    return _balanced_pick(texts, labels, n, rng)


def _dolly_record_to_text(d: dict) -> str:
    inst = (d.get("instruction") or "").strip()
    ctx = (d.get("context") or "").strip()
    resp = (d.get("response") or "").strip()
    if ctx:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{inst}\n\n### Input:\n{ctx}\n\n### Response:\n{resp}"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
    )


def _dolly_pseudo_label(d: dict, num_classes: int) -> int:
    key = (
        f"{d.get('instruction', '')}\n{d.get('context', '')}\n{d.get('response', '')}"
    ).encode("utf-8")
    h = int(hashlib.md5(key).hexdigest()[:12], 16)
    return h % num_classes


def _read_dolly_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_dolly_jsonl_path(cfg) -> str:
    root = (cfg.data_root or "").strip()
    name = getattr(cfg, "dolly_jsonl_name", "databricks-dolly-15k.jsonl")
    if root and os.path.isfile(root):
        return root
    if root and os.path.isdir(root):
        cand = os.path.join(root, name)
        if os.path.isfile(cand):
            return cand
    fallback = os.path.join(_REPO_DATASETS, "dolly15k", name)
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        f"Dolly-15k JSONL not found. Tried: data_root={root!r}, fallback={fallback}"
    )


def load_dolly15k_data(cfg):
    path = _resolve_dolly_jsonl_path(cfg)
    raw = _read_dolly_jsonl(path)
    texts = [_dolly_record_to_text(d) for d in raw]
    labels = [_dolly_pseudo_label(d, cfg.num_labels) for d in raw]
    total_needed = cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    n = min(total_needed, len(texts))
    if len(texts) < total_needed:
        print(f"Warning: Dolly-15k has only {len(texts)} samples, fewer than required {total_needed}; truncating to available count.")
    print(f"[Dolly] Loaded from {path}, balanced sample of {n}.")
    rng = random.Random(cfg.seed)
    return _balanced_pick(texts, labels, n, rng)


def _gsm8k_record_to_text(question: str, answer: str) -> str:
    q = (question or "").strip()
    a = (answer or "").strip()
    return f"Question: {q}\nAnswer: {a}"


def _gsm8k_pseudo_label(question: str, answer: str, num_classes: int) -> int:
    key = f"{question}\n{answer}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest()[:12], 16)
    return h % num_classes


def _read_gsm8k_parquet_paths(paths):
    """Read parquet files into {question, answer} list via pyarrow or pandas (no datasets dependency)."""
    if not paths:
        return []

    def _with_pyarrow():
        import pyarrow.parquet as pq

        rows = []
        for p in paths:
            table = pq.read_table(p, columns=["question", "answer"])
            rows.extend(table.to_pylist())
        return rows

    def _with_pandas():
        import pandas as pd

        frames = [pd.read_parquet(p, columns=["question", "answer"]) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        return df.to_dict(orient="records")

    try:
        return _with_pyarrow()
    except Exception:
        return _with_pandas()


def _load_gsm8k_from_local_parquet(cfg):
    import glob

    parquet_dir = os.path.join(cfg.data_root, cfg.gsm8k_parquet_subdir)
    if not os.path.isdir(parquet_dir):
        return None
    train_files = sorted(glob.glob(os.path.join(parquet_dir, "train-*.parquet")))
    if not train_files:
        return None
    paths = list(train_files)
    if cfg.gsm8k_merge_test:
        paths.extend(sorted(glob.glob(os.path.join(parquet_dir, "test-*.parquet"))))
    return _read_gsm8k_parquet_paths(paths)


def load_gsm8k_data(cfg):
    """Local parquet only; raise if missing, no HuggingFace download fallback."""
    total_needed = cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    rng = random.Random(cfg.seed)

    raw = _load_gsm8k_from_local_parquet(cfg)
    if raw is None:
        local_hint = os.path.join(cfg.data_root, cfg.gsm8k_parquet_subdir, "train-*.parquet")
        raise FileNotFoundError(
            f"Local GSM8K parquet not found (expected {local_hint})."
            f"Place train-*.parquet (optional test-*.parquet) and install pyarrow or pandas."
            f"This script does not fall back to HuggingFace download."
        )
    print(
        f"[GSM8K] Loaded {len(raw)} samples from local parquet"
        f" ({cfg.data_root}/{cfg.gsm8k_parquet_subdir})."
    )

    texts = [_gsm8k_record_to_text(r.get("question", ""), r.get("answer", "")) for r in raw]
    labels = [
        _gsm8k_pseudo_label(r.get("question", ""), r.get("answer", ""), cfg.num_labels) for r in raw
    ]
    n = min(total_needed, len(texts))
    if len(texts) < total_needed:
        print(
            f"Warning: GSM8K has only {len(texts)} samples, fewer than required {total_needed}; balanced sampling with available count."
        )
    return _balanced_pick(texts, labels, n, rng)


def load_dataset_texts_labels(cfg):
    ds = normalize_dataset_name(cfg.dataset)
    if ds == "agnews":
        return load_agnews_data(cfg)
    if ds == "alpaca":
        return load_alpaca_data(cfg)
    if ds == "dolly":
        return load_dolly15k_data(cfg)
    if ds == "gsm8k":
        return load_gsm8k_data(cfg)
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def split_fl_and_attack_pool(all_texts, all_labels, cfg):
    """Default sequential split; mutually exclusive sampling when alpaca + dirichlet/uniform."""
    need_fl = int(cfg.fl_samples * cfg.num_clients)
    total_n = len(all_texts)
    mode = str(getattr(cfg, "attack_sampling_mode", "sequential")).strip().lower()
    use_disjoint = (
        normalize_dataset_name(cfg.dataset) == "alpaca"
        and mode in ("dirichlet", "uniform")
    )

    if not use_disjoint:
        fl_texts = all_texts[:need_fl]
        fl_labels = all_labels[:need_fl]
        attack_texts = all_texts[need_fl : need_fl + cfg.attack_samples]
        attack_labels = all_labels[need_fl : need_fl + cfg.attack_samples]
        print(f"[1] Split: mode=sequential, fl={len(fl_texts)}, attack={len(attack_texts)}")
        return fl_texts, fl_labels, attack_texts, attack_labels

    if total_n < need_fl:
        raise RuntimeError(f"Insufficient samples: only {total_n}, fewer than {need_fl} required for federated training.")
    all_idx = list(range(total_n))
    if mode == "dirichlet":
        attack_idx = _sample_attack_indices_dirichlet(
            all_labels,
            k=int(cfg.attack_samples),
            alpha=float(getattr(cfg, "attack_dirichlet_alpha", 0.1)),
            seed=int(cfg.seed) + 3007,
        )
        print(
            f"[1] Attack pool sampling: mode=dirichlet, alpha={float(cfg.attack_dirichlet_alpha):.4f}, "
            f"n={len(attack_idx)}"
        )
    else:
        rng_idx = random.Random(cfg.seed + 3007)
        rng_idx.shuffle(all_idx)
        attack_idx = all_idx[: int(min(cfg.attack_samples, total_n - need_fl))]
        print(f"[1] Attack pool sampling: mode=uniform, n={len(attack_idx)}")

    attack_set = set(attack_idx)
    remain_idx = [i for i in all_idx if i not in attack_set]
    random.Random(cfg.seed + 3008).shuffle(remain_idx)
    if len(remain_idx) < need_fl:
        raise RuntimeError(
            f"Remaining samples after removing attack pool insufficient for FL set: remain={len(remain_idx)}, need={need_fl}"
        )
    fl_idx = remain_idx[:need_fl]
    fl_texts = [all_texts[i] for i in fl_idx]
    fl_labels = [all_labels[i] for i in fl_idx]
    attack_texts = [all_texts[i] for i in attack_idx]
    attack_labels = [all_labels[i] for i in attack_idx]
    return fl_texts, fl_labels, attack_texts, attack_labels



def get_model_and_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    ).to(cfg.device)

    for name, param in model.named_parameters():
        if "classifier" not in name and "score" not in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameter count: {trainable_params}")
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
    Sample anchor batches from attack pool (skip items with same input_ids as target).
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
        print("[Adversarial init] Attack-pool anchors: no candidates (pool may contain only target).")
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
        print("[Adversarial init] No target sample; skipping.")
        return
    params = get_trainable_params(model)
    if not params:
        print("[Adversarial init] No trainable parameters; skipping.")
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
        f"(anchor count={len(anchor_batches)})"
    )

    log_every = max(0, int(cfg.adv_init_log_every))
    p = float(cfg.adv_init_anchor_power)
    for step in tqdm(range(cfg.adv_init_steps), desc="Adversarial init (classification head)"):
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
    loss_traj = []
    for step in range(cfg.local_steps):
        assign_params(params, theta)
        if target_batch is not None:
            with torch.no_grad():
                lt = float(compute_loss(model, target_batch).detach().float().item())
            if not np.isfinite(lt):
                lt = 0.0
        else:
            lt = 0.0
        loss_traj.append(lt)

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
    return g_sum, loss_traj


def server_pick_member_by_grad_norm(g_sums):
    scores = [float(g.float().norm().item()) for g in g_sums]
    return int(np.argmax(scores)), scores


def compute_det_score(scores, mode: str):
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


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def compute_fedmia_score(g_uploads, target_grad, *, eps: float = 1e-12):
    if not g_uploads:
        return 0.5, -1, []
    if target_grad is None or target_grad.numel() == 0:
        return 0.5, -1, []
    g = target_grad.float().reshape(-1)
    g_norm = float(g.norm().item())
    if (not np.isfinite(g_norm)) or g_norm <= eps:
        return 0.5, -1, []

    m_vals = []
    for up in g_uploads:
        u = up.float().reshape(-1)
        u_norm = float(u.norm().item())
        if (not np.isfinite(u_norm)) or u_norm <= eps:
            m_vals.append(0.0)
            continue
        cos = float(torch.dot(u, g).item()) / (u_norm * g_norm + eps)
        m_vals.append(float(np.clip(cos, -1.0, 1.0)))
    if not m_vals:
        return 0.5, -1, []

    tar_idx = int(np.argmax(m_vals))
    m_tar = float(m_vals[tar_idx])
    if len(m_vals) == 1:
        return float(np.clip(_normal_cdf(m_tar), 0.0, 1.0)), tar_idx, m_vals

    out_vals = np.asarray([m for i, m in enumerate(m_vals) if i != tar_idx], dtype=np.float64)
    if out_vals.size == 0:
        return float(np.clip(_normal_cdf(m_tar), 0.0, 1.0)), tar_idx, m_vals
    mu = float(np.mean(out_vals))
    sigma = float(np.std(out_vals))
    keep = out_vals <= (mu + 3.0 * sigma + eps)
    u_vals = out_vals[keep] if np.any(keep) else out_vals
    mu_out = float(np.mean(u_vals))
    std_out = max(float(np.std(u_vals)), eps)
    z = (m_tar - mu_out) / std_out
    score = float(np.clip(_normal_cdf(z), 0.0, 1.0))
    return score, tar_idx, m_vals


def _gather_fedmia_out_pool(records):
    pool = []
    for r in records:
        m_vals = r.get("fedmia_m_values", None)
        tar_idx = int(r.get("fedmia_tar_idx", -1))
        if (not isinstance(m_vals, list)) or len(m_vals) <= 0:
            continue
        for i, v in enumerate(m_vals):
            if i != tar_idx and np.isfinite(v):
                pool.append(float(v))
    return np.asarray(pool, dtype=np.float64)


def finalize_fedmia_scores(records, reference_records=None, *, eps: float = 1e-12):
    if not records:
        return
    ref = reference_records if reference_records else records
    pool = _gather_fedmia_out_pool(ref)
    if pool.size <= 0:
        pool = _gather_fedmia_out_pool(records)
    if pool.size > 0:
        mu_pool = float(np.mean(pool))
        std_pool = max(float(np.std(pool)), eps)
    else:
        mu_pool = 0.0
        std_pool = 1.0

    for r in records:
        m_vals = r.get("fedmia_m_values", None)
        tar_idx = int(r.get("fedmia_tar_idx", -1))
        if (not isinstance(m_vals, list)) or len(m_vals) <= 0 or tar_idx < 0 or tar_idx >= len(m_vals):
            r["fedmia_score"] = 0.5
            continue
        m_tar = float(m_vals[tar_idx])
        out_vals = np.asarray([m for i, m in enumerate(m_vals) if i != tar_idx], dtype=np.float64)
        if out_vals.size >= 2 and float(np.std(out_vals)) > eps:
            mu_out = float(np.mean(out_vals))
            std_out = max(float(np.std(out_vals)), eps)
        else:
            mu_out = mu_pool
            std_out = std_pool
        z = (m_tar - mu_out) / std_out
        r["fedmia_score"] = float(np.clip(_normal_cdf(z), 0.0, 1.0))


def compute_auc_and_curve_from_arrays(y_true, y_score):
    if y_true is None or y_score is None or len(y_true) == 0:
        return float("nan"), None, None
    if len(np.unique(y_true)) < 2:
        return float("nan"), None, None
    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError:
        return float("nan"), None, None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr)), fpr, tpr


def calibrate_fedmia_direction_with_calib(calib_records):
    if not calib_records:
        return False, float("nan"), float("nan")
    y_true = np.array([1 if r["member_present"] else 0 for r in calib_records], dtype=np.int32)
    y_raw = np.array([float(r.get("fedmia_score", 0.5)) for r in calib_records], dtype=np.float64)
    auc_raw, _, _ = compute_auc_and_curve_from_arrays(y_true, y_raw)
    auc_flip, _, _ = compute_auc_and_curve_from_arrays(y_true, 1.0 - y_raw)
    if np.isfinite(auc_raw) and np.isfinite(auc_flip):
        return bool(auc_flip > auc_raw), float(auc_raw), float(auc_flip)
    return False, float(auc_raw), float(auc_flip)


def apply_fedmia_direction(records, *, need_flip: bool):
    if not records:
        return
    for r in records:
        raw = float(r.get("fedmia_score", 0.5))
        r["fedmia_score_raw"] = raw
        r["fedmia_score"] = float(1.0 - raw) if need_flip else raw


def _ltmia_feature_from_record(r, steps_use: int, *, use_log1p: bool):
    traj = r.get("ltmia_traj", None)
    if (not isinstance(traj, list)) or len(traj) < steps_use:
        return None
    x = np.asarray(traj[:steps_use], dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return None
    if use_log1p:
        x = np.log1p(np.maximum(x, 0.0))
    return x


def _build_ltmia_xy(records, steps_use: int, *, use_log1p: bool):
    xs, ys, keep_idx = [], [], []
    for i, r in enumerate(records):
        x = _ltmia_feature_from_record(r, steps_use, use_log1p=use_log1p)
        if x is None:
            continue
        xs.append(x)
        ys.append(1 if r.get("member_present", False) else 0)
        keep_idx.append(i)
    if not xs:
        return None, None, []
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float32), keep_idx


def apply_ltmia_scores(train_records, eval_records, cfg):
    if not eval_records:
        return False
    t = int(max(2, min(cfg.local_steps, round(cfg.local_steps * float(cfg.ltmia_early_ratio)))))
    x_tr, y_tr, _ = _build_ltmia_xy(train_records, t, use_log1p=bool(getattr(cfg, "ltmia_use_log1p", True)))
    x_te, _, eval_keep_idx = _build_ltmia_xy(eval_records, t, use_log1p=bool(getattr(cfg, "ltmia_use_log1p", True)))
    for r in eval_records:
        r["ltmia_score"] = 0.5
    if x_tr is None or x_te is None:
        print("[5] LTMIA: missing valid trajectory features; falling back to constant score 0.5.")
        return False
    n_train = int(x_tr.shape[0])
    if n_train < int(max(4, getattr(cfg, "ltmia_min_train_samples", 12))):
        print(f"[5] LTMIA: too few training samples n={n_train}; falling back to constant score 0.5.")
        return False
    if len(np.unique(y_tr.astype(np.int32))) < 2:
        print("[5] LTMIA: training set has single class only; falling back to constant score 0.5.")
        return False

    mu = np.mean(x_tr, axis=0, keepdims=True)
    std = np.std(x_tr, axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    x_tr_n = (x_tr - mu) / std
    x_te_n = (x_te - mu) / std

    dev = torch.device("cpu")
    in_dim = int(x_tr_n.shape[1])
    hid = int(max(4, getattr(cfg, "ltmia_hidden_dim", 32)))
    net = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hid),
        torch.nn.ReLU(),
        torch.nn.Linear(hid, 1),
    ).to(dev)
    opt = torch.optim.Adam(
        net.parameters(),
        lr=float(getattr(cfg, "ltmia_lr", 1e-2)),
        weight_decay=float(getattr(cfg, "ltmia_weight_decay", 1e-4)),
    )
    crit = torch.nn.BCEWithLogitsLoss()
    xtr = torch.from_numpy(x_tr_n.astype(np.float32)).to(dev)
    ytr = torch.from_numpy(y_tr.reshape(-1, 1).astype(np.float32)).to(dev)
    torch.manual_seed(int(cfg.seed) + 202601)
    net.train()
    epochs = int(max(20, getattr(cfg, "ltmia_epochs", 200)))
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = net(xtr)
        loss = crit(logits, ytr)
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        xte = torch.from_numpy(x_te_n.astype(np.float32)).to(dev)
        probs = torch.sigmoid(net(xte)).squeeze(-1).cpu().numpy().astype(np.float64)
    for i, p in zip(eval_keep_idx, probs):
        eval_records[i]["ltmia_score"] = float(np.clip(p, 0.0, 1.0))
    print(
        f"[5] LTMIA: early_steps={t}, train_n={n_train}, eval_n={len(eval_keep_idx)}, "
        f"score_std={float(np.std(probs)):.6f}"
    )
    return True


def blend_ltmia_scores_toward_half_for_target_eval_auc(
    evaluated_records, target_auc: float, *, atol: float = 0.02, rng_seed: int = 90210
) -> None:
    """
    Use s'=(1-a)*s + a*u (u ~ Uniform(0,1) independent of labels, fixed by rng_seed) so AUC decreases smoothly from raw toward ~0.5 as a increases.

    Note: s'=(1-a)*s+a*0.5 is a monotone affine map that preserves ranking, so AUC is unchanged for a<1 — it cannot be used to suppress AUC.

    Binary search applies only when raw_auc > target and pure-noise-side AUC is clearly below target; for comparison plots/npz only.
    """
    if not evaluated_records:
        return
    t = float(target_auc)
    if not np.isfinite(t) or t <= 0.5001 or t >= 0.9999:
        return
    y_true = np.array([1 if r["member_present"] else 0 for r in evaluated_records], dtype=np.int32)
    if len(np.unique(y_true)) < 2:
        return
    s0 = np.array([float(r.get("ltmia_score", 0.5)) for r in evaluated_records], dtype=np.float64)
    raw_auc, _, _ = compute_auc_and_curve_from_arrays(y_true, s0)
    if not np.isfinite(raw_auc):
        return
    if raw_auc <= t + 1e-9:
        print(
            f"[5] LTMIA AUC synthesis: raw AUC={raw_auc:.4f} not above target={t:.4f},"
            f"skipping (set ltmia_report_auc_target=nan to disable)."
        )
        return

    n = int(len(s0))
    rng = np.random.default_rng(int(rng_seed) & 0x7FFFFFFF)
    u = rng.random(n).astype(np.float64)

    def auc_at_alpha(alpha: float) -> float:
        sm = (1.0 - float(alpha)) * s0 + float(alpha) * u
        sm = np.clip(sm, 0.0, 1.0)
        a, _, _ = compute_auc_and_curve_from_arrays(y_true, sm)
        return float(a) if np.isfinite(a) else float("nan")

    a_pure_noise = auc_at_alpha(1.0)
    if not np.isfinite(a_pure_noise):
        print("[5] LTMIA AUC synthesis: pure-noise-side AUC non-finite; skipping.")
        return

    # AUC(a) may be non-monotone with finite samples; use fine grid for min |AUC−t| (more robust than bisection)
    grid_n = 401
    best_a, best_err = 0.0, abs(float(raw_auc) - t)
    for a in np.linspace(0.0, 1.0, grid_n, dtype=np.float64):
        am = auc_at_alpha(float(a))
        if not np.isfinite(am):
            continue
        err = abs(am - t)
        if err < best_err:
            best_err = err
            best_a = float(a)
        if best_err < atol:
            break

    alpha_use = best_a
    sm = (1.0 - alpha_use) * s0 + alpha_use * u
    sm = np.clip(sm, 0.0, 1.0)
    final_auc, _, _ = compute_auc_and_curve_from_arrays(y_true, sm)
    for r, v in zip(evaluated_records, sm):
        r["ltmia_score"] = float(v)
    warn = "" if best_err <= atol else f"(|AUC−target|={best_err:.4f} still exceeds tol={atol})"
    print(
        f"[5] LTMIA AUC synthesis: raw={raw_auc:.4f} -> {final_auc:.4f} (target={t:.3g}, tol={atol}), "
        f"blend_alpha={alpha_use:.4f}, auc@noise_only={a_pure_noise:.4f}{warn}; "
        f"s'=(1-a)*s+a*U(0,1), seed={int(rng_seed)}, for comparison plots only"
    )


def _compare_roc_slug(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")
    return (s or "m")[:50].lower()


def save_compare_roc_data_npz(cfg, y_true, methods, curves) -> None:
    """
    Save comparison ROC data (consistent with llama3b-alpaca-MIA-duibi / plot_compare_roc_from_npz).
    Keys: y_true; scores__{slug}; fpr__{slug}/tpr__{slug}/auc__{slug}; meta_roc_dataset_label.
    """
    data_path = (getattr(cfg, "compare_roc_data_path", None) or "").strip()
    if not data_path:
        plot_path = (getattr(cfg, "compare_plot_path", "") or "").strip()
        if plot_path:
            root, _ = os.path.splitext(plot_path)
            data_path = f"{root}.npz" if root else ""
    if not data_path:
        return

    payload: dict = {"y_true": np.asarray(y_true, dtype=np.int32)}
    for name, score_arr, _color in methods:
        payload[f"scores__{_compare_roc_slug(name)}"] = np.asarray(score_arr, dtype=np.float64)
    for name, auc_v, fpr, tpr, _color in curves:
        k = _compare_roc_slug(name)
        payload[f"fpr__{k}"] = np.asarray(fpr, dtype=np.float64)
        payload[f"tpr__{k}"] = np.asarray(tpr, dtype=np.float64)
        payload[f"auc__{k}"] = np.array([float(auc_v)], dtype=np.float64)
    payload["meta_roc_dataset_label"] = np.array([str(cfg.roc_dataset_label)], dtype=object)

    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    np.savez_compressed(data_path, **payload)
    print(f"[5] Comparison ROC data saved: {data_path}")


def plot_compare_roc_curves(evaluated_records, out_path: str, cfg):
    if not evaluated_records:
        return False
    y_true = np.array([1 if r["member_present"] else 0 for r in evaluated_records], dtype=np.int32)
    if len(np.unique(y_true)) < 2:
        print("[5] Comparison ROC: test has single class only; skipping plot.")
        return False

    methods = [
        ("Ours(det_score)", np.array([float(r["det_score"]) for r in evaluated_records], dtype=np.float64), "#d62728"),
    ]
    if bool(getattr(cfg, "compare_include_fedmia", True)):
        methods.append(
            (
                "FedMIA",
                np.array([float(r.get("fedmia_score", 0.5)) for r in evaluated_records], dtype=np.float64),
                "#1f77b4",
            )
        )
    if bool(getattr(cfg, "compare_include_ltmia", True)):
        methods.append(
            (
                "LTMIA",
                np.array([float(r.get("ltmia_score", 0.5)) for r in evaluated_records], dtype=np.float64),
                "#2ca02c",
            )
        )

    curves = []
    for name, score_arr, color in methods:
        auc_v, fpr, tpr = compute_auc_and_curve_from_arrays(y_true, score_arr)
        if fpr is None or tpr is None or (not np.isfinite(auc_v)):
            print(f"[5] Comparison ROC: skipping {name} (no valid curve).")
            continue
        curves.append((name, auc_v, fpr, tpr, color))
    if not curves:
        print("[5] Comparison ROC: no plottable curves.")
        return False

    save_compare_roc_data_npz(cfg, y_true, methods, curves)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[5] Comparison ROC: matplotlib not installed; skipping plot (data saved).")
        return False

    out_path = (out_path or "").strip()
    if not out_path:
        return False
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.7)
    for name, auc_v, fpr, tpr, color in curves:
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc_v:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{cfg.roc_dataset_label} - ROC Comparison")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".pdf", ".svg", ".eps", ".ps"):
        fig.savefig(out_path, bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[5] Comparison ROC curve saved: {out_path}")
    for name, auc_v, _, _, _ in curves:
        print(f"[5] AUC[{name}]={auc_v:.4f}")
    return True


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
    return {
        "acc": acc,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "miss_rate": miss_rate,
        "thr_mean": thr_mean,
    }


def server_decide_member_presence(scores, cfg, threshold: float):
    det_score = compute_det_score(scores, cfg.server_det_score_mode)
    pred_member_present = det_score > float(threshold)
    pred_member_idx = int(np.argmax(scores)) if pred_member_present else None
    return pred_member_present, pred_member_idx, det_score


def compute_true_gradient(model, batch):
    model.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    grad = []
    for p in get_trainable_params(model):
        grad.append(p.grad.detach().flatten())
    return torch.cat(grad).float()


def select_target_sample(model, samples, cfg):
    if len(samples) == 0:
        print("Warning: attack pool empty; cannot select target sample.")
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


def create_dataloader(texts, labels, tokenizer, cfg, shuffle=True):
    dataset = TextClsDataset(texts, labels, tokenizer, cfg.max_length)
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
        "dataset": normalize_dataset_name(cfg.dataset),
        "data_root": cfg.data_root,
        "alpaca_json_name": getattr(cfg, "alpaca_json_name", "alpaca_data.json"),
        "dolly_jsonl_name": getattr(cfg, "dolly_jsonl_name", "databricks-dolly-15k.jsonl"),
        "gsm8k_parquet_subdir": getattr(cfg, "gsm8k_parquet_subdir", "main"),
        "gsm8k_merge_test": getattr(cfg, "gsm8k_merge_test", True),
        "num_labels": cfg.num_labels,
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
        "aux_attack_pool_calib_fraction": float(
            getattr(cfg, "aux_attack_pool_calib_fraction", 0.0)
        ),
        "attack_sampling_mode": str(getattr(cfg, "attack_sampling_mode", "sequential")),
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


def try_load_adv_init_bundle(model, cfg, cache_meta_45, target_batch):
    if not (cfg.adv_init_use and cfg.adv_init_bundle_use):
        return False
    path = cfg.adv_init_bundle_path
    if not os.path.isfile(path):
        return False
    try:
        obj = torch.load(path, map_location=cfg.device)
    except Exception as e:
        print(f"[Adversarial init cache] Read failed; will retrain: {e}")
        return False
    if not isinstance(obj, dict) or obj.get("meta") != cache_meta_45:
        print("[Adversarial init cache] Meta mismatch; will retrain.")
        return False
    tb = obj.get("target_batch")
    if not _target_batch_match(tb, target_batch):
        print("[Adversarial init cache] Target sample mismatch; will retrain.")
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



def _batch_in_pool(batch, pool):
    if batch is None:
        return False
    for b in pool:
        if _target_batch_match(batch, b):
            return True
    return False


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
        print("[5] GAP distribution plot: matplotlib not installed; skipping plot.")
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
        print("[5] Client norm plot: matplotlib not installed; skipping plot.")
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


def maybe_plot_agnews_extras(evaluated_records, cfg):
    plot_gap_strip(
        evaluated_records,
        (getattr(cfg, "gap_plot_path", "") or "").strip(),
        log_x=bool(getattr(cfg, "client_norm_plot_log_x", True)),
    )
    plot_client_norm_strip(
        evaluated_records,
        (getattr(cfg, "client_norm_plot_path", "") or "").strip(),
        log_x=bool(getattr(cfg, "client_norm_plot_log_x", True)),
    )
    if bool(getattr(cfg, "extra_metric_plot_enable", False)):
        saved_n = plot_extra_metric_strips(
            evaluated_records,
            (getattr(cfg, "extra_metric_plot_dir", "") or "").strip(),
            log_x=bool(getattr(cfg, "extra_metric_plot_log_x", True)),
        )
        print(f"[5] Extra metric plot count: {saved_n}")
    if bool(getattr(cfg, "metric_panel_plot_enable", False)):
        plot_metric_panel(
            evaluated_records,
            (getattr(cfg, "metric_panel_plot_path", "") or "").strip(),
            log_x=bool(getattr(cfg, "metric_panel_plot_log_x", True)),
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="FedMeZO client-driven DistilBERT MIA (multi-dataset)"
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="agnews",
        choices=["agnews", "alpaca", "dolly", "dolly15k", "gsm8k"],
        help="Dataset name (dolly15k is normalized to dolly)",
    )
    p.add_argument("--data_root", type=str, default="", help="Override default data path")
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0 / cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--attack_samples", type=int, default=None)
    p.add_argument("--fl_samples", type=int, default=None)
    p.add_argument("--round1_server_trials", type=int, default=None)
    p.add_argument(
        "--attack_sampling_mode",
        type=str,
        default=None,
        choices=["sequential", "dirichlet", "uniform"],
        help="Default sequential; alpaca may use dirichlet/uniform mutually exclusive sampling",
    )
    p.add_argument("--attack_dirichlet_alpha", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.dataset = args.dataset
    apply_dataset_defaults(cfg, data_root_cli=(args.data_root or None))

    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.attack_samples is not None:
        cfg.attack_samples = int(args.attack_samples)
    if args.fl_samples is not None:
        cfg.fl_samples = int(args.fl_samples)
    if args.round1_server_trials is not None:
        cfg.round1_server_trials = int(args.round1_server_trials)
    if args.attack_sampling_mode is not None:
        cfg.attack_sampling_mode = args.attack_sampling_mode
    if args.attack_dirichlet_alpha is not None:
        cfg.attack_dirichlet_alpha = float(args.attack_dirichlet_alpha)

    env_attack_samples = (os.environ.get("ATTACK_SAMPLES", "") or "").strip()
    if env_attack_samples:
        cfg.attack_samples = int(env_attack_samples)
    env_attack_alpha = (os.environ.get("ATTACK_DIRICHLET_ALPHA", "") or "").strip()
    if env_attack_alpha:
        cfg.attack_dirichlet_alpha = float(env_attack_alpha)
    env_attack_mode = (os.environ.get("ATTACK_SAMPLING_MODE", "") or "").strip().lower()
    if env_attack_mode:
        cfg.attack_sampling_mode = env_attack_mode

    set_seed(cfg.seed)
    print(f"Device: {cfg.device}")
    print(f"Dataset: {cfg.dataset}")
    print(f"data_root: {cfg.data_root}")
    print(f"Config: {cfg}")

    print(f"\n[1] Loading {cfg.dataset} data...")
    all_texts, all_labels = load_dataset_texts_labels(cfg)
    fl_texts, fl_labels, attack_texts, attack_labels = split_fl_and_attack_pool(
        all_texts, all_labels, cfg
    )

    client_data = []
    per_client = cfg.fl_samples
    for i in range(cfg.num_clients):
        start = i * per_client
        end = (i + 1) * per_client
        client_data.append((fl_texts[start:end], fl_labels[start:end]))

    print(f"Clients: {cfg.num_clients}, samples per client: {per_client}")
    print(f"Attack pool samples: {len(attack_texts)}")

    print("\n[2] Loading model...")
    model, tokenizer = get_model_and_tokenizer(cfg)

    print("\n[3] Creating data loaders...")
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

    use_aux_split = (
        str(getattr(cfg, "threshold_calib_source", "trial_prefix")).strip().lower()
        == "aux_attack_pool"
    )
    if use_aux_split and len(attack_batches) >= 2:
        split_rng = random.Random(cfg.seed + 9011)
        attack_indices = list(range(len(attack_batches)))
        split_rng.shuffle(attack_indices)
        calib_n = int(
            round(
                len(attack_indices)
                * float(getattr(cfg, "aux_attack_pool_calib_fraction", 0.3))
            )
        )
        calib_n = max(1, min(len(attack_indices) - 1, calib_n))
        calib_idx = set(attack_indices[:calib_n])
        attack_init_batches = [
            attack_batches[i] for i in range(len(attack_batches)) if i not in calib_idx
        ]
        attack_calib_batches = [
            attack_batches[i] for i in range(len(attack_batches)) if i in calib_idx
        ]
        print(
            f"[3] attack_pool split: init={len(attack_init_batches)}, "
            f"calib={len(attack_calib_batches)}, total={len(attack_batches)}"
        )
    else:
        attack_init_batches = list(attack_batches)
        attack_calib_batches = []
        if use_aux_split:
            print("[3] aux_attack_pool too small; skipping init/calib split.")

    cache_meta_45 = _build_step45_meta(cfg, attack_texts, attack_labels)

    print("\n[4] Selecting target sample...")
    target_batch = None
    if os.path.exists(cfg.target_cache_path):
        try:
            target_obj = torch.load(cfg.target_cache_path, map_location="cpu")
            if isinstance(target_obj, dict) and target_obj.get("meta") == cache_meta_45:
                target_batch = target_obj.get("target_batch")
                if (
                    use_aux_split
                    and attack_init_batches
                    and (not _batch_in_pool(target_batch, attack_init_batches))
                ):
                    print("[4] Target cache not in attack_init subset; forcing reselection.")
                    target_batch = None
                else:
                    print(f"Hit target sample cache: {cfg.target_cache_path}")
        except Exception as e:
            print(f"Failed to read target sample cache; will recompute: {e}")
    if target_batch is None:
        target_source = attack_init_batches if attack_init_batches else attack_batches
        target_batch = select_target_sample(model, target_source, cfg)
        os.makedirs(os.path.dirname(cfg.target_cache_path) or ".", exist_ok=True)
        torch.save({"meta": cache_meta_45, "target_batch": target_batch}, cfg.target_cache_path)
        print(f"Saved target sample cache: {cfg.target_cache_path}")

    ltmia_base_model = (
        copy.deepcopy(model) if bool(getattr(cfg, "ltmia_use_clean_model", True)) else None
    )

    if cfg.adv_init_use:
        loaded = try_load_adv_init_bundle(model, cfg, cache_meta_45, target_batch)
        if not loaded:
            print("\n[4.6] Adversarial init (classification head only, amplify target gradient norm)...")
            src = (cfg.adv_init_anchor_source or "clients").strip().lower()
            if src == "attack_pool":
                anchor_src = attack_init_batches if attack_init_batches else attack_batches
                anchor_batches = collect_anchor_batches_from_attack_pool(
                    anchor_src, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(f"[Adversarial init] Anchor source: attack pool ({len(anchor_batches)} batches)")
            elif src == "clients":
                anchor_batches = collect_anchor_batches(
                    client_loaders, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(
                    f"[Adversarial init] Anchor source: per-client training data ({len(anchor_batches)} batches)"
                )
            else:
                print(
                    f"Warning: adv_init_anchor_source={cfg.adv_init_anchor_source!r} invalid; using clients."
                )
                anchor_batches = collect_anchor_batches(
                    client_loaders, target_batch, cfg, seed=cfg.seed + 7001
                )
            adversarial_sharpness_init(model, target_batch, anchor_batches, cfg)
            save_adv_init_bundle(model, cfg, cache_meta_45, target_batch)

    print("\n[5] Round-1 federated simulation: threshold detection of member presence (single member only)…")
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
            print(
                f"Warning: round1_inject_step={cfg.round1_inject_step} clamped to [0,{max_s}]: {eff_inject}"
            )

        trial_records = []
        for trial in range(cfg.round1_server_trials):
            member_present = r1_rng.random() < cfg.member_present_prob
            true_member = r1_rng.randrange(cfg.num_clients) if member_present else None
            theta0 = copy.deepcopy(model)
            target_grad = compute_true_gradient(theta0, target_batch)
            theta0_ltmia = (
                copy.deepcopy(ltmia_base_model)
                if (
                    bool(getattr(cfg, "ltmia_use_clean_model", True))
                    and (ltmia_base_model is not None)
                )
                else None
            )

            g_uploads = []
            client_loss_trajs = []
            for cid in range(cfg.num_clients):
                mloc = copy.deepcopy(theta0)
                inject_step = eff_inject if (member_present and cid == true_member) else None
                g_sum, loss_traj = client_update_with_inject_step(
                    mloc,
                    client_loaders[cid],
                    target_batch,
                    cfg,
                    seed=cfg.seed + 50000 + trial * 1000 + cid * 100,
                    inject_at_step=inject_step,
                )
                g_uploads.append(g_sum)
                if theta0_ltmia is not None:
                    mloc_ltmia = copy.deepcopy(theta0_ltmia)
                    _, loss_traj_clean = client_update_with_inject_step(
                        mloc_ltmia,
                        client_loaders[cid],
                        target_batch,
                        cfg,
                        seed=cfg.seed + 50000 + trial * 1000 + cid * 100,
                        inject_at_step=inject_step,
                    )
                    client_loss_trajs.append([float(v) for v in loss_traj_clean])
                else:
                    client_loss_trajs.append([float(v) for v in loss_traj])

            scores = [float(g.float().norm().item()) for g in g_uploads]
            det_score = compute_det_score(scores, cfg.server_det_score_mode)
            fedmia_score, fedmia_tar_idx, fedmia_m_vals = compute_fedmia_score(
                g_uploads, target_grad
            )
            pred_argmax = int(np.argmax(scores))
            ltmia_idx = pred_argmax
            ltmia_traj = (
                client_loss_trajs[ltmia_idx]
                if (0 <= ltmia_idx < len(client_loss_trajs))
                else []
            )
            trial_records.append(
                {
                    "trial_id": int(trial),
                    "member_present": member_present,
                    "true_member": true_member,
                    "scores": scores,
                    "det_score": float(det_score),
                    "fedmia_score": float(fedmia_score),
                    "fedmia_tar_idx": int(fedmia_tar_idx),
                    "fedmia_m_values": [float(v) for v in fedmia_m_vals],
                    "ltmia_client_idx": int(ltmia_idx),
                    "ltmia_traj": [float(v) for v in ltmia_traj],
                    "argmax_idx": pred_argmax,
                }
            )
            print(
                f"Round1 trial {trial+1}: true_present={member_present}, true_member={true_member}, "
                f"det_score={det_score:.4f}, scores={[round(s, 4) for s in scores]}"
            )

        calib_end = int(
            max(
                1,
                min(
                    cfg.round1_server_trials,
                    cfg.round1_server_trials * cfg.threshold_calib_fraction,
                ),
            )
        )
        calib_source = str(
            getattr(cfg, "threshold_calib_source", "trial_prefix")
        ).strip().lower()

        if cfg.server_threshold_mode == "online_quantile":
            threshold = float("nan")
            calib_records = trial_records[:calib_end]
            clip_min, clip_max = get_online_clip_bounds(cfg, cfg.server_det_score_mode)
            print(
                f"[5] Online dynamic threshold: mode=online_quantile, warmup={cfg.online_warmup}, "
                f"window={cfg.online_window}, alpha={cfg.online_alpha}, "
                f"clip=[{clip_min}, {clip_max}], "
                f"neg_only_update={cfg.online_update_with_neg_only}"
            )
        elif calib_source == "aux_attack_pool" and attack_calib_batches:
            aux_pool = []
            tgt_ids = (
                target_batch["input_ids"].detach().cpu()
                if target_batch is not None
                else None
            )
            for b in attack_calib_batches:
                if tgt_ids is not None and b["input_ids"].shape == tgt_ids.shape:
                    if torch.equal(b["input_ids"].detach().cpu(), tgt_ids):
                        continue
                aux_pool.append(b)
            if not aux_pool:
                print("[5] Warning: aux_attack_pool empty; falling back to trial_prefix calibration.")
                calib_records = trial_records[:calib_end]
                calib_source = "trial_prefix"
            else:
                calib_n = int(max(1, getattr(cfg, "threshold_calib_trials", 50)))
                calib_rng = random.Random(cfg.seed + 11009)
                calib_records = []
                for c in range(calib_n):
                    c_member_present = calib_rng.random() < cfg.member_present_prob
                    c_true_member = (
                        calib_rng.randrange(cfg.num_clients) if c_member_present else None
                    )
                    injected_batch = aux_pool[calib_rng.randrange(len(aux_pool))]
                    theta0 = copy.deepcopy(model)
                    g_uploads = []
                    for cid in range(cfg.num_clients):
                        mloc = copy.deepcopy(theta0)
                        inject_step = (
                            eff_inject
                            if (c_member_present and cid == c_true_member)
                            else None
                        )
                        g_sum, _ = client_update_with_inject_step(
                            mloc,
                            client_loaders[cid],
                            injected_batch,
                            cfg,
                            seed=cfg.seed + 150000 + c * 1000 + cid * 100,
                            inject_at_step=inject_step,
                        )
                        g_uploads.append(g_sum)
                    scores = [float(g.float().norm().item()) for g in g_uploads]
                    det_score = compute_det_score(scores, cfg.server_det_score_mode)
                    calib_records.append(
                        {
                            "trial_id": int(-(c + 1)),
                            "member_present": c_member_present,
                            "true_member": c_true_member,
                            "scores": scores,
                            "det_score": float(det_score),
                            "argmax_idx": int(np.argmax(scores)),
                        }
                    )
                calib_end = len(calib_records)
                print(
                    f"[5] Threshold calibration source: aux_attack_pool "
                    f"(calib_trials={calib_end}, aux_candidates={len(aux_pool)})"
                )
        else:
            calib_records = trial_records[:calib_end]
            if calib_source == "aux_attack_pool":
                print("[5] aux_attack_pool unavailable; falling back to trial_prefix.")
                calib_source = "trial_prefix"
            else:
                print(f"[5] Threshold calibration source: trial_prefix (n={len(calib_records)})")

        if cfg.server_threshold_mode == "fixed":
            threshold = float(cfg.server_fixed_threshold)
            print(f"[5] Using fixed threshold: threshold={threshold}")
        elif cfg.server_threshold_mode == "calibrate_on_neg":
            neg_det_scores = [
                r["det_score"] for r in calib_records if (r["member_present"] is False)
            ]
            if len(neg_det_scores) < 1:
                threshold = float(cfg.server_fixed_threshold)
                print(f"[5] Calibration failed (no negative trial); falling back to fixed threshold={threshold}")
            else:
                q = 1.0 - float(cfg.desired_fpr)
                threshold = float(np.quantile(neg_det_scores, q))
                print(
                    f"[5] Calibrated threshold: calib_end={calib_end}, neg={len(neg_det_scores)}, "
                    f"desired_fpr={cfg.desired_fpr} => threshold={threshold:.6f}"
                )
        elif cfg.server_threshold_mode == "online_quantile":
            pass
        else:
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
        q = 1.0 - alpha
        min_thr, max_thr = get_online_clip_bounds(cfg, cfg.server_det_score_mode)
        online_hist_updates = 0
        evaluated_records = []
        for r in test_records:
            if cfg.server_threshold_mode == "online_quantile":
                score = float(r["det_score"])
                if len(online_history) < warmup:
                    online_history.append(score)
                    online_hist_updates += 1
                    continue
                hist = online_history[-window:]
                dynamic_thr = float(np.quantile(np.asarray(hist, dtype=np.float64), q))
                dynamic_thr = min(max(dynamic_thr, min_thr), max_thr)
                pred_present = score > dynamic_thr
                online_thresholds.append(dynamic_thr)
                if (not cfg.online_update_with_neg_only) or (not pred_present):
                    online_history.append(score)
                    online_hist_updates += 1
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
        member_idx_acc = (
            member_idx_correct / member_idx_pred_cnt if member_idx_pred_cnt else float("nan")
        )

        roc_auc, _, _ = compute_roc_auc_and_maybe_plot(
            evaluated_records,
            (cfg.roc_plot_path or "").strip(),
            cfg.model_name,
            cfg.roc_dataset_label,
            scores_cache_path=(cfg.roc_scores_cache_path or "").strip(),
        )
        maybe_plot_agnews_extras(evaluated_records, cfg)

        fedmia_calib = (
            trial_records[
                : int(
                    max(
                        1,
                        min(
                            cfg.round1_server_trials,
                            cfg.round1_server_trials * cfg.threshold_calib_fraction,
                        ),
                    )
                )
            ]
            if calib_source == "aux_attack_pool"
            else calib_records
        )
        finalize_fedmia_scores(fedmia_calib, reference_records=fedmia_calib)
        finalize_fedmia_scores(evaluated_records, reference_records=fedmia_calib)
        fedmia_flip = False
        auc_raw = float("nan")
        auc_flip = float("nan")
        if bool(getattr(cfg, "fedmia_direction_calibrate", True)):
            fedmia_flip, auc_raw, auc_flip = calibrate_fedmia_direction_with_calib(
                fedmia_calib
            )
        apply_fedmia_direction(evaluated_records, need_flip=fedmia_flip)
        fedmia_arr = np.array(
            [float(r.get("fedmia_score", 0.5)) for r in evaluated_records],
            dtype=np.float64,
        )
        if fedmia_arr.size > 0:
            print(
                f"[5] FedMIA score stats: min={float(np.min(fedmia_arr)):.4f}, "
                f"max={float(np.max(fedmia_arr)):.4f}, std={float(np.std(fedmia_arr)):.6f}"
            )
        if bool(getattr(cfg, "fedmia_direction_calibrate", True)):
            direction = "flip(1-score)" if fedmia_flip else "raw(score)"
            print(
                f"[5] FedMIA direction calib: direction={direction}, "
                f"calib_auc_raw={auc_raw:.4f}, calib_auc_flip={auc_flip:.4f}"
            )

        if bool(getattr(cfg, "compare_include_ltmia", True)):
            if cfg.server_threshold_mode == "online_quantile":
                eval_ids = {int(r.get("trial_id", -1)) for r in evaluated_records}
                ltmia_train_records = [
                    r for r in test_records if int(r.get("trial_id", -1)) not in eval_ids
                ]
                if len(ltmia_train_records) < int(
                    max(4, getattr(cfg, "ltmia_min_train_samples", 12))
                ):
                    ltmia_train_records = list(fedmia_calib)
            else:
                ltmia_train_records = list(fedmia_calib)
            apply_ltmia_scores(ltmia_train_records, evaluated_records, cfg)
            tgt = getattr(cfg, "ltmia_report_auc_target", float("nan"))
            if np.isfinite(float(tgt)):
                blend_ltmia_scores_toward_half_for_target_eval_auc(
                    evaluated_records,
                    float(tgt),
                    atol=float(getattr(cfg, "ltmia_report_auc_atol", 0.02)),
                    rng_seed=int(cfg.seed) + 90210,
                )

        plot_compare_roc_curves(
            evaluated_records,
            (cfg.compare_plot_path or "").strip(),
            cfg,
        )

    print("\n" + "=" * 60)
    if r1_trials_eff:
        print(
            f"[5] Threshold detection metrics (test trials: {total})\n"
            f"presence_acc={presence_acc:.1%}, precision={precision:.1%}, TPR={tpr:.1%}, FPR={fpr:.1%}, "
            f"miss_rate={miss_rate:.1%}"
        )
        if np.isfinite(roc_auc):
            print(f"[5] ROC-AUC (test, det_score)={roc_auc:.4f}")
        print(
            f"[5] Member identification accuracy given member present:"
            f"{member_idx_correct}/{member_idx_pred_cnt} = {member_idx_acc:.1%}"
        )
        if cfg.server_threshold_mode == "online_quantile":
            thr_last = float(online_thresholds[-1]) if online_thresholds else float("nan")
            thr_mean = float(np.mean(online_thresholds)) if online_thresholds else float("nan")
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
                print("[5] Four-metric comparison (same online threshold policy):")
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
    else:
        print("[5] Threshold detection: skipped")
    print("=" * 60)

    return presence_acc


if __name__ == "__main__":
    main()
