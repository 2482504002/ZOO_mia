"""
FedMeZO client-driven membership inference experiment (multi-dataset + Open-Llama-3B + PEFT LoRA).

Usage:
  python MIA_llama3b.py --dataset agnews
  python MIA_llama3b.py --dataset alpaca
  python MIA_llama3b.py --dataset dolly
  python MIA_llama3b.py --dataset gsm8k

Task types:
  agnews              -> AutoModelForSequenceClassification + LoRA SEQ_CLS
  alpaca/dolly/gsm8k  -> AutoModelForCausalLM + LoRA CAUSAL_LM

Default dataset paths (override with --data_root):
  agnews  -> /home/zhike/JWH/ZOO_MIA/datasets/agnews (local parquet / h5, no network)
  alpaca  -> /home/zhike/JWH/ZOO_MIA/datasets/alpaca_data.json
  dolly   -> /home/zhike/JWH/ZOO_MIA/datasets/dolly15k/databricks-dolly-15k.jsonl
  gsm8k   -> /home/zhike/JWH/ZOO_MIA/datasets/gsm8k (local parquet only, no HF fallback)
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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

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
    model_name: str = "/home/zhike/JWH/model/open_llama_3b_v2/"
    max_length: int = 512
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"

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
    target_cache_path: str = ""

    round1_server_trials: int = 100
    round1_inject_step: int = 0

    member_present_prob: float = 0.5
    server_det_score_mode: str = "gap"
    server_threshold_mode: str = "online_quantile"
    threshold_calib_fraction: float = 0.5
    server_fixed_threshold: float = 0.0
    online_warmup: int = 20
    online_window: int = 40
    online_alpha: float = 0.1
    online_min_threshold: float = 10000
    online_max_threshold: float = 1000000
    online_update_with_neg_only: bool = True
    online_auto_clip_by_mode: bool = True
    online_init_with_warmup_low_cluster: bool = True
    roc_plot_path: str = ""
    roc_scores_cache_path: str = ""
    roc_dataset_label: str = ""

    adv_init_use: bool = True
    adv_init_steps: int = 500
    adv_init_lr: float = 1e-4
    adv_init_w_target: float = 1.5
    adv_init_w_anchor: float = 0.15
    adv_init_anchor_power: float = 2.0
    adv_init_w_anchor_max: float = 0.1
    adv_init_clip_grad_norm: float = 0.5
    adv_init_anchors_per_client: int = 50
    adv_init_anchor_subset_size: int = 4
    adv_init_anchor_source: str = "attack_pool"
    adv_init_log_every: int = 10
    adv_init_bundle_path: str = ""
    adv_init_bundle_use: bool = True
    adv_init_gradient_checkpointing: bool = False


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


def is_seq_cls(cfg) -> bool:
    return normalize_dataset_name(cfg.dataset) == "agnews"


def apply_dataset_defaults(cfg: Config, *, data_root_cli: Optional[str] = None) -> Config:
    """Fill data_root / max_length / output cache paths from --dataset to avoid cross-dataset overwrites."""
    ds = normalize_dataset_name(cfg.dataset)
    cfg.dataset = ds

    defaults = {
        "agnews": {
            "data_root": os.path.join(_REPO_DATASETS, "agnews"),
            "max_length": 128,
            "num_labels": 4,
            "attack_samples": 500,
            "round1_server_trials": 1000,
            "adv_init_steps": 100,
            "adv_init_lr": 5e-4,
            "adv_init_w_target": 1.0,
            "adv_init_w_anchor": 0.4,
            "adv_init_w_anchor_max": 0.12,
            "roc_dataset_label": "AG News (Llama3B)",
            "target_cache_path": "outputs/llama3b_agnews_mia_target_cache.pt",
            "adv_init_bundle_path": "outputs/llama3b_agnews_mia_adv_init_bundle.pt",
            "roc_scores_cache_path": "outputs/mia_roc_scores_llama3b_agnews.npz",
            "roc_plot_path": "outputs/mia_roc_llama3b_agnews.png",
        },
        "alpaca": {
            "data_root": os.path.join(_REPO_DATASETS, "alpaca_data.json"),
            "max_length": 512,
            "num_labels": 4,
            "attack_samples": 100,
            "round1_server_trials": 100,
            "adv_init_clip_grad_norm": 0.2,
            "online_alpha": 0.3,
            "roc_dataset_label": "Alpaca (Llama3B)",
            "target_cache_path": "outputs/llama3b_alpaca_mia_target_cache.pt",
            "adv_init_bundle_path": "outputs/llama3b_alpaca_mia_adv_init_bundle.pt",
            "roc_scores_cache_path": "outputs/mia_roc_scores_llama3b_alpaca.npz",
            "roc_plot_path": "outputs/mia_roc_llama3b_alpaca.png",
        },
        "dolly": {
            "data_root": os.path.join(
                _REPO_DATASETS, "dolly15k", "databricks-dolly-15k.jsonl"
            ),
            "max_length": 512,
            "num_labels": 4,
            "attack_samples": 100,
            "round1_server_trials": 100,
            "roc_dataset_label": "Dolly-15k (Llama3B)",
            "target_cache_path": "outputs/llama3b_dolly15k_mia_target_cache.pt",
            "adv_init_bundle_path": "outputs/llama3b_dolly15k_mia_adv_init_bundle.pt",
            "roc_scores_cache_path": "outputs/mia_roc_scores_llama3b_dolly15k.npz",
            "roc_plot_path": "outputs/mia_roc_llama3b_dolly15k.png",
        },
        "gsm8k": {
            "data_root": os.path.join(_REPO_DATASETS, "gsm8k"),
            "max_length": 512,
            "num_labels": 4,
            "attack_samples": 100,
            "round1_server_trials": 100,
            "online_alpha": 0.12,
            "roc_dataset_label": "GSM8K (Llama3B)",
            "target_cache_path": "outputs/llama3b_gsm8k_mia_target_cache.pt",
            "adv_init_bundle_path": "outputs/llama3b_gsm8k_mia_adv_init_bundle.pt",
            "roc_scores_cache_path": "outputs/mia_roc_scores_llama3b_gsm8k.npz",
            "roc_plot_path": "outputs/mia_roc_llama3b_gsm8k.png",
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
    """AG News sequence classification: batch key is label."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class CausalLmDataset(Dataset):
    """Alpaca / Dolly / GSM8K CausalLM: labels=-100 at pad positions."""

    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
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


def _batch_to_dict(batch, cfg):
    """Extract attack dict from DataLoader batch / attack batch (key names branch by task type)."""
    out = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    if is_seq_cls(cfg):
        out["label"] = batch["label"]
    else:
        out["labels"] = batch["labels"]
    return out


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

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
    print(f"[AGNews] Loaded {len(texts)} samples from parquet ({len(parquet_files)} files).")
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
    """Prefer local parquet, then h5; raise explicitly if missing (no silent synthesis / no network download)."""
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
    # Consistent with llama3b-alpaca-MIA.py
    instruction = str(d.get("instruction", "")).strip()
    inp = str(d.get("input", "")).strip()
    output = str(d.get("output", "")).strip()
    return f"Instruction: {instruction}\nInput: {inp}\nOutput: {output}"


def _alpaca_pseudo_label(d: dict, num_classes: int) -> int:
    key = f"{d.get('instruction', '')}\n{d.get('output', '')}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest()[:12], 16)
    return h % num_classes


def load_alpaca_data(cfg):
    path = _resolve_alpaca_json_path(cfg)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or len(raw) == 0:
        raise RuntimeError(f"Alpaca data format invalid or empty: {path}")
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
        print(
            f"Warning: Dolly-15k has only {len(texts)} samples, fewer than required {total_needed}; balanced sampling with available count."
        )
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
    if not paths:
        return []

    def _with_pyarrow():
        import pyarrow.parquet as pq

        rows = []
        for p in paths:
            table = pq.read_table(p, columns=["question", "answer"])
            rows.extend(table.to_pylist())
        return [{"question": r["question"], "answer": r["answer"]} for r in rows]

    def _with_pandas():
        import pandas as pd

        frames = [pd.read_parquet(p, columns=["question", "answer"]) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        return df.to_dict("records")

    try:
        return _with_pyarrow()
    except ImportError:
        pass
    except Exception as e_py:
        try:
            return _with_pandas()
        except Exception:
            raise e_py

    try:
        return _with_pandas()
    except ImportError as e:
        raise ImportError(
            "Reading local GSM8K parquet requires pyarrow or pandas (suggested: pip install pyarrow)"
        ) from e


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


# ---------------------------------------------------------------------------
# Model / ZO / attack helpers
# ---------------------------------------------------------------------------

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

    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    target_modules = list(cfg.lora_target_modules)
    if is_seq_cls(cfg):
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=int(cfg.num_labels),
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        ).to(cfg.device)
        model.config.pad_token_id = tokenizer.pad_token_id
        peft_cfg = LoraConfig(
            task_type="SEQ_CLS",
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_cfg)
        # Consistent with llama3b-agnews-MIA.py: train LoRA parameters only
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameter count (LoRA/SEQ_CLS only): {trainable_params}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model = model.to(cfg.device)
        model.config.pad_token_id = tokenizer.pad_token_id
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
        print(f"Trainable parameter count (LoRA+CausalLM): {trainable_params}")

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
    if "label" in batch and "labels" not in batch:
        outputs = model(
            input_ids=batch["input_ids"].to(dev),
            attention_mask=batch["attention_mask"].to(dev),
        )
        return F.cross_entropy(outputs.logits, batch["label"].to(dev))
    outputs = model(
        input_ids=batch["input_ids"].to(dev),
        attention_mask=batch["attention_mask"].to(dev),
        labels=batch["labels"].to(dev),
    )
    return outputs.loss


def mezo_step(model, batch, params, theta, z, cfg):
    theta_orig = theta.clone()
    # Zeroth-order diff needs scalar loss only; do not build autograd graph for two forwards
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
            b = _batch_to_dict(batch, cfg)
            if tid is not None and b["input_ids"].shape == tid.shape:
                if torch.equal(b["input_ids"].cpu(), tid):
                    continue
            anchors.append(b)
            count += 1
    return anchors


def collect_anchor_batches_from_attack_pool(attack_batches, target_batch, cfg, seed):
    rng = random.Random(seed)
    tid = target_batch["input_ids"].detach().cpu() if target_batch is not None else None
    candidates = []
    for b in attack_batches:
        bdict = _batch_to_dict(b, cfg)
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


def _set_adversarial_grad_checkpointing(model, enable: bool) -> bool:
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
        print(f"[Adversarial init] gradient checkpointing toggle failed (may ignore): {e}")
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
        print("[Adversarial init] No target sample; skipping.")
        return
    params = get_trainable_params(model)
    if not params:
        print("[Adversarial init] No trainable parameters; skipping.")
        return

    gc_on = False
    if getattr(cfg, "adv_init_gradient_checkpointing", False):
        gc_on = _set_adversarial_grad_checkpointing(model, True)
        if gc_on:
            print("[Adversarial init] Enabled gradient checkpointing (adversarial init phase only).")

    model.train()
    opt = torch.optim.Adam(params, lr=cfg.adv_init_lr)
    n_anchor = max(1, len(anchor_batches))
    subset_k = max(
        1,
        min(int(getattr(cfg, "adv_init_anchor_subset_size", n_anchor)), n_anchor),
    )
    if subset_k < len(anchor_batches):
        print(
            f"[Adversarial init] Each step cycles {subset_k}/{len(anchor_batches)} anchors in order for loss."
        )
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
    anchor_power = float(cfg.adv_init_anchor_power)
    nan_happened = False
    for step in tqdm(range(cfg.adv_init_steps), desc="Adversarial init (LoRA)"):
        opt.zero_grad(set_to_none=True)

        n_t = _grad_norm_classifier(model, target_batch, create_graph=True)
        if not torch.isfinite(n_t).item():
            nan_happened = True
            print(f"[Adversarial init] Non-finite n_t: {n_t}; stopping early at step={step+1}.")
            break
        loss = -cfg.adv_init_w_target * n_t
        anchor_norms = []
        if anchor_batches:
            n_all = len(anchor_batches)
            start_idx = (step * subset_k) % max(1, n_all)
            use_idx = [(start_idx + t) % n_all for t in range(subset_k)]
            for j in use_idx:
                ab = anchor_batches[j]
                na = _grad_norm_classifier(model, ab, create_graph=True)
                if not torch.isfinite(na).item():
                    nan_happened = True
                    print(f"[Adversarial init] Non-finite anchor na: {na}; stopping early at step={step+1}.")
                    break
                anchor_norms.append(na)
                loss = loss + (cfg.adv_init_w_anchor / max(1, subset_k)) * na.pow(anchor_power)
            if nan_happened:
                break
            if cfg.adv_init_w_anchor_max > 0 and anchor_norms:
                loss = loss + cfg.adv_init_w_anchor_max * torch.stack(anchor_norms).max()

        if not torch.isfinite(loss).item():
            nan_happened = True
            print(f"[Adversarial init] Non-finite loss: {loss}; stopping early at step={step+1}.")
            break
        loss.backward()

        clip = float(getattr(cfg, "adv_init_clip_grad_norm", 0.0) or 0.0)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)

        for prm in params:
            if prm.grad is not None and not torch.isfinite(prm.grad).all():
                nan_happened = True
                print("[Adversarial init] NaN/Inf in parameter gradients; stopping early.")
                break
        if nan_happened:
            break
        opt.step()

        done = step + 1
        if log_every > 0 and (done % log_every == 0 or done == cfg.adv_init_steps):
            if anchor_batches:
                n_all = len(anchor_batches)
                start_idx = (step * subset_k) % max(1, n_all)
                use_idx = [(start_idx + t) % n_all for t in range(subset_k)]
                eval_anchors = [anchor_batches[j] for j in use_idx]
            else:
                eval_anchors = []

            nt, ma, mxa, Lm = _adv_init_loss_terms(model, target_batch, eval_anchors, cfg)
            print(
                f"[Adversarial init]  step {done:4d}/{cfg.adv_init_steps} (eval subset={len(eval_anchors)}): "
                f"||∇L_target||={nt:.6f}, anchor mean||∇L||={ma:.6f}, max||∇L||={mxa:.6f}, L={Lm:.6f}"
            )

    if gc_on:
        _set_adversarial_grad_checkpointing(model, False)

    if nan_happened:
        print("[Adversarial init] NaN/Inf detected; stopping adversarial init (keeping current parameters).")
        return

    model.eval()
    model.zero_grad(set_to_none=True)
    n_tf, mean_af, max_af, Lf = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
    rel = (n_tf - n_t0) / (n_t0 + 1e-12) * 100.0
    print(
        f"[Adversarial init] Done: ||∇L_target|| {n_t0:.6f} -> {n_tf:.6f} "
        f"(relative {rel:+.2f}%), anchor mean||∇L|| {mean_a0:.6f} -> {mean_af:.6f}, "
        f"max||∇L|| {max_a0:.6f} -> {max_af:.6f}, L {L0:.6f} -> {Lf:.6f}"
    )
    if n_tf < 0.05 * max(n_t0, 1e-8):
        print(
            "[Adversarial init] Warning: final ||∇L_target|| still well below start; try reducing adv_init_lr,"
            "reduce adv_init_clip_grad_norm (tighter clipping), or temporarily set adv_init_w_anchor_max to 0."
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
    if is_seq_cls(cfg):
        dataset = TextClsDataset(texts, labels, tokenizer, cfg.max_length)
    else:
        dataset = CausalLmDataset(texts, tokenizer, cfg.max_length)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=True)


def build_attack_batch(text, label, tokenizer, cfg):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=cfg.max_length,
        return_tensors="pt",
    )
    if is_seq_cls(cfg):
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": torch.tensor([int(label)], dtype=torch.long),
        }
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


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
        "dataset": cfg.dataset,
        "data_root": cfg.data_root,
        "dolly_jsonl": cfg.dolly_jsonl_name,
        "alpaca_json": cfg.alpaca_json_name,
        "gsm8k_parquet_subdir": cfg.gsm8k_parquet_subdir,
        "num_labels": cfg.num_labels,
        "model_name": cfg.model_name,
        "max_length": cfg.max_length,
        "task": "seq_cls" if is_seq_cls(cfg) else "causal_lm",
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
        if not torch.equal(a["input_ids"].cpu(), b["input_ids"].cpu()):
            return False
        if "label" in a or "label" in b:
            return torch.equal(a["label"].cpu(), b["label"].cpu())
        return torch.equal(a["labels"].cpu(), b["labels"].cpu())
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
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        if _is_cuda_oom_error(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Adversarial init cache] OOM loading into model; retrain/skip: {e}")
            return False
        raise
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


def parse_args():
    p = argparse.ArgumentParser(
        description="FedMeZO client-driven Llama-3B MIA (multi-dataset)"
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

    set_seed(cfg.seed)
    task = "SEQ_CLS" if is_seq_cls(cfg) else "CAUSAL_LM"
    print(f"Device: {cfg.device}")
    print(f"Dataset: {cfg.dataset} (task={task})")
    print(f"data_root: {cfg.data_root}")
    print(f"Config: {cfg}")

    print(f"\n[1] Loading {cfg.dataset} data...")
    all_texts, all_labels = load_dataset_texts_labels(cfg)
    need_fl = cfg.fl_samples * cfg.num_clients
    fl_texts = all_texts[:need_fl]
    fl_labels = all_labels[:need_fl]
    attack_texts = all_texts[need_fl : need_fl + cfg.attack_samples]
    attack_labels = all_labels[need_fl : need_fl + cfg.attack_samples]

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

    attack_batches = [
        build_attack_batch(attack_texts[i], attack_labels[i], tokenizer, cfg)
        for i in range(len(attack_texts))
    ]

    cache_meta_45 = _build_step45_meta(cfg, attack_texts, attack_labels)

    print("\n[4] Selecting target sample...")
    target_batch = None
    if os.path.exists(cfg.target_cache_path):
        try:
            target_obj = torch.load(cfg.target_cache_path, map_location="cpu")
            if isinstance(target_obj, dict) and target_obj.get("meta") == cache_meta_45:
                target_batch = target_obj.get("target_batch")
                print(f"Hit target sample cache: {cfg.target_cache_path}")
        except Exception as e:
            print(f"Failed to read target sample cache; will recompute: {e}")
    if target_batch is None:
        target_batch = select_target_sample(model, attack_batches, cfg)
        os.makedirs(os.path.dirname(cfg.target_cache_path) or ".", exist_ok=True)
        torch.save({"meta": cache_meta_45, "target_batch": target_batch}, cfg.target_cache_path)
        print(f"Saved target sample cache: {cfg.target_cache_path}")

    if cfg.adv_init_use:
        loaded = try_load_adv_init_bundle(model, cfg, cache_meta_45, target_batch)
        if not loaded:
            print(f"\n[4.6] Adversarial init (LoRA + {task}, amplify target gradient norm)...")
            src = (cfg.adv_init_anchor_source or "clients").strip().lower()
            if src == "attack_pool":
                anchor_batches = collect_anchor_batches_from_attack_pool(
                    attack_batches, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(f"[Adversarial init] Anchor source: attack pool ({len(anchor_batches)} batches)")
            elif src == "clients":
                anchor_batches = collect_anchor_batches(
                    client_loaders, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(f"[Adversarial init] Anchor source: per-client training data ({len(anchor_batches)} batches)")
            else:
                print(f"Warning: adv_init_anchor_source={cfg.adv_init_anchor_source!r} invalid; using clients.")
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
                    print(f"[Adversarial init] OOM; skipping adversarial init and continuing: {e}")
                else:
                    raise
            if adv_init_ok:
                save_adv_init_bundle(model, cfg, cache_meta_45, target_batch)

    print(
        "\n[5] Round-1 federated simulation: probabilistic member injection + det_score + online/ROC threshold; metrics ASR/TPR/FPR/AUC…"
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
        else:
            test_records = trial_records[calib_end:]
            if not test_records:
                print("[5] Warning: no test trials left after calibration; evaluating on all trials (calibration/test overlap, optimistic metrics).")
                test_records = list(trial_records)
                calib_records = trial_records
            if mode == "fixed":
                print(f"[5] Using fixed threshold: threshold={threshold}")
            elif mode == "roc_youden":
                y_true_calib = np.array([1 if r["member_present"] else 0 for r in calib_records], dtype=np.int32)
                y_score_calib = np.array([r["det_score"] for r in calib_records], dtype=np.float64)
                if len(np.unique(y_true_calib)) < 2:
                    print(f"[5] ROC-Youden calibration failed (calib single class); falling back to fixed threshold={threshold}")
                else:
                    try:
                        from sklearn.metrics import roc_curve
                    except ImportError:
                        print("[5] ROC-Youden calibration failed (sklearn not installed); falling back to fixed threshold")
                    else:
                        fpr_calib, tpr_calib, thr_calib = roc_curve(y_true_calib, y_score_calib)
                        valid = np.isfinite(thr_calib)
                        if not np.any(valid):
                            print("[5] ROC-Youden calibration failed (threshold non-finite); falling back to fixed threshold")
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
            f"ASR={asr:.1%} (member-present trials: correct member prediction and client ID / member-present trial count)"
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
