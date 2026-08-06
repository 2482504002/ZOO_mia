import argparse
import sys
import os
import json
import hashlib
import time
import datetime

# ============= IMPORTANT: enable debug mode to avoid multiprocessing issues =============
os.environ['DEBUG_MODE'] = '1'  # Disable multiprocessing data loading for easier debugging
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # HuggingFace mirror for faster downloads

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FWDLLM_ROOT = os.path.join(_SCRIPT_DIR, "FwdLLM-master")
if os.path.isdir(_FWDLLM_ROOT):
    sys.path.insert(0, _FWDLLM_ROOT)
sys.path.append('./FwdLLM-master')

_REPO_DATASETS = os.path.join(os.path.dirname(_SCRIPT_DIR), "datasets")

DATASET_DEFAULTS = {
    "agnews": {
        "data_file_path": os.path.join(_REPO_DATASETS, "agnews", "data", "agnews_data.h5"),
        "partition_file_path": os.path.join(
            _FWDLLM_ROOT, "fednlp_data", "partition_files", "agnews_partition.h5"
        ),
        "partition_method": "uniform_client_1000",
        "output_dir": os.path.join(
            _FWDLLM_ROOT,
            "experiments",
            "distributed",
            "transformer_exps",
            "run_tc_exps",
            "mia_results",
            "agnews",
        ),
    },
    "alpaca": {
        "data_file_path": os.path.join(_REPO_DATASETS, "alpaca_data.json"),
        "partition_file_path": "",
        "partition_method": "alpaca_default",
        "output_dir": os.path.join(
            _FWDLLM_ROOT,
            "experiments",
            "distributed",
            "transformer_exps",
            "run_tc_exps",
            "mia_results",
            "alpaca",
        ),
    },
    "dolly": {
        "data_file_path": os.path.join(
            _REPO_DATASETS, "dolly15k", "databricks-dolly-15k.jsonl"
        ),
        "partition_file_path": "",
        "partition_method": "dolly15k_default",
        "output_dir": os.path.join(
            _FWDLLM_ROOT,
            "experiments",
            "distributed",
            "transformer_exps",
            "run_tc_exps",
            "mia_results",
            "dolly15k",
        ),
    },
    "gsm8k": {
        "data_file_path": os.path.join(_REPO_DATASETS, "gsm8k"),
        "partition_file_path": "",
        "partition_method": "gsm8k_default",
        "output_dir": os.path.join(
            _FWDLLM_ROOT,
            "experiments",
            "distributed",
            "transformer_exps",
            "run_tc_exps",
            "mia_results",
            "gsm8k",
        ),
    },
}

parser = argparse.ArgumentParser(
    description="Unified DistilBERT zero-order training + MIA script (select dataset via --dataset)"
)

# Basic parameters
parser.add_argument("--run_id", type=int, default=0)
parser.add_argument("--is_debug_mode", type=int, default=0)

# Data-related parameters
parser.add_argument(
    "--dataset",
    type=str,
    default="agnews",
    choices=["agnews", "alpaca", "dolly", "dolly15k", "gsm8k"],
    help="Select dataset: agnews / alpaca / dolly(dolly15k) / gsm8k",
)
parser.add_argument(
    "--data_file_path",
    type=str,
    default=None,
    help="Data path; auto-filled from --dataset by default",
)
parser.add_argument(
    "--partition_file_path",
    type=str,
    default=None,
    help="Partition file path (agnews only); auto-filled from --dataset by default",
)
parser.add_argument(
    "--partition_method",
    type=str,
    default=None,
    help="Partition method; auto-filled from --dataset by default",
)

# Model-related parameters
parser.add_argument("--model_type", type=str, default="distilbert")
parser.add_argument(
    "--model_name",
    type=str,
    default="/home/zhike/JWH/model/distilbert-base-uncased/",
)
parser.add_argument("--do_lower_case", type=bool, default=True)

# Training-related parameters
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--max_seq_length", type=int, default=64)
parser.add_argument("--n_gpu", type=int, default=1)
parser.add_argument("--fp16", default=False, action="store_true")
parser.add_argument("--manual_seed", type=int, default=42)

# I/O-related parameters
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory; defaults to mia_results/<dataset>/ based on --dataset",
)

# Training round parameters (simplified)
parser.add_argument("--fl_algorithm", type=str, default="FedFwd")
parser.add_argument("--backend", type=str, default="MPI")
parser.add_argument("--comm_round", type=int, default=1)
parser.add_argument("--is_mobile", type=int, default=0)
parser.add_argument("--client_num_in_total", type=int, default=-1)
parser.add_argument("--client_num_per_round", type=int, default=100)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument(
    "--use_centralized_data",
    type=bool,
    default=True,
    help="Use centralized data (all data) instead of federated partitions",
)
parser.add_argument(
    "--max_train_samples",
    type=int,
    default=1000,
    help="Limit training samples (for quick debugging)",
)
parser.add_argument(
    "--max_test_samples",
    type=int,
    default=100,
    help="Limit test samples (for quick debugging)",
)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--client_optimizer", type=str, default="adam")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--server_optimizer", type=str, default="sgd")
parser.add_argument("--server_lr", type=float, default=0.1)
parser.add_argument("--server_momentum", type=float, default=0)
parser.add_argument("--fedprox_mu", type=float, default=1)
parser.add_argument("--evaluate_during_training_steps", type=int, default=100)
parser.add_argument("--frequency_of_the_test", type=int, default=1)

# GPU device management
parser.add_argument("--gpu_mapping_file", type=str, default="gpu_mapping.yaml")
parser.add_argument("--gpu_mapping_key", type=str, default="mapping_myMap")
parser.add_argument("--ci", type=int, default=0)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    choices=["cpu", "cuda"],
    help="Device to run on",
)

# Cache-related
parser.add_argument(
    "--reprocess_input_data",
    action="store_true",
    default=False,
    help="Reprocess data and ignore cache (to fix cache issues)",
)

# Freezing-related
parser.add_argument("--freeze_layers", type=str, default="")
parser.add_argument("--use_adapter", type=bool, default=False)

# Forward-mode related
parser.add_argument("--forward_mode", action="store_true", default=True)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--worker_num", type=int, default=1)
parser.add_argument("--peft_method", type=str, default="adapter")
parser.add_argument("--var_control", action="store_true", default=True)
parser.add_argument("--perturbation_sampling", action="store_true", default=True)
parser.add_argument(
    "--enable_mia",
    action="store_true",
    default=True,
    help="Enable membership inference attack",
)
parser.add_argument(
    "--mia_loss_variant",
    type=str,
    default="full",
    choices=["full", "no_square", "no_linear"],
    help="v-optimization loss variant: full=full, no_square=drop square term, no_linear=drop linear term (ablation)",
)
parser.add_argument(
    "--mia_aux_sampling",
    type=str,
    default="random",
    choices=["balanced", "random"],
    help="Auxiliary dataset sampling: balanced=uniform by label, random=random (may miss some labels)",
)
parser.add_argument(
    "--mia_threshold_mode",
    type=str,
    default="fixed",
    choices=["fixed", "online_quantile"],
    help="MIA threshold mode: fixed=fixed threshold, online_quantile=online dynamic threshold at test time",
)
parser.add_argument(
    "--mia_online_warmup",
    type=int,
    default=10,
    help="Warmup samples for online_quantile (warmup accumulates history only, no classification)",
)
parser.add_argument(
    "--mia_online_window",
    type=int,
    default=40,
    help="History window size for quantile threshold estimation in online_quantile mode",
)
parser.add_argument(
    "--mia_online_alpha",
    type=float,
    default=0.1,
    help="Quantile parameter for online_quantile; threshold=quantile(history, 1-alpha)",
)
parser.add_argument(
    "--mia_online_min_threshold",
    type=float,
    default=0.0,
    help="Lower bound for online_quantile threshold",
)
parser.add_argument(
    "--mia_online_max_threshold",
    type=float,
    default=1e12,
    help="Upper bound for online_quantile threshold",
)
parser.add_argument(
    "--mia_online_update_with_neg_only",
    type=int,
    default=1,
    help="online_quantile history update: 1=update only when predicted non-member, 0=update on all samples",
)
parser.add_argument(
    "--mia_aux_samples_fixed",
    type=int,
    default=300,
    help="Fixed MIA auxiliary sample count (sampled from full_train_dl, excluding client train_dl)",
)
parser.add_argument(
    "--mia_aux_seed",
    type=int,
    default=12345,
    help="Random seed for MIA auxiliary/target sampling (decoupled from manual_seed)",
)

# Pseudo-label parameters per dataset
parser.add_argument(
    "--pseudo_num_labels",
    type=int,
    default=4,
    help="Number of pseudo-label classes for alpaca/dolly/gsm8k",
)
parser.add_argument(
    "--dolly_jsonl_name",
    type=str,
    default="databricks-dolly-15k.jsonl",
    help="Dolly JSONL filename when --data_file_path is a directory",
)
parser.add_argument(
    "--gsm8k_parquet_subdir",
    type=str,
    default="main",
    help="Subdirectory under gsm8k data_file_path containing train-*.parquet / test-*.parquet",
)
parser.add_argument(
    "--gsm8k_merge_test",
    type=int,
    default=1,
    help="Merge test-*.parquet into train/attack pool (1=yes, 0=no)",
)

args = parser.parse_args()

# Normalize dataset name and fill default paths
if args.dataset.lower() == "dolly15k":
    args.dataset = "dolly"
else:
    args.dataset = args.dataset.lower()

_ds_defaults = DATASET_DEFAULTS[args.dataset]
if args.data_file_path is None:
    args.data_file_path = _ds_defaults["data_file_path"]
if args.partition_file_path is None:
    args.partition_file_path = _ds_defaults["partition_file_path"]
if args.partition_method is None:
    args.partition_method = _ds_defaults["partition_method"]
if args.output_dir is None:
    args.output_dir = _ds_defaults["output_dir"]

import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import random
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from data_preprocessing.text_classification_preprocessor import TLMPreprocessor
from forward_training.tc_transformer_trainer_distribute import ForwardTextClassificationTrainer
from model.transformer.model_args import ClassificationArgs
from data_manager.text_classification_data_manager import TextClassificationDataManager
from data_manager.base_data_manager import BaseDataManager
from data_preprocessing.base.base_data_loader import BaseDataLoader
from transformers import (
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    AlbertConfig,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    DebertaConfig,
    DebertaForSequenceClassification,
    DebertaTokenizer,
)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _mia_stats_to_auc(stats):
    if not stats or stats.get("member_total", 0) == 0:
        roc = stats.get("roc_auc") if stats else None
        if roc is not None and np.isfinite(roc):
            return float(roc)
        return float("nan")
    roc = stats.get("roc_auc")
    if roc is not None and np.isfinite(roc):
        return float(roc)
    y_true = np.array(stats.get("true_labels", []), dtype=np.int32)
    y_score = np.array(stats.get("jvp_values", []), dtype=np.float64)
    valid = np.isfinite(y_score)
    if not valid.any():
        return float("nan")
    y_true = y_true[valid]
    y_score = y_score[valid]
    if len(y_true) <= 1 or len(np.unique(y_true)) <= 1:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# -------------------- Generic text dataset construction --------------------
class TextTupleDataset(Dataset):
    """Returns 5-tuple compatible with existing trainer: (idx, input_ids, attention_mask, token_type_ids, label)"""

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = torch.zeros_like(input_ids)
        self.labels = labels

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return (
            torch.tensor(idx, dtype=torch.long),
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.labels[idx],
        )


def _build_text_dataloader(texts, labels, tokenizer, max_seq_length, batch_size, shuffle):
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )
    dataset = TextTupleDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(labels, dtype=torch.long),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    dataloader.examples = list(range(len(dataset)))
    dataloader.features = dataloader.examples
    return dataloader


def _split_and_build_loaders(
    all_texts,
    all_labels,
    tokenizer,
    max_seq_length,
    train_batch_size,
    eval_batch_size,
    max_train_samples,
    max_test_samples,
    shuffle_seed,
    dataset_name,
):
    if len(all_texts) < 2:
        raise ValueError(f"{dataset_name} too few samples: {len(all_texts)}; need at least 2.")

    indices = list(range(len(all_texts)))
    rng = random.Random(shuffle_seed)
    rng.shuffle(indices)
    all_texts = [all_texts[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    split_idx = max(1, int(len(all_texts) * 0.9))
    if split_idx >= len(all_texts):
        split_idx = len(all_texts) - 1

    train_texts_full = all_texts[:split_idx]
    train_labels_full = all_labels[:split_idx]
    test_texts = all_texts[split_idx:]
    test_labels = all_labels[split_idx:]

    if max_train_samples is not None:
        train_texts = train_texts_full[: min(max_train_samples, len(train_texts_full))]
        train_labels = train_labels_full[: min(max_train_samples, len(train_labels_full))]
    else:
        train_texts = train_texts_full
        train_labels = train_labels_full

    if max_test_samples is not None:
        test_texts = test_texts[: min(max_test_samples, len(test_texts))]
        test_labels = test_labels[: min(max_test_samples, len(test_labels))]

    if len(train_texts) == 0 or len(test_texts) == 0:
        raise ValueError(
            f"Empty train/test split: train={len(train_texts)}, test={len(test_texts)}."
            "Adjust max_train_samples/max_test_samples or the data split."
        )

    full_train_dl = _build_text_dataloader(
        train_texts_full, train_labels_full, tokenizer, max_seq_length, train_batch_size, True
    )
    train_dl = _build_text_dataloader(
        train_texts, train_labels, tokenizer, max_seq_length, train_batch_size, True
    )
    test_dl = _build_text_dataloader(
        test_texts, test_labels, tokenizer, max_seq_length, eval_batch_size, False
    )
    logger.info(
        f"{dataset_name} loaded: total={len(all_texts)}, train_full={len(train_texts_full)}, "
        f"train={len(train_texts)}, test={len(test_texts)}"
    )
    return train_dl, test_dl, full_train_dl


# -------------------- Alpaca --------------------
def _alpaca_record_to_text(record):
    instruction = (record.get("instruction") or "").strip()
    input_text = (record.get("input") or "").strip()
    output = (record.get("output") or "").strip()
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    )


def _alpaca_pseudo_label(record, num_labels):
    key = f"{record.get('instruction', '')}\n{record.get('output', '')}".encode("utf-8")
    hashed = int(hashlib.md5(key).hexdigest()[:12], 16)
    return hashed % num_labels


def _load_alpaca_texts_labels(json_path, num_labels):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("alpaca_data.json must be a JSON array list[dict]")
    texts = [_alpaca_record_to_text(item) for item in raw]
    labels = [_alpaca_pseudo_label(item, num_labels) for item in raw]
    return texts, labels


# -------------------- Dolly --------------------
def _dolly_record_to_text(record):
    instruction = (record.get("instruction") or "").strip()
    context = (record.get("context") or "").strip()
    response = (record.get("response") or "").strip()
    if context:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{response}"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    )


def _dolly_pseudo_label(record, num_labels):
    key = (
        f"{record.get('instruction', '')}\n{record.get('context', '')}\n{record.get('response', '')}"
    ).encode("utf-8")
    hashed = int(hashlib.md5(key).hexdigest()[:12], 16)
    return hashed % num_labels


def _read_dolly_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_dolly_texts_labels(data_path, jsonl_name, num_labels):
    if os.path.isfile(data_path):
        jsonl_path = data_path
    else:
        jsonl_path = os.path.join(data_path, jsonl_name)
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(
            f"Dolly15k data not found: {jsonl_path}\n"
            f"Ensure the path exists, or pass a directory and set --dolly_jsonl_name."
        )
    raw = _read_dolly_jsonl(jsonl_path)
    texts = [_dolly_record_to_text(item) for item in raw]
    labels = [_dolly_pseudo_label(item, num_labels) for item in raw]
    return texts, labels


# -------------------- GSM8K --------------------
def _gsm8k_record_to_text(question, answer):
    question = (question or "").strip()
    answer = (answer or "").strip()
    return f"Question: {question}\nAnswer: {answer}"


def _gsm8k_pseudo_label(question, answer, num_labels):
    key = f"{question}\n{answer}".encode("utf-8")
    hashed = int(hashlib.md5(key).hexdigest()[:12], 16)
    return hashed % num_labels


def _read_gsm8k_parquet_paths(paths):
    """Prefer pyarrow, fall back to pandas; read question/answer columns from parquet."""
    if not paths:
        return []

    def _with_pyarrow():
        import pyarrow.parquet as pq

        rows = []
        for path in paths:
            table = pq.read_table(path, columns=["question", "answer"])
            rows.extend(table.to_pylist())
        return [{"question": row["question"], "answer": row["answer"]} for row in rows]

    def _with_pandas():
        import pandas as pd

        frames = [pd.read_parquet(path, columns=["question", "answer"]) for path in paths]
        df = pd.concat(frames, ignore_index=True)
        return df.to_dict("records")

    try:
        return _with_pyarrow()
    except ImportError:
        pass
    except Exception as pyarrow_error:
        try:
            return _with_pandas()
        except Exception:
            raise pyarrow_error

    try:
        return _with_pandas()
    except ImportError as err:
        raise ImportError(
            "Reading GSM8K parquet requires pyarrow or pandas (suggested: pip install pyarrow)"
        ) from err


def _load_gsm8k_texts_labels(data_root, parquet_subdir, merge_test, num_labels):
    import glob

    parquet_dir = os.path.join(data_root, parquet_subdir)
    train_files = sorted(glob.glob(os.path.join(parquet_dir, "train-*.parquet")))
    if (not train_files) and os.path.abspath(parquet_dir) != os.path.abspath(data_root):
        parquet_dir = data_root
        train_files = sorted(glob.glob(os.path.join(parquet_dir, "train-*.parquet")))
    if not train_files:
        raise FileNotFoundError(
            f"GSM8K parquet not found: {parquet_dir}/train-*.parquet\n"
            f"Ensure {data_root} contains subdirectory {parquet_subdir} with train-*.parquet."
        )
    parquet_paths = list(train_files)
    if bool(merge_test):
        parquet_paths.extend(sorted(glob.glob(os.path.join(parquet_dir, "test-*.parquet"))))

    raw = _read_gsm8k_parquet_paths(parquet_paths)
    texts = [
        _gsm8k_record_to_text(item.get("question", ""), item.get("answer", "")) for item in raw
    ]
    labels = [
        _gsm8k_pseudo_label(item.get("question", ""), item.get("answer", ""), num_labels)
        for item in raw
    ]
    return texts, labels


def create_model(model_args, formulation="classification"):
    """Create model, tokenizer, and config"""
    MODEL_CLASSES = {
        "classification": {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
            "roberta-large": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            "deberta": (DebertaConfig, DebertaForSequenceClassification, DebertaTokenizer),
        },
    }

    config_class, model_class, tokenizer_class = MODEL_CLASSES[formulation][model_args.model_type]

    logger.info(f"Loading model: {model_args.model_name}")
    config = config_class.from_pretrained(model_args.model_name, **model_args.config)
    model = model_class.from_pretrained(
        model_args.model_name, config=config, ignore_mismatched_sizes=True
    )
    tokenizer = tokenizer_class.from_pretrained(
        model_args.model_name, do_lower_case=model_args.do_lower_case
    )

    logger.info(f"Model before PEFT: {model.__class__.__name__}")
    total_params_before = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters before PEFT: {total_params_before:,}")

    if model_args.peft_method == "adapter":
        if hasattr(model, "add_adapter") and hasattr(model, "train_adapter"):
            logger.info("Applying Adapter method...")
            adapter_config = {
                "original_ln_before": True,
                "original_ln_after": True,
                "residual_before_ln": True,
                "adapter_residual_before_ln": False,
                "ln_before": False,
                "ln_after": False,
                "mh_adapter": False,
                "output_adapter": True,
                "non_linearity": "relu",
                "reduction_factor": 16,
                "inv_adapter": None,
                "inv_adapter_reduction_factor": None,
                "cross_adapter": False,
                "leave_out": [],
            }
            model.add_adapter("zero_order_adapter", adapter_config)
            model.train_adapter("zero_order_adapter")
            logger.info("Adapter added and activated")
        else:
            logger.warning(
                "peft_method=adapter but %s has no add_adapter/train_adapter (common for standard transformers); "
                "skipping Adapter; training as a plain classification model.",
                model.__class__.__name__,
            )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    return config, model, tokenizer


def _load_agnews_dataloaders(cli_args, model_args, preprocessor, data_shuffle_seed):
    dm = TextClassificationDataManager(
        cli_args,
        model_args,
        preprocessor,
        process_id=0,
        num_workers=cli_args.client_num_per_round,
    )

    if cli_args.use_centralized_data:
        logger.info("Using centralized data (all data combined)...")
        train_dl, test_dl = dm.load_centralized_data(cut_off=None)
        full_train_dl = train_dl
    else:
        logger.info("Using federated data loading (server mode)...")
        _, _, test_dl, _, _, _, _ = dm.load_federated_data(process_id=0)
        dm_client = TextClassificationDataManager(
            cli_args,
            model_args,
            preprocessor,
            process_id=1,
            num_workers=cli_args.client_num_per_round,
        )
        _, _, _, _, train_data_local_dict, _, _ = dm_client.load_federated_data(process_id=1)
        train_dl = train_data_local_dict[0]
        logger.info("Loading full dataset for MIA attack...")
        full_train_dl, _ = dm.load_centralized_data(cut_off=None)
        logger.info(f"Full dataset size: {len(full_train_dl.dataset)} samples")

    if hasattr(train_dl, "dataset") and train_dl.dataset is not None:
        dataset_size = len(train_dl.dataset)
        sampler = RandomSampler(
            train_dl.dataset, generator=torch.Generator().manual_seed(data_shuffle_seed)
        )
        train_dl = BaseDataLoader(
            train_dl.examples,
            train_dl.features,
            train_dl.dataset,
            batch_size=model_args.train_batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        logger.info(f"Training dataset shuffled (size: {dataset_size})")

    if hasattr(full_train_dl, "dataset") and full_train_dl.dataset is not None:
        full_dataset_size = len(full_train_dl.dataset)
        full_sampler = RandomSampler(
            full_train_dl.dataset, generator=torch.Generator().manual_seed(data_shuffle_seed)
        )
        full_train_dl = BaseDataLoader(
            full_train_dl.examples,
            full_train_dl.features,
            full_train_dl.dataset,
            batch_size=model_args.train_batch_size,
            sampler=full_sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        logger.info(f"Full dataset shuffled (size: {full_dataset_size})")

    if cli_args.max_train_samples is not None:
        logger.info(f"Limiting training samples to {cli_args.max_train_samples}...")
        original_train_size = len(train_dl.dataset)
        train_indices = list(range(min(cli_args.max_train_samples, len(train_dl.dataset))))
        train_subset = Subset(train_dl.dataset, train_indices)
        train_dl = BaseDataLoader(
            train_dl.examples[: cli_args.max_train_samples],
            train_dl.features[: cli_args.max_train_samples],
            train_subset,
            batch_size=model_args.train_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        logger.info(f"Training data reduced: {original_train_size} → {len(train_dl.dataset)}")

    if cli_args.max_test_samples is not None:
        logger.info(f"Limiting test samples to {cli_args.max_test_samples}...")
        original_test_size = len(test_dl.dataset)
        test_indices = list(range(min(cli_args.max_test_samples, len(test_dl.dataset))))
        test_subset = Subset(test_dl.dataset, test_indices)
        test_dl = BaseDataLoader(
            test_dl.examples[: cli_args.max_test_samples],
            test_dl.features[: cli_args.max_test_samples],
            test_subset,
            batch_size=model_args.eval_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        logger.info(f"Test data reduced: {original_test_size} → {len(test_dl.dataset)}")

    return train_dl, test_dl, full_train_dl


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info(f"Starting zero-order optimization training | dataset={args.dataset}")
    logger.info("=" * 80)

    set_seed(args.manual_seed)
    logger.info(f"Random seed: {args.manual_seed}")

    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            logger.warning("Requested device=cuda but CUDA is unavailable, fallback to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    use_custom_text = args.dataset in {"alpaca", "dolly", "gsm8k"}
    logger.info(f"Loading dataset from: {args.data_file_path}")

    if use_custom_text:
        num_labels = int(args.pseudo_num_labels)
        attributes = {"label_vocab": list(range(num_labels))}
        logger.info(f"Dataset: {args.dataset} (custom text mode), Labels: {num_labels}")
        logger.info("Label vocabulary: pseudo labels via hash(...) mod num_labels")
    else:
        attributes = BaseDataManager.load_attributes(args.data_file_path)
        num_labels = len(attributes["label_vocab"])
        logger.info(f"Dataset: {args.dataset}, Labels: {num_labels}")
        logger.info(f"Label vocabulary: {attributes['label_vocab']}")

    logger.info("Creating model configuration...")
    model_args = ClassificationArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.update_from_dict(
        {
            "fl_algorithm": args.fl_algorithm,
            "freeze_layers": args.freeze_layers,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "do_lower_case": args.do_lower_case,
            "manual_seed": args.manual_seed,
            "reprocess_input_data": args.reprocess_input_data,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "evaluate_during_training": False,
            "evaluate_during_training_steps": args.evaluate_during_training_steps,
            "fp16": args.fp16,
            "data_file_path": args.data_file_path,
            "partition_file_path": args.partition_file_path,
            "partition_method": args.partition_method,
            "dataset": args.dataset if args.dataset != "dolly" else "dolly15k",
            "output_dir": args.output_dir,
            "is_debug_mode": args.is_debug_mode,
            "fedprox_mu": args.fedprox_mu,
            "use_adapter": args.use_adapter,
            "comm_round": args.comm_round,
            "peft_method": args.peft_method,
            "var_control": args.var_control,
            "perturbation_sampling": args.perturbation_sampling,
            "enable_mia": args.enable_mia,
            "mia_loss_variant": args.mia_loss_variant,
            "mia_aux_sampling": args.mia_aux_sampling,
            "mia_aux_samples_fixed": args.mia_aux_samples_fixed,
            "mia_threshold_mode": args.mia_threshold_mode,
            "mia_online_warmup": args.mia_online_warmup,
            "mia_online_window": args.mia_online_window,
            "mia_online_alpha": args.mia_online_alpha,
            "mia_online_min_threshold": args.mia_online_min_threshold,
            "mia_online_max_threshold": args.mia_online_max_threshold,
            "mia_online_update_with_neg_only": bool(args.mia_online_update_with_neg_only),
        }
    )
    model_args.config["num_labels"] = num_labels

    logger.info("=" * 80)
    logger.info("Creating model...")
    logger.info("=" * 80)
    model_config, model, tokenizer = create_model(model_args, formulation="classification")
    print(model)

    logger.info("=" * 80)
    logger.info("Creating Forward (Zero-Order) Trainer...")
    logger.info("=" * 80)
    trainer = ForwardTextClassificationTrainer(model_args, device, model, None, None)

    preprocessor = None
    if not use_custom_text:
        logger.info("Creating data preprocessor...")
        preprocessor = TLMPreprocessor(
            args=model_args,
            label_vocab=attributes["label_vocab"],
            tokenizer=tokenizer,
        )

    logger.info("=" * 80)
    logger.info("Loading data (centralized mode)...")
    logger.info("Note: DEBUG_MODE=1, using single-process data loading")
    logger.info("=" * 80)

    data_shuffle_seed = int(time.time() * 1000000) % (2**31)
    random.seed(data_shuffle_seed)
    np.random.seed(data_shuffle_seed)
    logger.info(f"Data loading shuffle seed: {data_shuffle_seed} (different order each run)")

    if args.dataset == "alpaca":
        logger.info("Using Alpaca JSON loader")
        all_texts, all_labels = _load_alpaca_texts_labels(args.data_file_path, num_labels)
        train_dl, test_dl, full_train_dl = _split_and_build_loaders(
            all_texts,
            all_labels,
            tokenizer,
            args.max_seq_length,
            model_args.train_batch_size,
            model_args.eval_batch_size,
            args.max_train_samples,
            args.max_test_samples,
            data_shuffle_seed,
            "Alpaca",
        )
    elif args.dataset == "dolly":
        logger.info("Using Dolly15k JSONL loader")
        all_texts, all_labels = _load_dolly_texts_labels(
            args.data_file_path, args.dolly_jsonl_name, num_labels
        )
        train_dl, test_dl, full_train_dl = _split_and_build_loaders(
            all_texts,
            all_labels,
            tokenizer,
            args.max_seq_length,
            model_args.train_batch_size,
            model_args.eval_batch_size,
            args.max_train_samples,
            args.max_test_samples,
            data_shuffle_seed,
            "Dolly15k",
        )
    elif args.dataset == "gsm8k":
        logger.info("Using GSM8K parquet loader")
        all_texts, all_labels = _load_gsm8k_texts_labels(
            args.data_file_path,
            args.gsm8k_parquet_subdir,
            bool(args.gsm8k_merge_test),
            num_labels,
        )
        train_dl, test_dl, full_train_dl = _split_and_build_loaders(
            all_texts,
            all_labels,
            tokenizer,
            args.max_seq_length,
            model_args.train_batch_size,
            model_args.eval_batch_size,
            args.max_train_samples,
            args.max_test_samples,
            data_shuffle_seed,
            "GSM8K",
        )
    else:
        train_dl, test_dl, full_train_dl = _load_agnews_dataloaders(
            args, model_args, preprocessor, data_shuffle_seed
        )

    trainer.train_dl = train_dl
    trainer.test_dl = test_dl
    trainer.set_data(train_dl, test_dl, full_train_dl=full_train_dl)

    logger.info(f"Training batches: {len(train_dl)}")
    logger.info(f"Testing batches: {len(test_dl)}")
    logger.info(f"Training samples: {len(train_dl.dataset)}")
    logger.info(f"Testing samples: {len(test_dl.dataset)}")
    logger.info(f"Full dataset samples: {len(full_train_dl.dataset)} (for MIA attack)")
    logger.info(
        f"Fixed auxiliary set size={args.mia_aux_samples_fixed}, mia_aux_seed={args.mia_aux_seed}"
    )
    random.seed(args.mia_aux_seed)
    trainer.args.mia_aux_samples_fixed = args.mia_aux_samples_fixed
    trainer.args.mia_aux_external_pool = None

    logger.info(f"Starting training (aux_size={args.mia_aux_samples_fixed})...")
    logger.info(f"Training rounds: {args.comm_round}, epochs/round: {args.epochs}")

    all_round_mia_stats = {
        "rounds": [],
        "member_accuracy": [],
        "non_member_accuracy": [],
        "overall_accuracy": [],
        "tpr": [],
        "fpr": [],
        "auc": [],
        "member_correct": [],
        "member_total": [],
        "non_member_correct": [],
        "non_member_total": [],
    }

    for round_idx in range(args.comm_round):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Round {round_idx + 1}/{args.comm_round}")
        logger.info("=" * 80)

        logger.info(
            f"📊 [Before train_model] old_grad = "
            f"{'None' if trainer.old_grad is None else f'List[{len(trainer.old_grad)} tensors]'}"
        )

        global_step, tr_loss = trainer.train_model(device)
        logger.info(f"Training completed: global_step={global_step}, loss={tr_loss:.6f}")

        if hasattr(trainer, "mia_stats") and trainer.mia_stats.get("member_total", 0) > 0:
            stats = trainer.mia_stats
            member_acc = (
                stats["member_correct"] / stats["member_total"] if stats["member_total"] > 0 else 0
            )
            non_member_acc = (
                stats["non_member_correct"] / stats["non_member_total"]
                if stats["non_member_total"] > 0
                else 0
            )
            total_eval = stats["member_total"] + stats["non_member_total"]
            overall_acc = (
                (stats["member_correct"] + stats["non_member_correct"]) / total_eval
                if total_eval > 0
                else 0
            )
            tpr = stats.get("tpr", member_acc)
            fpr = stats.get("fpr", 1.0 - non_member_acc)
            round_auc = _mia_stats_to_auc(stats)

            all_round_mia_stats["rounds"].append(round_idx + 1)
            all_round_mia_stats["member_accuracy"].append(member_acc)
            all_round_mia_stats["non_member_accuracy"].append(non_member_acc)
            all_round_mia_stats["overall_accuracy"].append(overall_acc)
            all_round_mia_stats["tpr"].append(tpr)
            all_round_mia_stats["fpr"].append(fpr)
            all_round_mia_stats["auc"].append(round_auc)
            all_round_mia_stats["member_correct"].append(stats["member_correct"])
            all_round_mia_stats["member_total"].append(stats["member_total"])
            all_round_mia_stats["non_member_correct"].append(stats["non_member_correct"])
            all_round_mia_stats["non_member_total"].append(stats["non_member_total"])

            logger.info("")
            logger.info(f"📊 [Round {round_idx + 1} MIA Stats]")
            logger.info(
                f"  Member Accuracy: {member_acc:.2%} ({stats['member_correct']}/{stats['member_total']})"
            )
            logger.info(
                f"  Non-Member Accuracy: {non_member_acc:.2%} "
                f"({stats['non_member_correct']}/{stats['non_member_total']})"
            )
            logger.info(f"  Overall Accuracy: {overall_acc:.2%}")
            logger.info(f"  TPR: {tpr:.2%}, FPR: {fpr:.2%}")
            logger.info(f"  AUC: {round_auc:.4f}" if np.isfinite(round_auc) else "  AUC: N/A")

        if hasattr(trainer, "apply_gradient_update"):
            trainer.apply_gradient_update(current_round=round_idx)
            logger.info("Applied gradient update")
        else:
            logger.error("❌ apply_gradient_update method NOT found!")

        if (round_idx + 1) % args.frequency_of_the_test == 0:
            logger.info("Evaluating model...")
            metrics, _, _ = trainer.eval_model()
            logger.info(f"Evaluation metrics: {metrics}")
            logger.info(f"Accuracy: {metrics.get('acc', 0):.4f}")

    final_auc = _mia_stats_to_auc(getattr(trainer, "mia_stats", None))
    logger.info(
        f"MIA finished (dataset={args.dataset}, aux_size={args.mia_aux_samples_fixed}), AUC={final_auc:.4f}"
        if np.isfinite(final_auc)
        else f"MIA finished (dataset={args.dataset}, aux_size={args.mia_aux_samples_fixed}), AUC=N/A"
    )

    if len(all_round_mia_stats["rounds"]) > 0:
        logger.info("")
        logger.info("=" * 100)
        logger.info("[MIA - Cross-Round Summary Report]")
        logger.info("=" * 100)

        total_member_correct = sum(all_round_mia_stats["member_correct"])
        total_member_total = sum(all_round_mia_stats["member_total"])
        total_non_member_correct = sum(all_round_mia_stats["non_member_correct"])
        total_non_member_total = sum(all_round_mia_stats["non_member_total"])

        final_member_acc = (
            total_member_correct / total_member_total if total_member_total > 0 else 0
        )
        final_non_member_acc = (
            total_non_member_correct / total_non_member_total if total_non_member_total > 0 else 0
        )
        final_overall_acc = (
            (total_member_correct + total_non_member_correct)
            / (total_member_total + total_non_member_total)
            if (total_member_total + total_non_member_total) > 0
            else 0
        )
        final_tpr = final_member_acc
        final_fpr = 1.0 - final_non_member_acc
        finite_auc = [x for x in all_round_mia_stats["auc"] if np.isfinite(x)]
        final_auc = float(np.mean(finite_auc)) if len(finite_auc) > 0 else float("nan")

        logger.info("\n[Overall Attack Performance (All Rounds)]")
        logger.info(f"  Dataset: {args.dataset}")
        logger.info(f"  Test rounds: {len(all_round_mia_stats['rounds'])}")
        logger.info("  Member samples (Ground Truth: IN batch):")
        logger.info(f"    - Total tested: {total_member_total}")
        logger.info(f"    - Correctly identified: {total_member_correct}")
        logger.info(f"    - Accuracy: {final_member_acc:.2%}")
        logger.info("  Non-member samples (Ground Truth: NOT in batch):")
        logger.info(f"    - Total tested: {total_non_member_total}")
        logger.info(f"    - Correctly identified: {total_non_member_correct}")
        logger.info(f"    - Accuracy: {final_non_member_acc:.2%}")
        logger.info("  Overall performance:")
        logger.info(f"    - Total test samples: {total_member_total + total_non_member_total}")
        logger.info(f"    - Total correct: {total_member_correct + total_non_member_correct}")
        logger.info(f"    - Overall accuracy: {final_overall_acc:.2%}")
        logger.info(f"    - TPR: {final_tpr:.2%}")
        logger.info(f"    - FPR: {final_fpr:.2%}")
        logger.info(
            f"    - AUC (mean): {final_auc:.4f}"
            if np.isfinite(final_auc)
            else "    - AUC (mean): N/A"
        )

        logger.info("\n[Per-Round Attack Accuracy]")
        logger.info("  Round | Member Acc | Non-Member Acc | Overall Acc | TPR | FPR | AUC")
        logger.info(f"  {'-' * 100}")
        for i, round_num in enumerate(all_round_mia_stats["rounds"]):
            mem_acc = all_round_mia_stats["member_accuracy"][i]
            non_mem_acc = all_round_mia_stats["non_member_accuracy"][i]
            ovr_acc = all_round_mia_stats["overall_accuracy"][i]
            tpr_i = all_round_mia_stats["tpr"][i]
            fpr_i = all_round_mia_stats["fpr"][i]
            auc_i = all_round_mia_stats["auc"][i]
            auc_str = f"{auc_i:.4f}" if np.isfinite(auc_i) else "N/A"
            logger.info(
                f"  {round_num:4d} |    {mem_acc:6.2%}   |     {non_mem_acc:6.2%}    |    "
                f"{ovr_acc:6.2%} | {tpr_i:6.2%} | {fpr_i:6.2%} | {auc_str}"
            )

        report_file = os.path.join(
            args.output_dir,
            f"mia_attack_report_{args.dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        os.makedirs(args.output_dir, exist_ok=True)

        report_data = {
            "attack_config": {
                "dataset": args.dataset,
                "data_file_path": args.data_file_path,
                "batch_size": args.train_batch_size,
                "method": "Influence Function Analysis with Adversarial v Optimization",
                "total_rounds": len(all_round_mia_stats["rounds"]),
                "total_batches_tested": total_member_total,
            },
            "final_results": {
                "asr": final_overall_acc,
                "tpr": final_tpr,
                "fpr": final_fpr,
                "auc": final_auc if np.isfinite(final_auc) else None,
                "member_accuracy": final_member_acc,
                "non_member_accuracy": final_non_member_acc,
                "overall_accuracy": final_overall_acc,
                "member_correct": int(total_member_correct),
                "member_total": int(total_member_total),
                "non_member_correct": int(total_non_member_correct),
                "non_member_total": int(total_non_member_total),
            },
            "per_round_results": [
                {
                    "round": int(r),
                    "asr": float(all_round_mia_stats["overall_accuracy"][i]),
                    "tpr": float(all_round_mia_stats["tpr"][i]),
                    "fpr": float(all_round_mia_stats["fpr"][i]),
                    "auc": (
                        float(all_round_mia_stats["auc"][i])
                        if np.isfinite(all_round_mia_stats["auc"][i])
                        else None
                    ),
                    "member_accuracy": float(all_round_mia_stats["member_accuracy"][i]),
                    "non_member_accuracy": float(all_round_mia_stats["non_member_accuracy"][i]),
                    "overall_accuracy": float(all_round_mia_stats["overall_accuracy"][i]),
                    "member_correct": int(all_round_mia_stats["member_correct"][i]),
                    "member_total": int(all_round_mia_stats["member_total"][i]),
                    "non_member_correct": int(all_round_mia_stats["non_member_correct"][i]),
                    "non_member_total": int(all_round_mia_stats["non_member_total"][i]),
                }
                for i, r in enumerate(all_round_mia_stats["rounds"])
            ],
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"\nAttack report saved to: {report_file}")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            try:
                from matplotlib import font_manager

                chinese_fonts = ["AR PL UMing CN", "Noto Sans CJK SC", "SimHei", "Microsoft YaHei"]
                available_fonts = {f.name for f in font_manager.fontManager.ttflist}
                font_found = None
                for font_name in chinese_fonts:
                    if font_name in available_fonts:
                        font_found = font_name
                        break
                if font_found:
                    plt.rcParams["font.sans-serif"] = [font_found] + plt.rcParams["font.sans-serif"]
            except Exception:
                pass
            plt.rcParams["axes.unicode_minus"] = False

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            rounds = all_round_mia_stats["rounds"]

            axes[0, 0].plot(
                rounds,
                [x * 100 for x in all_round_mia_stats["member_accuracy"]],
                marker="o",
                label="Member accuracy",
                linewidth=2,
            )
            axes[0, 0].set_xlabel("Training round")
            axes[0, 0].set_ylabel("Accuracy (%)")
            axes[0, 0].set_title(f"Member Sample Identification Accuracy ({args.dataset})")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(
                rounds,
                [x * 100 for x in all_round_mia_stats["non_member_accuracy"]],
                marker="s",
                color="orange",
                label="Non-member accuracy",
                linewidth=2,
            )
            axes[0, 1].set_xlabel("Training round")
            axes[0, 1].set_ylabel("Accuracy (%)")
            axes[0, 1].set_title(f"Non-Member Sample Identification Accuracy ({args.dataset})")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(
                rounds,
                [x * 100 for x in all_round_mia_stats["overall_accuracy"]],
                marker="^",
                color="green",
                label="Overall accuracy",
                linewidth=2,
            )
            axes[1, 0].set_xlabel("Training round")
            axes[1, 0].set_ylabel("Accuracy (%)")
            axes[1, 0].set_title(f"Overall Attack Accuracy ({args.dataset})")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(
                rounds,
                [x * 100 for x in all_round_mia_stats["member_accuracy"]],
                marker="o",
                label="Member",
                linewidth=2,
            )
            axes[1, 1].plot(
                rounds,
                [x * 100 for x in all_round_mia_stats["non_member_accuracy"]],
                marker="s",
                label="Non-member",
                linewidth=2,
            )
            axes[1, 1].plot(
                rounds,
                [x * 100 for x in all_round_mia_stats["overall_accuracy"]],
                marker="^",
                label="Overall",
                linewidth=2,
            )
            axes[1, 1].set_xlabel("Training round")
            axes[1, 1].set_ylabel("Accuracy (%)")
            axes[1, 1].set_title(f"Attack Accuracy Comparison ({args.dataset})")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = os.path.join(
                args.output_dir,
                f"mia_attack_results_{args.dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Attack visualization saved to: {plot_file}")
        except Exception as e:
            logger.warning(f"Failed to generate visualization: {e}")

    logger.info("=" * 100)
