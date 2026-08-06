import argparse
import sys
import os
import json
import hashlib
# ============= IMPORTANT: set debug mode to avoid multiprocessing issues =============
os.environ['DEBUG_MODE'] = '1'  # disable multiprocess data loading for easier debugging
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # HuggingFace mirror for faster downloads

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FWDLLM_ROOT = os.path.join(_SCRIPT_DIR, "FwdLLM-master")
if os.path.isdir(_FWDLLM_ROOT):
    sys.path.insert(0, _FWDLLM_ROOT)
sys.path.append('/home/zhike/JWH/FwdLLM-master')
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
            "experiments", "distributed", "transformer_exps", "run_tc_exps",
            "mia_results", "agnews",
        ),
        "mia_aux_samples_fixed": 0,
    },
    "alpaca": {
        "data_file_path": os.path.join(_REPO_DATASETS, "alpaca_data.json"),
        "partition_file_path": "",
        "partition_method": "alpaca_default",
        "output_dir": os.path.join(
            _FWDLLM_ROOT,
            "experiments", "distributed", "transformer_exps", "run_tc_exps",
            "mia_results", "alpaca",
        ),
        "mia_aux_samples_fixed": 100,
    },
    "dolly": {
        "data_file_path": os.path.join(
            _REPO_DATASETS, "dolly15k", "databricks-dolly-15k.jsonl"
        ),
        "partition_file_path": "",
        "partition_method": "dolly15k_default",
        "output_dir": os.path.join(
            _FWDLLM_ROOT,
            "experiments", "distributed", "transformer_exps", "run_tc_exps",
            "mia_results", "dolly15k",
        ),
        "mia_aux_samples_fixed": 100,
    },
    "gsm8k": {
        "data_file_path": os.path.join(_REPO_DATASETS, "gsm8k"),
        "partition_file_path": "",
        "partition_method": "gsm8k_default",
        "output_dir": os.path.join(
            _FWDLLM_ROOT,
            "experiments", "distributed", "transformer_exps", "run_tc_exps",
            "mia_results", "gsm8k",
        ),
        "mia_aux_samples_fixed": 100,
    },
}

parser = argparse.ArgumentParser(
    description="Unified Llama-3B zeroth-order training + MIA script (select dataset via --dataset)"
)

# Basic parameters
parser.add_argument("--run_id", type=int, default=0)
parser.add_argument("--is_debug_mode", type=int, default=0)

# Data-related parameters
parser.add_argument(
    '--dataset',
    type=str,
    default='agnews',
    choices=['agnews', 'alpaca', 'dolly', 'dolly15k', 'gsm8k'],
    help='Dataset: agnews / alpaca / dolly(dolly15k) / gsm8k',
)
parser.add_argument(
    '--data_file_path',
    type=str,
    default=None,
    help='Data path; auto-filled from --dataset by default',
)
parser.add_argument(
    '--partition_file_path',
    type=str,
    default=None,
    help='Partition file path (agnews only); auto-filled from --dataset by default',
)
parser.add_argument(
    '--partition_method',
    type=str,
    default=None,
    help='Partition method; auto-filled from --dataset by default',
)

# Model-related parameters
parser.add_argument('--model_type', type=str, default='llama')
parser.add_argument('--model_name', type=str, default='/home/zhike/JWH/model/open_llama_3b_v2/')
parser.add_argument('--do_lower_case', type=bool, default=False)
parser.add_argument('--use_lora', type=int, default=1, help='Enable LoRA (1=on, 0=off)')
parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank r')
parser.add_argument('--lora_alpha', type=int, default=128, help='LoRA alpha')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')

# Training-related parameters
parser.add_argument('--train_batch_size', type=int, default=4,
                    help='Train batch size (keep small for Llama+MIA backward v optimization to save VRAM)')
parser.add_argument('--eval_batch_size', type=int, default=4)
parser.add_argument('--max_seq_length', type=int, default=64)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--fp16', default=False, action="store_true")
parser.add_argument('--manual_seed', type=int, default=42)

# I/O-related parameters
parser.add_argument(
    '--output_dir',
    type=str,
    default=None,
    help='Output dir; default mia_results/<dataset>/ from --dataset',
)

# Training round parameters (simplified)
parser.add_argument('--fl_algorithm', type=str, default='FedFwd')
parser.add_argument('--backend', type=str, default="MPI")
parser.add_argument('--comm_round', type=int, default=1)#3000
parser.add_argument('--is_mobile', type=int, default=0)
parser.add_argument('--client_num_in_total', type=int, default=-1)
parser.add_argument('--client_num_per_round', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--use_centralized_data', type=bool, default=True,
                    help='Use centralized data (all data) instead of federated partitions')
parser.add_argument('--max_train_samples', type=int, default=1000,
                    help='Limit train samples (quick debugging)')
parser.add_argument('--max_test_samples', type=int, default=100,
                    help='Limit test samples (quick debugging)')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--client_optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--server_optimizer', type=str, default='sgd')
parser.add_argument('--server_lr', type=float, default=0.1)
parser.add_argument('--server_momentum', type=float, default=0)
parser.add_argument('--fedprox_mu', type=float, default=1)
parser.add_argument('--evaluate_during_training_steps', type=int, default=100)
parser.add_argument('--frequency_of_the_test', type=int, default=1)

# GPU device management
parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml")
parser.add_argument('--gpu_mapping_key', type=str, default='mapping_myMap')
parser.add_argument('--ci', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:1',
                    help='Device: default cpu (Llama trial saves VRAM); GPU: cuda / cuda:0 / cuda:1')
parser.add_argument(
    '--require_cuda',
    action='store_true',
    default=False,
    help='Exit if --device is cuda but PyTorch sees no GPU (avoid silent CPU fallback).',
)
parser.add_argument(
    '--force_cuda_device',
    type=str,
    default='cuda:1',
    help='If set, align trainer.model to this device before first eval/train (e.g. cuda:1). '
         'Use cuda:0 or leave empty on single-GPU. Syncs self.device in train_model to fix MIA device bugs.',
)

# Cache-related
parser.add_argument('--reprocess_input_data', action='store_true', default=False,
                    help='Reprocess data, ignore cache (fix cache issues)')

# Freeze-related
parser.add_argument('--freeze_layers', type=str, default='')
parser.add_argument('--use_adapter', type=bool, default=False)

# Forward-mode related
parser.add_argument('--forward_mode', action='store_true', default=True)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--worker_num', type=int, default=1)
parser.add_argument('--peft_method', type=str, default='adapter')
parser.add_argument('--var_control', action='store_true', default=True)
parser.add_argument('--perturbation_sampling', action='store_true', default=True)
parser.add_argument('--enable_mia', action='store_true', default=True,
                    help='Enable membership inference attack')
parser.add_argument('--mia_loss_variant', type=str, default='full',
                    choices=['full', 'no_square', 'no_linear'],
                    help='v optimization loss: full=complete, no_square=drop square term, no_linear=drop linear term (ablation)')
parser.add_argument('--mia_aux_sampling', type=str, default='random',
                    choices=['balanced', 'random'],
                    help='Auxiliary sampling: balanced=uniform by label, random=random (may miss labels)')
parser.add_argument('--mia_aux_samples_fixed', type=int, default=None,
                    help='Fixed auxiliary count (>0 priority; 0=proportional to dataset; default follows --dataset)')
parser.add_argument('--mia_aux_agnews_divisor', type=int, default=9,
                    help='When auxiliary count not fixed: num_aux=max(1, total_samples // this divisor)')
parser.add_argument('--mia_use_backprop', type=int, default=1,
                    help='v optimization: 1=backprop (default), 0=evolution strategies (ES)')
parser.add_argument('--mia_fd_h', type=float, default=1e-3,
                    help='MIA finite-diff step theta +/- h*v; large models: 1e-3~1e-4; too large causes logit blow-up')
parser.add_argument('--mia_opt_eval_every', type=int, default=10,
                    help='Steps between full auxiliary-set metric eval during backward v opt (save VRAM); 1=every step')
parser.add_argument('--mia_opt_log_every', type=int, default=1,
                    help='On non-full-eval steps, print grad_norm / v norm every N steps (full steps always print JVP/obj)')
parser.add_argument('--mia_threshold_mode', type=str, default='fixed',
                    choices=['fixed', 'online_quantile'],
                    help='MIA threshold mode: fixed=fixed threshold, online_quantile=online dynamic threshold at test time')
parser.add_argument('--mia_online_warmup', type=int, default=10,
                    help='Warmup samples for online_quantile (warmup accumulates history only, no decisions)')
parser.add_argument('--mia_online_window', type=int, default=40,
                    help='History window size for quantile threshold in online_quantile mode')
parser.add_argument('--mia_online_alpha', type=float, default=0.1,
                    help='Quantile alpha for online_quantile; threshold=quantile(history, 1-alpha)')
parser.add_argument('--mia_online_min_threshold', type=float, default=0.0,
                    help='Lower bound for online_quantile threshold')
parser.add_argument('--mia_online_max_threshold', type=float, default=1e12,
                    help='Upper bound for online_quantile threshold')
parser.add_argument('--mia_online_update_with_neg_only', type=int, default=1,
                    help='online_quantile history update: 1=update only on non-member prediction, 0=all samples')
parser.add_argument('--pseudo_num_labels', type=int, default=4,
                    help='Pseudo-label class count for alpaca/dolly/gsm8k')
parser.add_argument('--dolly_jsonl_name', type=str, default='databricks-dolly-15k.jsonl',
                    help='Dolly JSONL filename when --data_file_path is a directory')
parser.add_argument('--gsm8k_parquet_subdir', type=str, default='main',
                    help='Subdir under gsm8k data_file_path with train-*.parquet / test-*.parquet')
parser.add_argument('--gsm8k_merge_test', type=int, default=1,
                    help='Merge test-*.parquet into train/attack pool (1=yes, 0=no)')

# Parse arguments
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
if args.mia_aux_samples_fixed is None:
    args.mia_aux_samples_fixed = _ds_defaults["mia_aux_samples_fixed"]

import torch
import logging

# ============= Configure logging =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import random
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

# Import required modules
from data_preprocessing.text_classification_preprocessor import TLMPreprocessor
from forward_training.tc_transformer_trainer_distribute import ForwardTextClassificationTrainer
from model.transformer.model_args import ClassificationArgs
from data_manager.text_classification_data_manager import TextClassificationDataManager
from data_manager.base_data_manager import BaseDataManager
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    BertForQuestionAnswering,
    DistilBertConfig,
    DistilBertTokenizer,
    DistilBertForTokenClassification,
    DistilBertForQuestionAnswering,
    BartConfig, 
    BartForConditionalGeneration, 
    BartTokenizer,
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
    LlamaConfig,
    LlamaForSequenceClassification,
    LlamaTokenizer,
)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class TextTupleDataset(Dataset):
    """Five-tuple format compatible with existing trainer."""

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
        raise ValueError(f"{dataset_name} too few samples: {len(all_texts)}, need at least 2.")

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
            f"Empty train/test samples: train={len(train_texts)}, test={len(test_texts)}."
            "Adjust max_train_samples/max_test_samples or the split."
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


def _load_agnews_texts_labels(total_needed, seed, data_file_path):
    """Load text and labels offline from local agnews_data.h5 (no HuggingFace)."""
    import h5py

    rng = random.Random(seed)

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
            picked.extend(label_to_idx[y][:take_per_class[y]])
        rng.shuffle(picked)
        return [texts[i] for i in picked], [int(labels[i]) for i in picked]

    if not os.path.isfile(data_file_path):
        raise FileNotFoundError(
            f"Local AGNews h5 not found: {data_file_path}\n"
            "Check path or set --data_file_path."
        )

    logger.info(f"Loading AGNews offline from local h5: {data_file_path}")
    with h5py.File(data_file_path, "r") as f:
        attrs = json.loads(f["attributes"][()])
        label_vocab = attrs["label_vocab"]
        index_list = list(attrs.get("train_index_list") or attrs["index_list"])
        rng.shuffle(index_list)
        pool_n = min(len(index_list), max(int(total_needed) * 4, int(total_needed)))
        texts, labels = [], []
        for idx in index_list[:pool_n]:
            x = f["X"][str(idx)][()]
            y = f["Y"][str(idx)][()]
            if isinstance(x, (bytes, bytearray)):
                x = x.decode("utf-8")
            if isinstance(y, (bytes, bytearray)):
                y = y.decode("utf-8")
            texts.append(x)
            labels.append(int(label_vocab[str(y)]))

    n = min(int(total_needed), len(texts))
    return balanced_pick(texts, labels, n)


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
            "Check path or pass a directory and set --dolly_jsonl_name."
        )
    raw = _read_dolly_jsonl(jsonl_path)
    texts = [_dolly_record_to_text(item) for item in raw]
    labels = [_dolly_pseudo_label(item, num_labels) for item in raw]
    return texts, labels


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
            "Reading GSM8K parquet requires pyarrow or pandas (try: pip install pyarrow)"
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
            f"Ensure {data_root} has {parquet_subdir}/ with train-*.parquet."
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


def create_model(args, formulation="classification", device=None, prefer_cuda_str=None):
    """Create model, tokenizer, and config.

Note: Transformers ``from_pretrained`` loads weights to **CPU** first (library default, not this repo).
    If ``device`` is passed (e.g. ``torch.device('cuda:1')``), after PEFT/LoRA etc. ``model.to(device)`` runs
    to move to GPU early; pass ``device=None`` if VRAM is tight and migrate only in Trainer.

    ``prefer_cuda_str``: when ``device`` is CPU by mistake but ``torch.cuda.is_available()`` is True,
    pass ``args.device`` string to re-resolve CUDA here (debugging).
    """
    MODEL_CLASSES = {
        "classification": {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
            "roberta-large": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            "deberta": (DebertaConfig, DebertaForSequenceClassification, DebertaTokenizer),
            "llama": (LlamaConfig, LlamaForSequenceClassification, LlamaTokenizer),
        },
    }
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[formulation][args.model_type]
    
    logger.info(f"Loading model: {args.model_name}")
    config = config_class.from_pretrained(args.model_name, **args.config)
    model = model_class.from_pretrained(args.model_name, config=config, ignore_mismatched_sizes=True)
    if args.model_type == "llama":
        tokenizer = tokenizer_class.from_pretrained(args.model_name, use_fast=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer = tokenizer_class.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    
    logger.info(f"Model before PEFT: {model.__class__.__name__}")
    total_params_before = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters before PEFT: {total_params_before:,}")

    # Apply PEFT method
    if args.peft_method == 'adapter' and args.model_type != "llama":
        logger.info("Applying Adapter method...")
        adapter_config = {
            'original_ln_before': True, 
            'original_ln_after': True, 
            'residual_before_ln': True, 
            'adapter_residual_before_ln': False, 
            'ln_before': False, 
            'ln_after': False,
            'mh_adapter': False, 
            'output_adapter': True, 
            'non_linearity': 'relu', 
            'reduction_factor': 16, 
            'inv_adapter': None, 
            'inv_adapter_reduction_factor': None,
            'cross_adapter': False, 
            'leave_out': []
        }
        model.add_adapter("zero_order_adapter", adapter_config)
        model.train_adapter("zero_order_adapter")
        logger.info("Adapter added and activated")
    elif args.model_type == "llama" and bool(args.use_lora):
        logger.info("Applying LoRA for Llama sequence classification...")
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as e:
            raise ImportError(
                "Llama+LoRA requires peft to be installed."
            ) from e
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        logger.info(
            f"LoRA applied: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, "
            f"targets={target_modules}"
        )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    move_dev = device
    if (
        move_dev is not None
        and move_dev.type == "cpu"
        and prefer_cuda_str
        and str(prefer_cuda_str).strip().lower().startswith("cuda")
        and torch.cuda.is_available()
    ):
        ps = str(prefer_cuda_str).strip().lower()
        try:
            if ps == "cuda":
                idx = 0
            else:
                idx = int(ps.split(":")[1])
            if 0 <= idx < torch.cuda.device_count():
                move_dev = torch.device(f"cuda:{idx}")
                torch.cuda.set_device(idx)
                logger.warning(
                    "create_model: device=cpu but CUDA available; switched via prefer_cuda_str=%r to %s",
                    prefer_cuda_str,
                    move_dev,
                )
        except Exception as e:
            logger.warning("create_model: cannot switch to CUDA via prefer_cuda_str: %s", e)

    if move_dev is not None:
        model = model.to(move_dev)
        _first = next(model.parameters())
        if move_dev.type == "cuda" and _first.device.type != "cuda":
            logger.warning("After first .to(cuda) params still on %s; retry .to(%s)", _first.device, move_dev)
            model = model.to(move_dev)
            _first = next(model.parameters())
        logger.info(
            "Model moved to %s (after PEFT); first param device=%s",
            move_dev,
            _first.device,
        )
    return config, model, tokenizer

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info(f"Starting zeroth-order training | dataset={args.dataset}")
    logger.info("=" * 80)
    
    # 1. Set random seed (for model init, etc.)
    set_seed(args.manual_seed)
    logger.info(f"Random seed: {args.manual_seed}")
    
    # 2. Set device (supports cpu / cuda / cuda:N)
    requested_device = str(args.device).strip().lower()
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            if requested_device == "cuda":
                device_index = 0
            else:
                # Support cuda:N with strict index checks to avoid silent fallback to cuda:0
                try:
                    device_index = int(requested_device.split(":")[1])
                except Exception:
                    raise ValueError(f"Invalid device: {args.device}; use cpu/cuda/cuda:N")
            device_count = torch.cuda.device_count()
            if device_index < 0 or device_index >= device_count:
                raise ValueError(
                    f"CUDA device index out of range: cuda:{device_index}, available={device_count}"
                )
            device = torch.device(f"cuda:{device_index}")
            # Without set_device, some ops/cache may still land on cuda:0; nvidia-smi may look like the chosen GPU is unused
            torch.cuda.set_device(device_index)
            logger.info(f"Using CUDA device {device_index}: {torch.cuda.get_device_name(device_index)}")
        else:
            logger.warning(
                f"Requested device={args.device} but CUDA is unavailable, fallback to CPU."
            )
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    logger.info(
        "[CUDA diag] is_available=%s, device_count=%s, torch.version.cuda=%s, CUDA_VISIBLE_DEVICES=%s",
        torch.cuda.is_available(),
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
        getattr(torch.version, "cuda", None),
        os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
    )
    if str(args.device).strip().lower().startswith("cuda") and device.type == "cpu":
        if getattr(args, "require_cuda", False):
            logger.error(
                "--device %s set but running on CPU (often CPU-only PyTorch or invisible driver)."
                "Try: python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"",
                args.device,
            )
            raise SystemExit(2)
        logger.warning(
            "Will run on CPU (see log above). For GPU, install CUDA PyTorch and ensure nvidia-smi works."
        )

    # 3. Load dataset attributes (all four datasets use custom text loaders, bypassing TLMPreprocessor)
    use_custom_text_loader = True
    logger.info(f"Loading dataset from: {args.data_file_path}")
    if args.dataset == "agnews":
        num_labels = 4
        attributes = {"label_vocab": list(range(num_labels))}
        logger.info(f"Dataset: {args.dataset} (offline h5), Labels: {num_labels}")
    else:
        num_labels = int(args.pseudo_num_labels)
        attributes = {"label_vocab": list(range(num_labels))}
        logger.info(f"Dataset: {args.dataset} (custom text), Labels: {num_labels}")
        logger.info("Label vocabulary: pseudo labels via hash(...) mod num_labels")
    logger.info(f"Label vocabulary: {attributes['label_vocab']}")

    # 4. Create model config
    logger.info("Creating model configuration...")
    model_args = ClassificationArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.update_from_dict({
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
                                "use_lora": bool(args.use_lora),
                                "lora_r": args.lora_r,
                                "lora_alpha": args.lora_alpha,
                                "lora_dropout": args.lora_dropout,
                                "var_control": args.var_control,
                                "perturbation_sampling": args.perturbation_sampling,
                                "enable_mia": args.enable_mia,
                                "mia_loss_variant": args.mia_loss_variant,
                                "mia_aux_sampling": args.mia_aux_sampling,
                                "mia_aux_samples_fixed": args.mia_aux_samples_fixed,
                                "mia_aux_agnews_divisor": args.mia_aux_agnews_divisor,
                                "mia_use_backprop": bool(args.mia_use_backprop),
                                "mia_fd_h": float(args.mia_fd_h),
                                "mia_opt_eval_every": int(args.mia_opt_eval_every),
                                "mia_opt_log_every": int(args.mia_opt_log_every),
                                "mia_threshold_mode": args.mia_threshold_mode,
                                "mia_online_warmup": args.mia_online_warmup,
                                "mia_online_window": args.mia_online_window,
                                "mia_online_alpha": args.mia_online_alpha,
                                "mia_online_min_threshold": args.mia_online_min_threshold,
                                "mia_online_max_threshold": args.mia_online_max_threshold,
                                "mia_online_update_with_neg_only": bool(args.mia_online_update_with_neg_only),
                                })
    model_args.config["num_labels"] = num_labels
    
    # Retry CUDA detection before loading the large model (rare envs become ready only after first probe)
    if str(args.device).strip().lower().startswith("cuda") and device.type == "cpu":
        try:
            torch.cuda.init()
        except Exception as _e:
            logger.info("[CUDA] deferred init: %s", _e)
        if torch.cuda.is_available():
            rd = str(args.device).strip().lower()
            try:
                if rd == "cuda":
                    device_index = 0
                else:
                    device_index = int(rd.split(":")[1])
                if 0 <= device_index < torch.cuda.device_count():
                    device = torch.device(f"cuda:{device_index}")
                    torch.cuda.set_device(device_index)
                    logger.warning(
                        "[CUDA] GPU available before model load; device changed from CPU to %s",
                        device,
                    )
            except Exception as _e:
                logger.warning("[CUDA] failed to re-resolve device before model load: %s", _e)

    # 5. Create model
    logger.info("=" * 80)
    logger.info("Creating model...")
    logger.info("=" * 80)
    logger.info("[create_model] using device=%s (prefer_cuda_str=%r)", device, args.device)
    model_config, model, tokenizer = create_model(
        model_args,
        formulation="classification",
        device=device,
        prefer_cuda_str=args.device,
    )
    #print(model)
    # 6. Create trainer (zeroth-order optimization)
    logger.info("=" * 80)
    logger.info("Creating Forward (Zero-Order) Trainer...")
    logger.info("=" * 80)
    trainer = ForwardTextClassificationTrainer(model_args, device, model, None, None)
    if device.type == "cuda":
        _p0 = next(trainer.model.parameters())
        _ix = device.index if device.index is not None else 0
        logger.info(
            f"[GPU check] first layer param device={_p0.device}, "
            f"torch.cuda.current_device()={torch.cuda.current_device()}, "
            f"allocated={torch.cuda.memory_allocated(_ix) / 1024 ** 3:.2f} GiB"
        )
    
    # 7. Create data preprocessor (not needed for custom text branch)
    preprocessor = None
    if not use_custom_text_loader:
        logger.info("Creating data preprocessor...")
        preprocessor = TLMPreprocessor(
            args=model_args,
            label_vocab=attributes["label_vocab"],
            tokenizer=tokenizer
        )
    
    # 8. Load data
    logger.info("=" * 80)
    logger.info("Loading data (centralized mode)...")
    logger.info("Note: DEBUG_MODE=1, using single-process data loading")
    logger.info("=" * 80)
    
    import time
    data_shuffle_seed = int(time.time() * 1000000) % (2**31)
    random.seed(data_shuffle_seed)
    np.random.seed(data_shuffle_seed)
    logger.info(f"Data load random seed: {data_shuffle_seed} (different order each run)")
    
    if args.dataset == "agnews":
        logger.info("Using offline local-h5 AGNews loader for Llama tokenizer compatibility...")
        total_needed = max(
            args.max_train_samples if args.max_train_samples is not None else 1000,
            1000
        ) + max(args.max_test_samples if args.max_test_samples is not None else 100, 100)
        all_texts, all_labels = _load_agnews_texts_labels(
            total_needed=total_needed,
            seed=data_shuffle_seed,
            data_file_path=args.data_file_path,
        )
        train_dl, test_dl, full_train_dl = _split_and_build_loaders(
            all_texts, all_labels, tokenizer, args.max_seq_length,
            model_args.train_batch_size, model_args.eval_batch_size,
            args.max_train_samples, args.max_test_samples, data_shuffle_seed, "AGNews",
        )
    elif args.dataset == "alpaca":
        logger.info("Using Alpaca JSON loader...")
        all_texts, all_labels = _load_alpaca_texts_labels(args.data_file_path, num_labels)
        train_dl, test_dl, full_train_dl = _split_and_build_loaders(
            all_texts, all_labels, tokenizer, args.max_seq_length,
            model_args.train_batch_size, model_args.eval_batch_size,
            args.max_train_samples, args.max_test_samples, data_shuffle_seed, "Alpaca",
        )
    elif args.dataset == "dolly":
        logger.info("Using Dolly15k JSONL loader...")
        all_texts, all_labels = _load_dolly_texts_labels(
            args.data_file_path, args.dolly_jsonl_name, num_labels
        )
        train_dl, test_dl, full_train_dl = _split_and_build_loaders(
            all_texts, all_labels, tokenizer, args.max_seq_length,
            model_args.train_batch_size, model_args.eval_batch_size,
            args.max_train_samples, args.max_test_samples, data_shuffle_seed, "Dolly15k",
        )
    elif args.dataset == "gsm8k":
        logger.info("Using GSM8K parquet loader...")
        all_texts, all_labels = _load_gsm8k_texts_labels(
            args.data_file_path,
            args.gsm8k_parquet_subdir,
            bool(args.gsm8k_merge_test),
            num_labels,
        )
        train_dl, test_dl, full_train_dl = _split_and_build_loaders(
            all_texts, all_labels, tokenizer, args.max_seq_length,
            model_args.train_batch_size, model_args.eval_batch_size,
            args.max_train_samples, args.max_test_samples, data_shuffle_seed, "GSM8K",
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    trainer.train_dl = train_dl
    trainer.test_dl = test_dl
    trainer.set_data(train_dl, test_dl, full_train_dl=full_train_dl)
    
    logger.info(f"Training batches: {len(train_dl)}")
    logger.info(f"Testing batches: {len(test_dl)}")
    logger.info(f"Training samples: {len(train_dl.dataset)}")
    logger.info(f"Testing samples: {len(test_dl.dataset)}")
    logger.info(f"Full dataset samples: {len(full_train_dl.dataset)} (for MIA attack)")
    logger.info(f"mia_aux_samples_fixed={args.mia_aux_samples_fixed}")
    
    _force_dev = str(getattr(args, "force_cuda_device", "") or "").strip()
    if _force_dev:
        if not torch.cuda.is_available():
            logger.warning(f"[GPU] force_cuda_device={_force_dev!r} ignored: CUDA unavailable")
        else:
            device = torch.device(_force_dev)
            trainer.device = device
            if device.type == "cuda":
                _ixf = device.index if device.index is not None else 0
                torch.cuda.set_device(_ixf)
            trainer.model.to(device)
            logger.info(
                f"[GPU] force_cuda_device: trainer and model moved to {device} "
                f"(current_device={torch.cuda.current_device() if device.type == 'cuda' else 'n/a'})"
            )
    
    logger.info("=" * 80)
    logger.info(f"Starting Zero-Order Optimization Training")
    logger.info(f"Training rounds: {args.comm_round}")
    logger.info(f"Epochs per round: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 80)
    
    logger.info("Evaluating initial model...")
    initial_metrics, _, _ = trainer.eval_model()
    logger.info(f"Initial accuracy: {initial_metrics.get('acc', 0):.4f}")
    
    all_round_mia_stats = {
        'rounds': [],
        'member_accuracy': [],
        'non_member_accuracy': [],
        'overall_accuracy': [],
        'tpr': [],
        'fpr': [],
        'auc': [],
        'member_correct': [],
        'member_total': [],
        'non_member_correct': [],
        'non_member_total': []
    }
    
    for round_idx in range(args.comm_round):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Round {round_idx + 1}/{args.comm_round}")
        logger.info("=" * 80)
        
        logger.info(f"📊 [Before train_model] old_grad = {'None' if trainer.old_grad is None else f'List[{len(trainer.old_grad)} tensors]'}")
        
        global_step, tr_loss = trainer.train_model(device)
        logger.info(f"Training completed: global_step={global_step}, loss={tr_loss:.6f}")
        
        if hasattr(trainer, 'mia_stats') and trainer.mia_stats.get('member_total', 0) > 0:
            stats = trainer.mia_stats
            member_acc = stats['member_correct'] / stats['member_total'] if stats['member_total'] > 0 else 0
            non_member_acc = stats['non_member_correct'] / stats['non_member_total'] if stats['non_member_total'] > 0 else 0
            total_eval = stats['member_total'] + stats['non_member_total']
            overall_acc = (stats['member_correct'] + stats['non_member_correct']) / total_eval if total_eval > 0 else 0
            tpr = stats.get('tpr', member_acc)
            fpr = stats.get('fpr', 1.0 - non_member_acc)
            round_auc = float('nan')
            y_true = np.array(stats.get('true_labels', []), dtype=np.int32)
            y_score = np.array(stats.get('jvp_values', []), dtype=np.float64)
            valid = np.isfinite(y_score)
            if valid.any():
                y_true = y_true[valid]
                y_score = y_score[valid]
                if len(y_true) > 1 and len(np.unique(y_true)) > 1:
                    round_auc = roc_auc_score(y_true, y_score)
            
            all_round_mia_stats['rounds'].append(round_idx + 1)
            all_round_mia_stats['member_accuracy'].append(member_acc)
            all_round_mia_stats['non_member_accuracy'].append(non_member_acc)
            all_round_mia_stats['overall_accuracy'].append(overall_acc)
            all_round_mia_stats['tpr'].append(tpr)
            all_round_mia_stats['fpr'].append(fpr)
            all_round_mia_stats['auc'].append(round_auc)
            all_round_mia_stats['member_correct'].append(stats['member_correct'])
            all_round_mia_stats['member_total'].append(stats['member_total'])
            all_round_mia_stats['non_member_correct'].append(stats['non_member_correct'])
            all_round_mia_stats['non_member_total'].append(stats['non_member_total'])

            logger.info("")
            logger.info(f"📊 [Round {round_idx + 1} MIA Stats]")
            logger.info(f"  Member Accuracy: {member_acc:.2%} ({stats['member_correct']}/{stats['member_total']})")
            logger.info(f"  Non-Member Accuracy: {non_member_acc:.2%} ({stats['non_member_correct']}/{stats['non_member_total']})")
            logger.info(f"  Overall Accuracy: {overall_acc:.2%}")
            logger.info(f"  TPR: {tpr:.2%}, FPR: {fpr:.2%}")
            logger.info(f"  AUC: {round_auc:.4f}" if np.isfinite(round_auc) else "  AUC: N/A")
        
        if hasattr(trainer, 'apply_gradient_update'):
            trainer.apply_gradient_update(current_round=round_idx)
            logger.info("Applied gradient update")
            logger.info(f"📊 [After apply_gradient_update] old_grad = {'None' if trainer.old_grad is None else f'List[{len(trainer.old_grad)} tensors]'}")
        else:
            logger.error("❌ apply_gradient_update method NOT found!")
        
        if (round_idx + 1) % args.frequency_of_the_test == 0:
            logger.info("Evaluating model...")
            metrics, _, _ = trainer.eval_model()
            logger.info(f"Evaluation metrics: {metrics}")
            logger.info(f"Accuracy: {metrics.get('acc', 0):.4f}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)
    
    if len(all_round_mia_stats['rounds']) > 0:
        logger.info("")
        logger.info("=" * 100)
        logger.info("[Membership Inference Attack - cross-round summary]")
        logger.info("=" * 100)
        
        total_member_correct = sum(all_round_mia_stats['member_correct'])
        total_member_total = sum(all_round_mia_stats['member_total'])
        total_non_member_correct = sum(all_round_mia_stats['non_member_correct'])
        total_non_member_total = sum(all_round_mia_stats['non_member_total'])
        
        final_member_acc = total_member_correct / total_member_total if total_member_total > 0 else 0
        final_non_member_acc = total_non_member_correct / total_non_member_total if total_non_member_total > 0 else 0
        final_overall_acc = (total_member_correct + total_non_member_correct) / (total_member_total + total_non_member_total) if (total_member_total + total_non_member_total) > 0 else 0
        final_tpr = final_member_acc
        final_fpr = 1.0 - final_non_member_acc
        finite_auc = [x for x in all_round_mia_stats['auc'] if np.isfinite(x)]
        final_auc = float(np.mean(finite_auc)) if len(finite_auc) > 0 else float('nan')
        
        logger.info(f"\n[Overall attack results (all rounds)]")
        logger.info(f"  Dataset: {args.dataset}")
        logger.info(f"  Test rounds: {len(all_round_mia_stats['rounds'])}")
        logger.info(f"  Member samples (Ground Truth: IN batch):")
        logger.info(f"    - Total tested: {total_member_total}")
        logger.info(f"    - Correct: {total_member_correct}")
        logger.info(f"    - Accuracy: {final_member_acc:.2%}")
        logger.info(f"  Non-member samples (Ground Truth: NOT in batch):")
        logger.info(f"    - Total tested: {total_non_member_total}")
        logger.info(f"    - Correct: {total_non_member_correct}")
        logger.info(f"    - Accuracy: {final_non_member_acc:.2%}")
        logger.info(f"  Overall:")
        logger.info(f"    - Total samples: {total_member_total + total_non_member_total}")
        logger.info(f"    - Total correct: {total_member_correct + total_non_member_correct}")
        logger.info(f"    - Overall accuracy: {final_overall_acc:.2%}")
        logger.info(f"    - TPR: {final_tpr:.2%}")
        logger.info(f"    - FPR: {final_fpr:.2%}")
        logger.info(f"    - AUC (mean): {final_auc:.4f}" if np.isfinite(final_auc) else "    - AUC (mean): N/A")
        
        logger.info(f"\n[Per-round attack accuracy]")
        logger.info(f"  Round | Member acc | Non-member acc | Overall acc | TPR | FPR | AUC")
        logger.info(f"  {'-'*100}")
        for i, round_num in enumerate(all_round_mia_stats['rounds']):
            mem_acc = all_round_mia_stats['member_accuracy'][i]
            non_mem_acc = all_round_mia_stats['non_member_accuracy'][i]
            ovr_acc = all_round_mia_stats['overall_accuracy'][i]
            tpr_i = all_round_mia_stats['tpr'][i]
            fpr_i = all_round_mia_stats['fpr'][i]
            auc_i = all_round_mia_stats['auc'][i]
            auc_str = f"{auc_i:.4f}" if np.isfinite(auc_i) else "N/A"
            logger.info(f"  {round_num:4d} |    {mem_acc:6.2%}   |     {non_mem_acc:6.2%}    |    {ovr_acc:6.2%} | {tpr_i:6.2%} | {fpr_i:6.2%} | {auc_str}")
        
        import datetime
        report_file = os.path.join(
            args.output_dir,
            f"mia_attack_report_{args.dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        os.makedirs(args.output_dir, exist_ok=True)
        
        report_data = {
            'attack_config': {
                'dataset': args.dataset,
                'data_file_path': args.data_file_path,
                'batch_size': args.train_batch_size,
                'method': 'Influence Function Analysis with Adversarial v Optimization',
                'total_rounds': len(all_round_mia_stats['rounds']),
                'total_batches_tested': total_member_total
            },
            'final_results': {
                'asr': final_overall_acc,
                'tpr': final_tpr,
                'fpr': final_fpr,
                'auc': final_auc if np.isfinite(final_auc) else None,
                'member_accuracy': final_member_acc,
                'non_member_accuracy': final_non_member_acc,
                'overall_accuracy': final_overall_acc,
                'member_correct': int(total_member_correct),
                'member_total': int(total_member_total),
                'non_member_correct': int(total_non_member_correct),
                'non_member_total': int(total_non_member_total)
            },
            'per_round_results': [
                {
                    'round': int(r),
                    'asr': float(all_round_mia_stats['overall_accuracy'][i]),
                    'tpr': float(all_round_mia_stats['tpr'][i]),
                    'fpr': float(all_round_mia_stats['fpr'][i]),
                    'auc': float(all_round_mia_stats['auc'][i]) if np.isfinite(all_round_mia_stats['auc'][i]) else None,
                    'member_accuracy': float(all_round_mia_stats['member_accuracy'][i]),
                    'non_member_accuracy': float(all_round_mia_stats['non_member_accuracy'][i]),
                    'overall_accuracy': float(all_round_mia_stats['overall_accuracy'][i]),
                    'member_correct': int(all_round_mia_stats['member_correct'][i]),
                    'member_total': int(all_round_mia_stats['member_total'][i]),
                    'non_member_correct': int(all_round_mia_stats['non_member_correct'][i]),
                    'non_member_total': int(all_round_mia_stats['non_member_total'][i])
                }
                for i, r in enumerate(all_round_mia_stats['rounds'])
            ]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nAttack report saved to: {report_file}")
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            try:
                from matplotlib import font_manager
                chinese_fonts = ['AR PL UMing CN', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei']
                available_fonts = {f.name for f in font_manager.fontManager.ttflist}
                font_found = None
                for font_name in chinese_fonts:
                    if font_name in available_fonts:
                        font_found = font_name
                        break
                if font_found:
                    plt.rcParams['font.sans-serif'] = [font_found] + plt.rcParams['font.sans-serif']
            except Exception:
                pass
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            rounds = all_round_mia_stats['rounds']
            
            axes[0, 0].plot(rounds, [x*100 for x in all_round_mia_stats['member_accuracy']], marker='o', label='Member accuracy', linewidth=2)
            axes[0, 0].set_xlabel('Training round'); axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].set_title(f'Member identification accuracy ({args.dataset})'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(rounds, [x*100 for x in all_round_mia_stats['non_member_accuracy']], marker='s', color='orange', label='Non-member accuracy', linewidth=2)
            axes[0, 1].set_xlabel('Training round'); axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title(f'Non-member identification accuracy ({args.dataset})'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].plot(rounds, [x*100 for x in all_round_mia_stats['overall_accuracy']], marker='^', color='green', label='Overall accuracy', linewidth=2)
            axes[1, 0].set_xlabel('Training round'); axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_title(f'Overall attack accuracy ({args.dataset})'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(rounds, [x*100 for x in all_round_mia_stats['member_accuracy']], marker='o', label='Member', linewidth=2)
            axes[1, 1].plot(rounds, [x*100 for x in all_round_mia_stats['non_member_accuracy']], marker='s', label='Non-member', linewidth=2)
            axes[1, 1].plot(rounds, [x*100 for x in all_round_mia_stats['overall_accuracy']], marker='^', label='Overall', linewidth=2)
            axes[1, 1].set_xlabel('Training round'); axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].set_title(f'Attack accuracy comparison ({args.dataset})'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(
                args.output_dir,
                f"mia_attack_results_{args.dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Attack visualization saved to: {plot_file}")
        except Exception as e:
            logger.warning(f"Failed to generate visualization: {e}")
    
    logger.info("=" * 100)
