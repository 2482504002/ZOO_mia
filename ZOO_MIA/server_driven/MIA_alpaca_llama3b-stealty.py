import argparse
import sys
import os
import pdb
import json
import hashlib
# ============= IMPORTANT: set debug mode to avoid multiprocessing issues =============
os.environ['DEBUG_MODE'] = '1'  # disable multiprocess data loading for easier debugging
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # HuggingFace mirror for faster downloads

# Prefer this repo (server-llama)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Create argument parser
parser = argparse.ArgumentParser()

# Basic parameters
parser.add_argument("--run_id", type=int, default=0)
parser.add_argument("--is_debug_mode", type=int, default=0)

# Data-related parameters
parser.add_argument('--dataset', type=str, default='alpaca')
parser.add_argument('--data_file_path', type=str, default='/home/zhike/JWH/fedmezo-MIA/FedMeZO-main/data/alpaca_data.json')
parser.add_argument('--partition_file_path', type=str, default='')
parser.add_argument('--partition_method', type=str, default='alpaca_default')

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
parser.add_argument('--manual_seed', type=int, default=-1,
                    help='Random seed; default -1 random each run for variance; positive int for reproducibility')

# I/O-related parameters
parser.add_argument('--output_dir', type=str, default="/home/zhike/JWH/FwdLLM-master/experiments/distributed/transformer_exps/run_tc_exps/mia_results/alpaca/",
                    help='Output directory for attack reports and plots')

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
parser.add_argument('--mia_aux_samples_fixed', type=int, default=100,
                    help='Fixed auxiliary count (>0 priority; 0=same proportional sampling as DistilBERT)')
parser.add_argument('--mia_aux_agnews_divisor', type=int, default=9,
                    help='Auxiliary count divisor when not fixed: num_aux=max(1, total_samples // this divisor)')
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
parser.add_argument('--alpaca_num_labels', type=int, default=4,
                    help='Alpaca JSON pseudo-label class count (Alpaca JSON loader only)')
parser.add_argument('--stealth_only', action='store_true', default=False,
                    help='Run v optimization and stealth metrics only; skip training loop')
parser.add_argument('--mia_stealth_metrics', type=int, default=1,
                    help='After v opt, compare Original (random v) vs After Adv. Init. (adversarial v) on four stealth metrics')
parser.add_argument('--mia_stealth_aux_cap', type=int, default=0,
                    help='Max aux subsample for stealth JVP eval; 0=use all auxiliary samples')
parser.add_argument('--force_reoptimize_mia', type=int, default=1,
                    help='1=ignore v cache and re-optimize; 0=allow cache load')

# Parse arguments
args = parser.parse_args()

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


class AlpacaTupleDataset(Dataset):
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


def _build_alpaca_dataloader(texts, labels, tokenizer, max_seq_length, batch_size, shuffle):
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    label_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = AlpacaTupleDataset(input_ids, attention_mask, label_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    # Compatible with BaseDataLoader; eval_model accesses examples/features
    dataloader.examples = list(range(len(dataset)))
    dataloader.features = dataloader.examples
    return dataloader

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


def _save_stealth_metrics_report(metrics, output_dir, logger):
    """Save stealth comparison table to JSON and return file path."""
    import datetime
    from forward_training.server_mia_initialization import (
        format_stealth_metrics_table,
        format_stealth_metrics_table_zh,
    )

    if not metrics:
        logger.warning("[stealth] no metrics to save")
        return None

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"mia_stealth_metrics_{ts}.json")

    payload = {
        "description": "Original=random v (same normalization); After Adv. Init.=adversarial optimized v",
        "table_en": format_stealth_metrics_table(metrics),
        "table_zh": format_stealth_metrics_table_zh(metrics),
        "metrics": metrics,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("=" * 80)
    logger.info("[Attack stealth metrics table]")
    logger.info("=" * 80)
    logger.info("\n%s", payload["table_zh"])
    logger.info("\n%s", payload["table_en"])
    logger.info("Stealth report saved: %s", report_path)
    return report_path


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Starting zeroth-order training (no MPI, simplified)")
    logger.info("=" * 80)
    
    # 1. Set random seed (random each run by default for variance; pass --manual_seed N to reproduce)
    import time
    if args.manual_seed is None or int(args.manual_seed) < 0:
        args.manual_seed = int(time.time() * 1000000) % (2**31)
        logger.info(f"Random seed (auto): {args.manual_seed}")
    else:
        args.manual_seed = int(args.manual_seed)
        logger.info(f"Random seed (fixed): {args.manual_seed}")
    set_seed(args.manual_seed)
    
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

    # 3. Load dataset attributes (Alpaca JSON custom branch, avoiding h5 dependency)
    use_alpaca_json = (args.dataset.lower() == "alpaca") or args.data_file_path.lower().endswith(".json")
    logger.info(f"Loading dataset from: {args.data_file_path}")
    if use_alpaca_json:
        num_labels = int(args.alpaca_num_labels)
        attributes = {"label_vocab": list(range(num_labels))}
        logger.info(f"Dataset: {args.dataset} (json mode), Labels: {num_labels}")
        logger.info("Label vocabulary: pseudo labels generated by hash(instruction, output) mod num_labels")
    else:
        attributes = BaseDataManager.load_attributes(args.data_file_path)
        num_labels = len(attributes["label_vocab"])
        logger.info(f"Dataset: {args.dataset}, Labels: {num_labels}")
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
                                "dataset": args.dataset,
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
                                "mia_stealth_metrics": bool(args.mia_stealth_metrics),
                                "mia_stealth_aux_cap": int(args.mia_stealth_aux_cap),
                                "force_reoptimize_mia": bool(args.force_reoptimize_mia),
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
    
    # 7. Create data preprocessor (JSON branch does not depend on it)
    preprocessor = None
    if not use_alpaca_json:
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
    
    # Data shuffle follows global seed (default seed differs each run, so order changes too)
    data_shuffle_seed = int(args.manual_seed)
    random.seed(data_shuffle_seed)
    np.random.seed(data_shuffle_seed)
    logger.info(f"Data load random seed: {data_shuffle_seed}")
    
    if use_alpaca_json:
        logger.info("Using Alpaca JSON loader")
        all_texts, all_labels = _load_alpaca_texts_labels(args.data_file_path, num_labels)
        if len(all_texts) < 2:
            raise ValueError(f"Alpaca JSON too few samples: {len(all_texts)}, need at least 2.")

        indices = list(range(len(all_texts)))
        rng = random.Random(data_shuffle_seed)
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

        if args.max_train_samples is not None:
            train_texts = train_texts_full[: min(args.max_train_samples, len(train_texts_full))]
            train_labels = train_labels_full[: min(args.max_train_samples, len(train_labels_full))]
        else:
            train_texts = train_texts_full
            train_labels = train_labels_full

        if args.max_test_samples is not None:
            test_texts = test_texts[: min(args.max_test_samples, len(test_texts))]
            test_labels = test_labels[: min(args.max_test_samples, len(test_labels))]

        if len(train_texts) == 0 or len(test_texts) == 0:
            raise ValueError(
                f"Empty train/test samples: train={len(train_texts)}, test={len(test_texts)}."
                "Adjust max_train_samples/max_test_samples or the split."
            )

        full_train_dl = _build_alpaca_dataloader(
            train_texts_full, train_labels_full, tokenizer, args.max_seq_length, model_args.train_batch_size, True
        )
        train_dl = _build_alpaca_dataloader(
            train_texts, train_labels, tokenizer, args.max_seq_length, model_args.train_batch_size, True
        )
        test_dl = _build_alpaca_dataloader(
            test_texts, test_labels, tokenizer, args.max_seq_length, model_args.eval_batch_size, False
        )
        logger.info(
            f"Alpaca JSON loaded: total={len(all_texts)}, train_full={len(train_texts_full)}, "
            f"train={len(train_texts)}, test={len(test_texts)}"
        )
    else:
        # Centralized data loading (no client split)
        dm = TextClassificationDataManager(
            args, model_args, preprocessor,
            process_id=0,  # server ID, load all data
            num_workers=args.client_num_per_round
        )

        # Load all train and test data
        if args.use_centralized_data:
            logger.info("Using centralized data (all data combined)...")
            train_dl, test_dl = dm.load_centralized_data(cut_off=None)
            # For centralized mode, train data is the full dataset
            full_train_dl = train_dl
        else:
            logger.info("Using federated data loading (server mode)...")
            _, _, test_dl, _, _, _, _ = dm.load_federated_data(process_id=0)
            # Use one client's data as train data
            dm_client = TextClassificationDataManager(
                args, model_args, preprocessor,
                process_id=1,
                num_workers=args.client_num_per_round
            )
            _, _, _, _, train_data_local_dict, _, _ = dm_client.load_federated_data(process_id=1)
            train_dl = train_data_local_dict[0]  # use first client's data

            # For federated learning, load full dataset for MIA (1% of full set as auxiliary)
            logger.info("Loading full dataset for MIA attack...")
            full_train_dl, _ = dm.load_centralized_data(cut_off=None)
            logger.info(f"Full dataset size: {len(full_train_dl.dataset)} samples")
    
    # Shuffle dataset so order differs each run
    # So attack tests run under different data distributions
    # Use RandomSampler for shuffle to avoid index mismatch
    from torch.utils.data import RandomSampler
    if (not use_alpaca_json) and hasattr(train_dl, 'dataset') and train_dl.dataset is not None:
        dataset_size = len(train_dl.dataset)
        # Create RandomSampler with the seed set earlier
        sampler = RandomSampler(train_dl.dataset, generator=torch.Generator().manual_seed(data_shuffle_seed))
        # Recreate DataLoader with RandomSampler
        from data_preprocessing.base.base_data_loader import BaseDataLoader
        train_dl = BaseDataLoader(
            train_dl.examples,
            train_dl.features,
            train_dl.dataset,
            batch_size=model_args.train_batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Train dataset shuffled (size: {dataset_size})")
    
    # Also shuffle full dataset (for MIA attack)
    if (not use_alpaca_json) and hasattr(full_train_dl, 'dataset') and full_train_dl.dataset is not None:
        full_dataset_size = len(full_train_dl.dataset)
        full_sampler = RandomSampler(full_train_dl.dataset, generator=torch.Generator().manual_seed(data_shuffle_seed))
        from data_preprocessing.base.base_data_loader import BaseDataLoader
        full_train_dl = BaseDataLoader(
            full_train_dl.examples,
            full_train_dl.features,
            full_train_dl.dataset,
            batch_size=model_args.train_batch_size,
            sampler=full_sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Full dataset shuffled (size: {full_dataset_size})")
    
    # Limit sample count (for quick debugging)
    # Note: apply limits after shuffle so we take first N from shuffled data
    if (not use_alpaca_json) and args.max_train_samples is not None:
        logger.info(f"Limiting training samples to {args.max_train_samples}...")
        original_train_size = len(train_dl.dataset)
        # Create subset (first N from shuffled data)
        from torch.utils.data import Subset
        # RandomSampler already randomizes order
        # so taking first N indices is fine
        train_indices = list(range(min(args.max_train_samples, len(train_dl.dataset))))
        train_subset = Subset(train_dl.dataset, train_indices)
        
        # Recreate DataLoader (drop sampler; Subset already limits range)
        from data_preprocessing.base.base_data_loader import BaseDataLoader
        train_dl = BaseDataLoader(
            train_dl.examples[:args.max_train_samples],
            train_dl.features[:args.max_train_samples],
            train_subset,
            batch_size=model_args.train_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Training data reduced: {original_train_size} → {len(train_dl.dataset)}")
    
    if (not use_alpaca_json) and args.max_test_samples is not None:
        logger.info(f"Limiting test samples to {args.max_test_samples}...")
        original_test_size = len(test_dl.dataset)
        # Create subset
        from torch.utils.data import Subset
        test_indices = list(range(min(args.max_test_samples, len(test_dl.dataset))))
        test_subset = Subset(test_dl.dataset, test_indices)
        
        # Recreate DataLoader
        from data_preprocessing.base.base_data_loader import BaseDataLoader
        test_dl = BaseDataLoader(
            test_dl.examples[:args.max_test_samples],
            test_dl.features[:args.max_test_samples],
            test_subset,
            batch_size=model_args.eval_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Test data reduced: {original_test_size} → {len(test_dl.dataset)}")
    
    trainer.train_dl = train_dl
    trainer.test_dl = test_dl
    # Set full dataset for MIA (1% of full set as auxiliary)
    trainer.set_data(train_dl, test_dl, full_train_dl=full_train_dl)
    
    logger.info(f"Training batches: {len(train_dl)}")
    logger.info(f"Testing batches: {len(test_dl)}")
    logger.info(f"Training samples: {len(train_dl.dataset)}")
    logger.info(f"Testing samples: {len(test_dl.dataset)}")
    logger.info(f"Full dataset samples: {len(full_train_dl.dataset)} (for MIA attack)")
    
    # Optional: force CUDA device and sync trainer (avoid stale self.device in MIA)
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

    if args.stealth_only:
        logger.info("=" * 80)
        logger.info("Stealth-only mode: v optimization + stealth metrics only; skipping training loop")
        logger.info("=" * 80)
        stealth_metrics = trainer.run_mia_stealth_eval_only(device)
        _save_stealth_metrics_report(stealth_metrics, args.output_dir, logger)
        raise SystemExit(0)

    # 9. Start training
    logger.info("=" * 80)
    logger.info(f"Starting Zero-Order Optimization Training")
    logger.info(f"Training rounds: {args.comm_round}")
    logger.info(f"Epochs per round: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 80)
    
    # Initial evaluation
    logger.info("Evaluating initial model...")
    initial_metrics, _, _ = trainer.eval_model()  # returns (result, model_outputs, wrong)
    logger.info(f"Initial accuracy: {initial_metrics.get('acc', 0):.4f}")
    
    # Collect MIA stats across rounds
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
    
    # Training loop
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

        if round_idx == 0 and getattr(trainer, "mia_stealth_metrics", None):
            _save_stealth_metrics_report(trainer.mia_stealth_metrics, args.output_dir, logger)

        if hasattr(trainer, 'mia_stats') and trainer.mia_stats.get('member_total', 0) + trainer.mia_stats.get('non_member_total', 0) > 0:
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
            logger.info(
                f"📊 [After apply_gradient_update] old_grad = "
                f"{'None' if trainer.old_grad is None else f'List[{len(trainer.old_grad)} tensors]'}"
            )
        else:
            logger.error("❌ apply_gradient_update method NOT found!")

        if (round_idx + 1) % args.frequency_of_the_test == 0:
            logger.info("Evaluating model...")
            metrics, _, _ = trainer.eval_model()
            logger.info(f"Evaluation metrics: {metrics}")
            logger.info(f"Accuracy: {metrics.get('acc', 0):.4f}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("Round completed.")
        logger.info("=" * 80)

    # ============= Generate final MIA report =============
    if len(all_round_mia_stats['rounds']) > 0:
        logger.info("")
        logger.info("=" * 100)
        logger.info("[Membership Inference Attack - cross-round summary]")
        logger.info("=" * 100)
        
        # Compute aggregate stats
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
        
        # Per-round accuracy changes
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
        
        # Save stats to file
        import json
        import datetime
        report_file = os.path.join(args.output_dir, f"mia_attack_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        os.makedirs(args.output_dir, exist_ok=True)
        
        report_data = {
            'attack_config': {
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
                    'asr': float(ovr_acc),
                    'tpr': float(all_round_mia_stats['tpr'][i]),
                    'fpr': float(all_round_mia_stats['fpr'][i]),
                    'auc': float(all_round_mia_stats['auc'][i]) if np.isfinite(all_round_mia_stats['auc'][i]) else None,
                    'member_accuracy': float(mem_acc),
                    'non_member_accuracy': float(non_mem_acc),
                    'overall_accuracy': float(ovr_acc),
                    'member_correct': int(all_round_mia_stats['member_correct'][i]),
                    'member_total': int(all_round_mia_stats['member_total'][i]),
                    'non_member_correct': int(all_round_mia_stats['non_member_correct'][i]),
                    'non_member_total': int(all_round_mia_stats['non_member_total'][i])
                }
                for i, r in enumerate(all_round_mia_stats['rounds'])
            ],
            'stealth_metrics': getattr(trainer, 'mia_stealth_metrics', None),
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nAttack report saved to: {report_file}")
        
        # Visualization (if possible)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Configure CJK font for plots
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
            except:
                pass
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            rounds = all_round_mia_stats['rounds']
            
            # Plot 1: member accuracy over rounds
            axes[0, 0].plot(rounds, [x*100 for x in all_round_mia_stats['member_accuracy']], 
                          marker='o', label='Member accuracy', linewidth=2)
            axes[0, 0].axhline(96.8, color='r', linestyle='--', alpha=0.5, label='Expected (96.8%)')
            axes[0, 0].set_xlabel('Training round')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].set_title('Member identification accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: non-member accuracy over rounds
            axes[0, 1].plot(rounds, [x*100 for x in all_round_mia_stats['non_member_accuracy']], 
                          marker='s', color='orange', label='Non-member accuracy', linewidth=2)
            axes[0, 1].axhline(55.6, color='r', linestyle='--', alpha=0.5, label='Expected (55.6%)')
            axes[0, 1].set_xlabel('Training round')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Non-member identification accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: overall accuracy over rounds
            axes[1, 0].plot(rounds, [x*100 for x in all_round_mia_stats['overall_accuracy']], 
                          marker='^', color='green', label='Overall accuracy', linewidth=2)
            axes[1, 0].axhline(76.2, color='r', linestyle='--', alpha=0.5, label='Expected (76.2%)')
            axes[1, 0].set_xlabel('Training round')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_title('Overall attack accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: combined comparison
            axes[1, 1].plot(rounds, [x*100 for x in all_round_mia_stats['member_accuracy']], 
                          marker='o', label='Member', linewidth=2)
            axes[1, 1].plot(rounds, [x*100 for x in all_round_mia_stats['non_member_accuracy']], 
                          marker='s', label='Non-member', linewidth=2)
            axes[1, 1].plot(rounds, [x*100 for x in all_round_mia_stats['overall_accuracy']], 
                          marker='^', label='Overall', linewidth=2)
            axes[1, 1].set_xlabel('Training round')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].set_title('Attack accuracy comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(args.output_dir, f"mia_attack_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Attack visualization saved to: {plot_file}")
        except Exception as e:
            logger.warning(f"Failed to generate visualization: {e}")
        
    logger.info("=" * 100)