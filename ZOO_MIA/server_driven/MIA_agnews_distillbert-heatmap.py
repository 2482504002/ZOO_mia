import argparse
import sys
import os

# ============= IMPORTANT: enable debug mode to avoid multiprocessing issues =============
os.environ['DEBUG_MODE'] = '1'  # Disable multiprocessing data loading for easier debugging
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # HuggingFace mirror for faster downloads

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Add project root to Python path
sys.path.append('/home/zhike/JWH/FwdLLM-master')

# Create argument parser
parser = argparse.ArgumentParser()

# Basic parameters
parser.add_argument("--run_id", type=int, default=0)
parser.add_argument("--is_debug_mode", type=int, default=0)

# Data-related parameters
parser.add_argument('--dataset', type=str, default='agnews')
parser.add_argument('--data_file_path', type=str, default='/home/zhike/JWH/FwdLLM-master/fednlp_data/data_files/agnews_data.h5')
parser.add_argument('--partition_file_path', type=str, default='/home/zhike/JWH/FwdLLM-master/fednlp_data/partition_files/agnews_partition.h5')
parser.add_argument('--partition_method', type=str, default='uniform_client_1000')

# Model-related parameters
parser.add_argument('--model_type', type=str, default='distilbert')
parser.add_argument('--model_name', type=str, default='/home/zhike/JWH/model/distilbert-base-uncased/')
parser.add_argument('--do_lower_case', type=bool, default=True)

# Training-related parameters
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--max_seq_length', type=int, default=64)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--fp16', default=False, action="store_true")
parser.add_argument('--manual_seed', type=int, default=42)

# I/O-related parameters
parser.add_argument('--output_dir', type=str, default="/home/zhike/JWH/FwdLLM-master/experiments/distributed/transformer_exps/run_tc_exps/mia_results/agnews/",
                    help='Output directory for attack reports and visualizations')

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
                    help='Limit training samples (for quick debugging)')
parser.add_argument('--max_test_samples', type=int, default=100,
                    help='Limit test samples (for quick debugging)')
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
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                    help='Device to run on; default cpu (avoids manually disabling GPU each run)')

# Cache-related
parser.add_argument('--reprocess_input_data', action='store_true', default=False,
                    help='Reprocess data and ignore cache (to fix cache issues)')

# Freezing-related
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
                    help='v-optimization loss variant: full=full, no_square=drop square term, no_linear=drop linear term (ablation)')
parser.add_argument('--mia_aux_sampling', type=str, default='random',
                    choices=['balanced', 'random'],
                    help='Auxiliary dataset sampling: balanced=uniform by label, random=random (may miss some labels)')
parser.add_argument('--mia_threshold_mode', type=str, default='fixed',
                    choices=['fixed', 'online_quantile'],
                    help='MIA threshold mode: fixed=fixed threshold, online_quantile=online dynamic threshold at test time')
parser.add_argument('--mia_online_warmup', type=int, default=10,
                    help='Warmup samples for online_quantile (warmup accumulates history only, no classification)')
parser.add_argument('--mia_online_window', type=int, default=40,
                    help='History window size for quantile threshold estimation in online_quantile mode')
parser.add_argument('--mia_online_alpha', type=float, default=0.1,
                    help='Quantile parameter for online_quantile; threshold=quantile(history, 1-alpha)')
parser.add_argument('--mia_online_min_threshold', type=float, default=0.0,
                    help='Lower bound for online_quantile threshold')
parser.add_argument('--mia_online_max_threshold', type=float, default=1e12,
                    help='Upper bound for online_quantile threshold')
parser.add_argument('--mia_online_update_with_neg_only', type=int, default=1,
                    help='online_quantile history update: 1=update only when predicted non-member, 0=update on all samples')

# FedMeZO style: adversarial init lambda1 x lambda2 grid + Round1 server ||g|| heatmap (separate from Forward MIA)
parser.add_argument('--reli_heatmap_enable', action='store_true',
                    help='Output TASG + server-driven Round1 success-rate heatmap (aligned with distilbert-agnews-MIA-reli.py)')
parser.add_argument('--reli_heatmap_only', action='store_true',
                    help='Run heatmap pipeline only; skip subsequent zero-order/DistilBERT Forward training in this script')
parser.add_argument('--reli_data_root', type=str,
                    default='/home/zhike/JWH/fedmezo-MIA/FedMeZO-main/data/agnews',
                    help='Local AG News parquet root (may contain data/*.parquet); falls back to datasets.load_dataset')
parser.add_argument('--reli_num_clients', type=int, default=2)
parser.add_argument('--reli_fl_samples', type=int, default=2000,
                    help='Samples per client for FL toy training')
parser.add_argument('--reli_local_steps', type=int, default=10)
parser.add_argument('--reli_mezo_batch_size', type=int, default=1)
parser.add_argument('--reli_mezo_lr', type=float, default=1e-5)
parser.add_argument('--reli_zo_eps', type=float, default=1e-3)
parser.add_argument('--reli_attack_samples', type=int, default=500)
parser.add_argument('--reli_heatmap_round1_trials', type=int, default=100,
                    help='Number of Round1 server discrimination trials per heatmap grid point')
parser.add_argument('--reli_adv_init_steps', type=int, default=50)
parser.add_argument('--reli_adv_init_lr', type=float, default=1e-3)
parser.add_argument('--reli_adv_init_anchors_per_client', type=int, default=50)
parser.add_argument('--reli_adv_init_anchor_source', type=str, default='attack_pool',
                    choices=['clients', 'attack_pool'])
parser.add_argument('--reli_w_anchor_grid', type=str, default=None,
                    help='lambda1 (adv_init_w_anchor) grid, comma-separated; default FedMeZO 5 points if omitted')
parser.add_argument('--reli_w_anchor_max_grid', type=str, default=None,
                    help='lambda2 (adv_init_w_anchor_max) grid, comma-separated')

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
    DebertaTokenizer
)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_model(args, formulation="classification"):
    """Create model, tokenizer, and config"""
    MODEL_CLASSES = {
        "classification": {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
            "roberta-large": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            "deberta": (DebertaConfig, DebertaForSequenceClassification, DebertaTokenizer)
        },
    }
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[formulation][args.model_type]
    
    logger.info(f"Loading model: {args.model_name}")
    config = config_class.from_pretrained(args.model_name, **args.config)
    model = model_class.from_pretrained(args.model_name, config=config, ignore_mismatched_sizes=True)
    tokenizer = tokenizer_class.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    
    logger.info(f"Model before PEFT: {model.__class__.__name__}")
    total_params_before = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters before PEFT: {total_params_before:,}")

    # Apply PEFT: only when model class has add_adapter (e.g. adapter-transformers); standard HF DistilBERT lacks this
    if args.peft_method == 'adapter':
        if hasattr(model, 'add_adapter') and hasattr(model, 'train_adapter'):
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
        else:
            logger.warning(
                "peft_method=adapter but %s has no add_adapter/train_adapter (common for standard transformers); "
                "skipping Adapter; training as a plain classification model.",
                model.__class__.__name__,
            )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return config, model, tokenizer

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Starting zero-order optimization training (no MPI, simplified)")
    logger.info("=" * 80)
    
    # 1. Set random seed (for model initialization, etc.)
    set_seed(args.manual_seed)
    logger.info(f"Random seed: {args.manual_seed}")
    
    # 2. Set device (defaults to CPU to avoid manually disabling GPU each run)
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            logger.warning("Requested device=cuda but CUDA is unavailable, fallback to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    if args.reli_heatmap_enable:
        from mia_agnews_relimezo_heatmap import run_relimezo_heatmap_pipeline

        os.makedirs(args.output_dir, exist_ok=True)
        run_relimezo_heatmap_pipeline(args, device)
        if args.reli_heatmap_only:
            logger.info("reli_heatmap_only=True; exiting.")
            sys.exit(0)
    
    # 3. Load dataset attributes
    logger.info(f"Loading dataset from: {args.data_file_path}")
    attributes = BaseDataManager.load_attributes(args.data_file_path)
    num_labels = len(attributes["label_vocab"])
    logger.info(f"Dataset: {args.dataset}, Labels: {num_labels}")
    logger.info(f"Label vocabulary: {attributes['label_vocab']}")

    # 4. Create model configuration
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
                                "var_control": args.var_control,
                                "perturbation_sampling": args.perturbation_sampling,
                                "enable_mia": args.enable_mia,
                                "mia_loss_variant": args.mia_loss_variant,
                                "mia_aux_sampling": args.mia_aux_sampling,
                                "mia_threshold_mode": args.mia_threshold_mode,
                                "mia_online_warmup": args.mia_online_warmup,
                                "mia_online_window": args.mia_online_window,
                                "mia_online_alpha": args.mia_online_alpha,
                                "mia_online_min_threshold": args.mia_online_min_threshold,
                                "mia_online_max_threshold": args.mia_online_max_threshold,
                                "mia_online_update_with_neg_only": bool(args.mia_online_update_with_neg_only),
                                })
    model_args.config["num_labels"] = num_labels
    
    # 5. Create model
    logger.info("=" * 80)
    logger.info("Creating model...")
    logger.info("=" * 80)
    model_config, model, tokenizer = create_model(model_args, formulation="classification")
    print(model)
    # 6. Create trainer (zero-order optimization)
    logger.info("=" * 80)
    logger.info("Creating Forward (Zero-Order) Trainer...")
    logger.info("=" * 80)
    trainer = ForwardTextClassificationTrainer(model_args, device, model, None, None)
    
    # 7. Create data preprocessor
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
    
    # Re-randomize before data loading so each run uses a different order
    # This lets attack tests run under different data distributions
    import time
    data_shuffle_seed = int(time.time() * 1000000) % (2**31)
    random.seed(data_shuffle_seed)
    np.random.seed(data_shuffle_seed)
    logger.info(f"Data loading shuffle seed: {data_shuffle_seed} (different order each run)")
    
    # Use centralized data loading (no client split)
    dm = TextClassificationDataManager(
        args, model_args, preprocessor, 
        process_id=0,  # Server ID: load all data
        num_workers=args.client_num_per_round
    )
    
    # Load all training and test data
    if args.use_centralized_data:
        logger.info("Using centralized data (all data combined)...")
        train_dl, test_dl = dm.load_centralized_data(cut_off=None)
        # For centralized data, training data is the full dataset
        full_train_dl = train_dl
    else:
        logger.info("Using federated data loading (server mode)...")
        _, _, test_dl, _, _, _, _ = dm.load_federated_data(process_id=0)
        # Use one client's data as training data
        dm_client = TextClassificationDataManager(
            args, model_args, preprocessor, 
            process_id=1,
            num_workers=args.client_num_per_round
        )
        _, _, _, _, train_data_local_dict, _, _ = dm_client.load_federated_data(process_id=1)
        train_dl = train_data_local_dict[0]  # Use first client's data
        
        # For federated learning, load full dataset for MIA (1% of full set as auxiliary)
        logger.info("Loading full dataset for MIA attack...")
        full_train_dl, _ = dm.load_centralized_data(cut_off=None)
        logger.info(f"Full dataset size: {len(full_train_dl.dataset)} samples")
    
    # Shuffle dataset so each run uses a different order
    # This lets attack tests run under different data distributions
    # Use RandomSampler for shuffle to avoid index mismatch
    from torch.utils.data import RandomSampler
    if hasattr(train_dl, 'dataset') and train_dl.dataset is not None:
        dataset_size = len(train_dl.dataset)
        # Create RandomSampler with the seed set above
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
        logger.info(f"Training dataset shuffled (size: {dataset_size})")
    
    # Also shuffle full dataset (for MIA attack)
    if hasattr(full_train_dl, 'dataset') and full_train_dl.dataset is not None:
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
    
    # Limit data size (for quick debugging)
    # Note: apply limits after shuffle so we take the first N shuffled samples
    if args.max_train_samples is not None:
        logger.info(f"Limiting training samples to {args.max_train_samples}...")
        original_train_size = len(train_dl.dataset)
        # Create subset (first N samples after shuffle)
        from torch.utils.data import Subset
        # RandomSampler already randomized order; take first N indices
        # So we can take the first N indices directly
        train_indices = list(range(min(args.max_train_samples, len(train_dl.dataset))))
        train_subset = Subset(train_dl.dataset, train_indices)
        
        # Recreate DataLoader (no sampler; Subset already limits range)
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
    
    if args.max_test_samples is not None:
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
    
    # 9. Start training
    logger.info("=" * 80)
    logger.info(f"Starting Zero-Order Optimization Training")
    logger.info(f"Training rounds: {args.comm_round}")
    logger.info(f"Epochs per round: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 80)
    
    # Initial evaluation
    logger.info("Evaluating initial model...")
    initial_metrics, _, _ = trainer.eval_model()  # Returns (result, model_outputs, wrong)
    logger.info(f"Initial accuracy: {initial_metrics.get('acc', 0):.4f}")
    
    # Collect MIA results across rounds
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
        logger.info("Round training step finished.")
        logger.info("=" * 80)

    # ============= Generate final MIA report =============
    if len(all_round_mia_stats['rounds']) > 0:
        logger.info("")
        logger.info("=" * 100)
        logger.info("[MIA - Cross-Round Summary Report]")
        logger.info("=" * 100)
        
        # Compute aggregate statistics
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
        
        logger.info(f"\n[Overall Attack Performance (All Rounds)]")
        logger.info(f"  Test rounds: {len(all_round_mia_stats['rounds'])}")
        logger.info(f"  Member samples (Ground Truth: IN batch):")
        logger.info(f"    - Total tested: {total_member_total}")
        logger.info(f"    - Correctly identified: {total_member_correct}")
        logger.info(f"    - Accuracy: {final_member_acc:.2%}")
        logger.info(f"  Non-member samples (Ground Truth: NOT in batch):")
        logger.info(f"    - Total tested: {total_non_member_total}")
        logger.info(f"    - Correctly identified: {total_non_member_correct}")
        logger.info(f"    - Accuracy: {final_non_member_acc:.2%}")
        logger.info(f"  Overall performance:")
        logger.info(f"    - Total test samples: {total_member_total + total_non_member_total}")
        logger.info(f"    - Total correct: {total_member_correct + total_non_member_correct}")
        logger.info(f"    - Overall accuracy: {final_overall_acc:.2%}")
        logger.info(f"    - TPR: {final_tpr:.2%}")
        logger.info(f"    - FPR: {final_fpr:.2%}")
        logger.info(f"    - AUC (mean): {final_auc:.4f}" if np.isfinite(final_auc) else "    - AUC (mean): N/A")
        
        # Per-round accuracy changes
        logger.info(f"\n[Per-Round Attack Accuracy]")
        logger.info(f"  Round | Member Acc | Non-Member Acc | Overall Acc | TPR | FPR | AUC")
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
        
        # Save statistics to file
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
            ]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nAttack report saved to: {report_file}")
        
        # Visualization (if available)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Configure CJK font (legacy; labels are now English)
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
            axes[0, 0].set_title('Member Sample Identification Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: non-member accuracy over rounds
            axes[0, 1].plot(rounds, [x*100 for x in all_round_mia_stats['non_member_accuracy']], 
                          marker='s', color='orange', label='Non-member accuracy', linewidth=2)
            axes[0, 1].axhline(55.6, color='r', linestyle='--', alpha=0.5, label='Expected (55.6%)')
            axes[0, 1].set_xlabel('Training round')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Non-Member Sample Identification Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: overall accuracy over rounds
            axes[1, 0].plot(rounds, [x*100 for x in all_round_mia_stats['overall_accuracy']], 
                          marker='^', color='green', label='Overall accuracy', linewidth=2)
            axes[1, 0].axhline(76.2, color='r', linestyle='--', alpha=0.5, label='Expected (76.2%)')
            axes[1, 0].set_xlabel('Training round')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_title('Overall Attack Accuracy')
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
            axes[1, 1].set_title('Attack Accuracy Comparison')
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