# ZOO_MIA

Membership Inference Attack (**MIA**) experiments for federated learning under **zeroth-order optimization** (MeZO / FedFwd).

| Pipeline | Directory | Description |
|----------|-----------|-------------|
| Client-driven | `client_driven/` | Single-process FedMeZO sim: local MeZO → upload `g_sum` → server detection |
| Server-driven | `server_driven/` | MIA inside a real FL loop via `FwdLLM-master` |
| On-device (Android) | `commu/` | Real-phone FL: Flask server + Android MeZO clients (PyTorch Android) |

**Models:** DistilBERT, Open-Llama-3B + LoRA.  
**Data:** AG News / Alpaca / Dolly / GSM8K (`datasets/`); some runs use BBC / HuffPost as out-of-domain aux sets.

---

## Environment

```text
torch, transformers, peft, numpy, pandas, tqdm, scikit-learn, matplotlib, h5py
```

Download pretrained weights locally (default paths in scripts may need editing):

```text
distilbert-base-uncased
open_llama_3b_v2
```

---

## Quick start

```bash
# Client: DistilBERT / Llama multi-dataset
cd client_driven
python MIA_distillbert.py --dataset agnews --device cuda:0
python MIA_llama3b.py --dataset alpaca --device cuda:0

# Client: upload defences (dp / spas / topk)
python llama3b-alpaca-MIA-defence.py --defence dp
python llama3b-alpaca-MIA-defence.py --defence spas
python llama3b-alpaca-MIA-defence.py --defence topk

# Server: FwdLLM Forward MIA
cd ../server_driven
python MIA_distillbert.py --dataset agnews --device cuda
python MIA_llama3b.py --dataset alpaca --device cuda:0
```

Outputs usually land in each folder’s `outputs/`, or under  
`server_driven/FwdLLM-master/.../mia_results/<dataset>/`.

### On-device FL (`commu/`)

Android MeZO clients + a Flask aggregation / MIA server for phone experiments.

```bash
cd commu
# start server (see env vars in commu/readme.md)
python server.py
# build & install the Android app (Android Studio or ./gradlew)
```

Full setup (venv, AG News / DistilBERT paths, phone networking): see [`commu/readme.md`](commu/readme.md).

---

## Scripts

### Client (`client_driven/`)

| Script | Purpose |
|--------|---------|
| `MIA_distillbert.py` | DistilBERT unified entry (`--dataset`) |
| `MIA_llama3b.py` | Llama-3B unified entry (`--dataset`) |
| `llama3b-alpaca-MIA-defence.py` | Upload defence: `dp` Gaussian noise / `spas` random sparsity / `topk` magnitude Top-K |
| `distilbert-agnews-MIA-zhibiao.py` | AG News full metrics + 2×2 density panel |
| `distilbert-agnews-bbcnews.py` / `*-huff.py` | BBC / HuffPost aux + size sweep |
| `distilbert-agnews-muticlient.py` | Multi-client scaling |
| `llama3b-alpaca-stealth.py` | Stealth metrics before/after adv init |
| `llama3b-alpaca-ablation.py` | Adv-init ablation (full / wo_mean / wo_max / target_only) |
| `llama3b-alpaca-xrNegative-Only.py` | Online threshold neg-only / window ablation |
| `mia_roc_plotting.py` | Shared ROC helpers (imported by other scripts) |

### Server (`server_driven/`)

| Script | Purpose |
|--------|---------|
| `MIA_distillbert.py` / `MIA_llama3b.py` | Multi-dataset unified entry (`--dataset`) |
| `MIA_agnews_distillbert-bbc.py` / `MIA_agnews_llama3b-BBC.py` | BBC aux + size sweep |
| `MIA_agnews_distillbert-heatmap.py` | λ-grid adv-init heatmap |
| `MIA_alpaca_distillbert-muticlient.py` | Multi-client FL simulation |
| `MIA_alpaca_llama3b-stealty.py` | Stealth metrics (`--stealth_only` supported) |
| `MIA_alpaca_llama3b-negonly.py` | Online threshold neg-only ablation |

Scripts without argparse: run directly and edit the in-file `Config`.

---

## Data

```text
datasets/agnews/
datasets/alpaca_data.json
datasets/dolly15k/databricks-dolly-15k.jsonl
datasets/gsm8k/                  # local parquet only
datasets/bbc-news/
datasets/HuffPost_News_Category/
```

---

## Notes

- Some default paths are machine-specific; override with `--data_root` / `--data_path` / `--model_name`.
- Run from `client_driven/` or `server_driven/` so relative `outputs/` resolve correctly.
- Client and server pipelines differ in detail — do not compare metrics blindly.
- For research use only; do not attack unauthorized data.
