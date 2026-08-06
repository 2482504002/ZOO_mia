"""
MIA threshold-detection ROC: single-curve plotting, test_records npz caching, multi-dataset combined ROC.
Shared by *-MIA-zhibiao.py and plot_mia_roc_distilbert_4in1.py.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# DistilBERT four-dataset default cache paths and legend names (aligned with roc_scores_cache_path / roc_dataset_label in each zhibiao script)
DEFAULT_DISTILBERT_ROC_NPZ: List[Tuple[str, str]] = [
    ("outputs/mia_roc_scores_distilbert_dolly.npz", "Dolly-15k"),
    ("outputs/mia_roc_scores_distilbert_gsm8k.npz", "GSM8K"),
    ("outputs/mia_roc_scores_distilbert_alpaca.npz", "Alpaca"),
    ("outputs/mia_roc_scores_distilbert_agnews.npz", "AG News"),
]

_MULTI_CURVE_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b")


def roc_plot_title_from_model_name(model_name: str) -> str:
    s = (model_name or "").strip().rstrip("/\\")
    if not s:
        return "ROC"
    base = os.path.basename(s)
    return base if base else s


def test_records_to_y_arrays(test_records):
    if not test_records:
        return None, None
    y_true = np.array([1 if r["member_present"] else 0 for r in test_records], dtype=np.int32)
    y_score = np.array([r["det_score"] for r in test_records], dtype=np.float64)
    return y_true, y_score


def save_roc_scores_npz(
    test_records,
    path: str,
    *,
    dataset_label: str = "",
    model_name: str = "",
) -> bool:
    """Save test-set labels and det_score as npz for combined plotting. Skip if path is empty."""
    path = (path or "").strip()
    if not path or not test_records:
        return False
    y_true, y_score = test_records_to_y_arrays(test_records)
    if y_true is None:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, y_true=y_true, y_score=y_score)
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    try:
        import json

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"dataset_label": dataset_label, "model_name": model_name, "n": int(len(y_true))},
                f,
                ensure_ascii=False,
                indent=2,
            )
    except OSError:
        pass
    print(f"[5] ROC scores cached: {path}")
    return True


def load_roc_scores_npz(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    path = (path or "").strip()
    if not path or not os.path.isfile(path):
        return None
    z = np.load(path)
    return z["y_true"], z["y_score"]


def compute_roc_auc_arrays(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    from sklearn.metrics import auc, roc_curve

    if y_true is None or len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), np.array([]), np.array([])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr)), fpr, tpr


def compute_roc_auc_and_maybe_plot(
    test_records,
    plot_path: str = "",
    model_name: str = "",
    dataset_label: str = "",
    *,
    scores_cache_path: str = "",
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute ROC-AUC on test_records; optionally save a single plot and/or write npz cache.
    Returns (roc_auc, fpr_or_none, tpr_or_none).
    """
    if not test_records:
        return float("nan"), None, None
    y_true, y_score = test_records_to_y_arrays(test_records)
    if y_true is None:
        return float("nan"), None, None
    if len(np.unique(y_true)) < 2:
        print("[5] ROC-AUC: test has single class only; skipping AUC.")
        return float("nan"), None, None

    if (scores_cache_path or "").strip():
        save_roc_scores_npz(
            test_records,
            scores_cache_path,
            dataset_label=dataset_label,
            model_name=model_name,
        )

    try:
        roc_auc, fpr, tpr = compute_roc_auc_arrays(y_true, y_score)
    except ImportError:
        print("[5] ROC-AUC: sklearn not installed; skipping AUC/ROC (pip install scikit-learn).")
        return float("nan"), None, None
    if not np.isfinite(roc_auc):
        return float("nan"), None, None

    plot_path = (plot_path or "").strip()
    if plot_path:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[5] ROC curve: matplotlib not installed; skipping plot (pip install matplotlib).")
            return roc_auc, fpr, tpr
        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_axisbelow(True)
        ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.7)
        leg = (dataset_label or "").strip() or "dataset"
        ax.plot(fpr, tpr, color="red", lw=2, label=leg)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(roc_plot_title_from_model_name(model_name))
        ax.legend(loc="lower right")
        fig.tight_layout()
        ext = os.path.splitext(plot_path)[1].lower()
        if ext in (".pdf", ".svg", ".eps", ".ps"):
            fig.savefig(plot_path, bbox_inches="tight")
        else:
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[5] ROC curve saved: {plot_path}")
    return roc_auc, fpr, tpr


def plot_multi_roc_from_npz_entries(
    entries: Sequence[Union[Tuple[str, str], Tuple[str, str, str]]],
    out_path: str,
    title: str,
    *,
    legend_with_auc: bool = True,
) -> bool:
    """
    entries: [(npz_path, legend_label), ...] or [(path, label, color), ...]
    Missing npz files print a warning and are skipped; at least one valid curve is required to plot.
    """
    try:
        from sklearn.metrics import auc, roc_curve
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[plot_multi_roc] requires sklearn and matplotlib: {e}")
        return False

    curves = []
    for i, ent in enumerate(entries):
        if len(ent) == 3:
            p, label, color = ent[0], ent[1], ent[2]
        else:
            p, label = ent[0], ent[1]
            color = _MULTI_CURVE_COLORS[i % len(_MULTI_CURVE_COLORS)]
        loaded = load_roc_scores_npz(p)
        if loaded is None:
            print(f"[plot_multi_roc] skipping (file missing or invalid): {p}")
            continue
        y_true, y_score = loaded
        if len(np.unique(y_true)) < 2:
            print(f"[plot_multi_roc] skipping (single class): {label} @ {p}")
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = float(auc(fpr, tpr))
        leg = f"{label} (AUC={roc_auc:.3f})" if legend_with_auc else label
        curves.append((fpr, tpr, leg, color))

    if not curves:
        print("[plot_multi_roc] no plottable curves; aborting plot.")
        return False

    out_path = (out_path or "").strip()
    if not out_path:
        return False
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.7)
    for fpr, tpr, leg, color in curves:
        ax.plot(fpr, tpr, color=color, lw=2, label=leg)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".pdf", ".svg", ".eps", ".ps"):
        fig.savefig(out_path, bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_multi_roc] saved: {out_path}")
    return True
