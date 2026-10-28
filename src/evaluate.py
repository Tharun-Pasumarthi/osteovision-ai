"""
Standard evaluation module for all models (ML, DL, and Multimodal).
Computes clinical metrics, per-class diagnostics, confusion matrices, and ROC-AUC curves.
"""
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ["Normal", "Osteopenia", "Osteoporosis"]

def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """
    Compute rigorous medical classification metrics:
    accuracy, balanced accuracy, macro/weighted precision/recall/F1,
    per-class metrics, confusion matrix, and one-vs-rest ROC-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # ROC-AUC (One-vs-Rest)
    roc_auc_macro = None
    roc_auc_ovr = {}
    if y_prob is not None:
        try:
            # Check if all classes present in y_true
            unique_classes = np.unique(y_true)
            if len(unique_classes) == len(class_names):
                roc_auc_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                for i, cname in enumerate(class_names):
                    y_true_binary = (y_true == i).astype(int)
                    roc_auc_ovr[f"{cname.lower()}_roc_auc"] = float(roc_auc_score(y_true_binary, y_prob[:, i]))
        except Exception:
            roc_auc_macro = None

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "support": {cname: int(sup) for cname, sup in zip(class_names, support)},
        "per_class": {
            cname: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "support": int(sup),
            }
            for cname, p, r, f1, sup in zip(class_names, per_class_p, per_class_r, per_class_f1, support)
        },
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_ovr": roc_auc_ovr,
    }
    return metrics

def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    class_names: List[str] = CLASS_NAMES,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Condition")
    plt.ylabel("Ground Truth Condition")
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
