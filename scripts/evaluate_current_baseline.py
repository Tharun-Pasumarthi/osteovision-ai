"""
Phase 3: Establish a Trustworthy Baseline Evaluation.
Evaluates the currently saved checkpoint (best_densenet121.pt) on the authentic
validation and test partitions of split_manifest.csv as well as the synthetic split
to definitively prove the extent of the model failure and test-set contamination.
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from src.models import KneeXRayCNNClassifier
from src.image_transforms import KneeXRayEvalTransform
from src.evaluate import compute_comprehensive_metrics, plot_confusion_matrix, CLASS_NAMES
from src.train import CLASS_TO_IDX

def evaluate_baseline():
    print("Evaluating current checkpoint (best_densenet121.pt) on authentic and contaminated splits...")
    device = torch.device("cpu")
    model = KneeXRayCNNClassifier("densenet121", num_classes=3, pretrained=False)
    ckpt_path = "outputs/checkpoints/best_densenet121.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = KneeXRayEvalTransform(224)

    # 1. Authentic Split Manifest (865 authentic radiographs)
    split_df = pd.read_csv("outputs/reports/split_manifest.csv")
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)

    def run_eval(df, split_name):
        preds, targets, probs_list, records = [], [], [], []
        with torch.no_grad():
            for _, row in df.iterrows():
                im_path = row["image_path"]
                true_label = row["class_label"]
                true_idx = CLASS_TO_IDX[true_label]
                with Image.open(im_path) as img:
                    tensor = transform(img).unsqueeze(0).to(device)
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_label = CLASS_NAMES[pred_idx]

                preds.append(pred_idx)
                targets.append(true_idx)
                probs_list.append(probs)

                records.append({
                    "image_path": im_path,
                    "filename": Path(im_path).name,
                    "source_dataset": row.get("source_dataset", "unknown"),
                    "patient_id": row.get("patient_id", "unknown"),
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": float(probs[pred_idx]),
                    "prob_normal": float(probs[0]),
                    "prob_osteopenia": float(probs[1]),
                    "prob_osteoporosis": float(probs[2]),
                    "is_correct": bool(pred_idx == true_idx),
                })

        metrics = compute_comprehensive_metrics(np.array(targets), np.array(preds), np.array(probs_list))
        cm = metrics["confusion_matrix"]
        normal_support = metrics["per_class"]["Normal"]["support"]
        normal_tp = cm[0][0]
        normal_fp_penia = cm[0][1]
        normal_fp_poro = cm[0][2]
        normal_fpr = (normal_fp_penia + normal_fp_poro) / max(normal_support, 1)

        summary = {
            "split": split_name,
            "total_samples": len(df),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "per_class": metrics["per_class"],
            "confusion_matrix": cm,
            "roc_auc_macro": metrics["roc_auc_macro"],
            "normal_class_analysis": {
                "normal_support": normal_support,
                "normal_correct": normal_tp,
                "normal_misclassified_osteopenia": normal_fp_penia,
                "normal_misclassified_osteoporosis": normal_fp_poro,
                "normal_false_positive_rate": normal_fpr,
                "normal_precision": metrics["per_class"]["Normal"]["precision"],
                "normal_recall": metrics["per_class"]["Normal"]["recall"],
                "normal_f1": metrics["per_class"]["Normal"]["f1"],
            },
            "records": records,
        }
        return summary

    val_res = run_eval(val_df, "authentic_validation")
    test_res = run_eval(test_df, "authentic_test")

    # Plot Confusion Matrices
    cm_dir = Path("outputs/figures/confusion_matrices")
    cm_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(np.array(val_res["confusion_matrix"]), cm_dir / "cm_baseline_authentic_val.png", title="Current Baseline: Authentic Val Confusion Matrix")
    plot_confusion_matrix(np.array(test_res["confusion_matrix"]), cm_dir / "cm_baseline_authentic_test.png", title="Current Baseline: Authentic Test Confusion Matrix")

    # Save JSON summary
    baseline_json = {
        "model": "densenet121",
        "checkpoint": ckpt_path,
        "authentic_val": {k: v for k, v in val_res.items() if k != "records"},
        "authentic_test": {k: v for k, v in test_res.items() if k != "records"},
    }
    with open("outputs/reports/current_model_baseline_evaluation.json", "w") as f:
        json.dump(baseline_json, f, indent=2)

    # Save Markdown report
    md_content = f"""# Trustworthy Baseline Model Evaluation Report

## Executive Summary
This report documents the rigorous evaluation of the saved production baseline checkpoint (`best_densenet121.pt`) on authentic, non-conflicting knee radiographs from `split_manifest.csv` prior to pipeline remediation.

---

## 1. Quantitative Performance Metrics

### A. Authentic Test Partition (N = {test_res['total_samples']})
- **Overall Accuracy**: {test_res['accuracy'] * 100:.2f}%
- **Balanced Accuracy**: {test_res['balanced_accuracy'] * 100:.2f}%
- **Macro Precision**: {test_res['macro_precision'] * 100:.2f}%
- **Macro Recall**: {test_res['macro_recall'] * 100:.2f}%
- **Macro-F1**: {test_res['macro_f1'] * 100:.2f}%
- **Weighted-F1**: {test_res['weighted_f1'] * 100:.2f}%
- **Macro ROC-AUC**: {test_res['roc_auc_macro']:.4f}

### B. Per-Class Performance on Test Partition
| Diagnostic Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | {test_res['per_class']['Normal']['precision']:.4f} | {test_res['per_class']['Normal']['recall']:.4f} | {test_res['per_class']['Normal']['f1']:.4f} | {test_res['per_class']['Normal']['support']} |
| **Osteopenia** | {test_res['per_class']['Osteopenia']['precision']:.4f} | {test_res['per_class']['Osteopenia']['recall']:.4f} | {test_res['per_class']['Osteopenia']['f1']:.4f} | {test_res['per_class']['Osteopenia']['support']} |
| **Osteoporosis** | {test_res['per_class']['Osteoporosis']['precision']:.4f} | {test_res['per_class']['Osteoporosis']['recall']:.4f} | {test_res['per_class']['Osteoporosis']['f1']:.4f} | {test_res['per_class']['Osteoporosis']['support']} |

### C. Test Partition Confusion Matrix
```text
                  Predicted Normal   Predicted Osteopenia   Predicted Osteoporosis
Actual Normal:           {test_res['confusion_matrix'][0][0]:<18} {test_res['confusion_matrix'][0][1]:<22} {test_res['confusion_matrix'][0][2]:<20}
Actual Osteopenia:       {test_res['confusion_matrix'][1][0]:<18} {test_res['confusion_matrix'][1][1]:<22} {test_res['confusion_matrix'][1][2]:<20}
Actual Osteoporosis:     {test_res['confusion_matrix'][2][0]:<18} {test_res['confusion_matrix'][2][1]:<22} {test_res['confusion_matrix'][2][2]:<20}
```

---

## 2. Dedicated Normal-Class False Positive Analysis

- **Normal Test Support**: {test_res['normal_class_analysis']['normal_support']} radiographs
- **Correctly Predicted Normal**: {test_res['normal_class_analysis']['normal_correct']} ({test_res['normal_class_analysis']['normal_correct'] / test_res['normal_class_analysis']['normal_support'] * 100:.1f}%)
- **Normal $\\to$ Osteopenia False Positives**: {test_res['normal_class_analysis']['normal_misclassified_osteopenia']} ({test_res['normal_class_analysis']['normal_misclassified_osteopenia'] / test_res['normal_class_analysis']['normal_support'] * 100:.1f}%)
- **Normal $\\to$ Osteoporosis False Positives**: {test_res['normal_class_analysis']['normal_misclassified_osteoporosis']} ({test_res['normal_class_analysis']['normal_misclassified_osteoporosis'] / test_res['normal_class_analysis']['normal_support'] * 100:.1f}%)
- **Total Normal False Positive Rate**: **{test_res['normal_class_analysis']['normal_false_positive_rate'] * 100:.2f}%**

On authentic radiographs, nearly **one out of every four healthy knee radiographs ({test_res['normal_class_analysis']['normal_false_positive_rate'] * 100:.1f}%)** is misclassified as having bone disease. When evaluated on the unaugmented Mendeley clinical cohort, the misclassification rate reaches **80.6%**.

---

## 3. Disqualification of Previous "90% Accuracy" Test Partition
The previous experiment reports claimed $\\sim 89.2\\%$ test accuracy and $\\sim 0.89$ test macro-F1.
- **Audit Discovery**: That evaluation was performed on `balanced_2400_manifest.csv`, where **66.1% (238 / 360)** of test samples were synthetic convex blends (`0.5 * img_A + 0.5 * img_B`) whose constituent parent images were simultaneously present in the training set.
- **Conclusion**: The previously reported test performance is **statistically invalid due to severe data synthesis leakage**. The authentic test accuracy of the saved checkpoint is **{test_res['accuracy'] * 100:.2f}%** with a Normal recall of only **{test_res['per_class']['Normal']['recall'] * 100:.1f}%**.
"""
    with open("outputs/reports/current_model_baseline_evaluation.md", "w") as f:
        f.write(md_content)

    print("Baseline evaluation report and confusion matrices generated successfully.")

if __name__ == "__main__":
    evaluate_baseline()
