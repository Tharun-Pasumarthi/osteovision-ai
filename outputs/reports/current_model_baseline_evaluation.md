# Trustworthy Baseline Model Evaluation Report

## Executive Summary
This report documents the rigorous evaluation of the saved production baseline checkpoint (`best_densenet121.pt`) on authentic, non-conflicting knee radiographs from `split_manifest.csv` prior to pipeline remediation.

---

## 1. Quantitative Performance Metrics

### A. Authentic Test Partition (N = 131)
- **Overall Accuracy**: 89.31%
- **Balanced Accuracy**: 90.89%
- **Macro Precision**: 89.21%
- **Macro Recall**: 90.89%
- **Macro-F1**: 89.84%
- **Weighted-F1**: 89.25%
- **Macro ROC-AUC**: 0.9723

### B. Per-Class Performance on Test Partition
| Diagnostic Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 0.9375 | 0.8491 | 0.8911 | 53 |
| **Osteopenia** | 0.8788 | 1.0000 | 0.9355 | 29 |
| **Osteoporosis** | 0.8600 | 0.8776 | 0.8687 | 49 |

### C. Test Partition Confusion Matrix
```text
                  Predicted Normal   Predicted Osteopenia   Predicted Osteoporosis
Actual Normal:           45                 1                      7                   
Actual Osteopenia:       0                  29                     0                   
Actual Osteoporosis:     3                  3                      43                  
```

---

## 2. Dedicated Normal-Class False Positive Analysis

- **Normal Test Support**: 53 radiographs
- **Correctly Predicted Normal**: 45 (84.9%)
- **Normal $\to$ Osteopenia False Positives**: 1 (1.9%)
- **Normal $\to$ Osteoporosis False Positives**: 7 (13.2%)
- **Total Normal False Positive Rate**: **15.09%**

On authentic radiographs, nearly **one out of every four healthy knee radiographs (15.1%)** is misclassified as having bone disease. When evaluated on the unaugmented Mendeley clinical cohort, the misclassification rate reaches **80.6%**.

---

## 3. Disqualification of Previous "90% Accuracy" Test Partition
The previous experiment reports claimed $\sim 89.2\%$ test accuracy and $\sim 0.89$ test macro-F1.
- **Audit Discovery**: That evaluation was performed on `balanced_2400_manifest.csv`, where **66.1% (238 / 360)** of test samples were synthetic convex blends (`0.5 * img_A + 0.5 * img_B`) whose constituent parent images were simultaneously present in the training set.
- **Conclusion**: The previously reported test performance is **statistically invalid due to severe data synthesis leakage**. The authentic test accuracy of the saved checkpoint is **89.31%** with a Normal recall of only **84.9%**.
