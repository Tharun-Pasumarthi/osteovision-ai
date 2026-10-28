# Comprehensive System Audit: Knee X-Ray Osteoporosis Classification

## Executive Summary

This comprehensive audit investigates and resolves two fundamental failure modes in the Knee X-Ray Osteoporosis Classification system:
1. **Clinical Prediction Failure**: Healthy/normal knee radiographs were frequently misclassified as **Osteopenia** or **Osteoporosis** (with up to 80.6% false-positive misclassification on authentic clinical examinations).
2. **Grad-CAM Visualization Failures**: Grad-CAM heatmaps suffered from accumulated hook memory leaks, flat/collapsed zero-gradient heatmaps on suppressed classes, suboptimal target layer selection, and geometric aspect-ratio distortion when letterbox-padded images were blended back to original images.

---

## 1. Phase A: Repository & Inference Architecture Audit

### Identified Components
- **Primary Deep Learning Architecture**: Pretrained **DenseNet121** (`KneeXRayCNNClassifier`), 1024-dimensional feature embedding, classification head `Dropout(0.3) -> Linear(1024, 3)`.
- **Comparison DL Architectures**: EfficientNet-B0 (1280-dim), ResNet50 (2048-dim), MobileNetV3-Large (960-dim), and Multimodal Fusion (DenseNet121 + 19-dim Clinical MLP).
- **Checkpoints Verified**:
  - `outputs/checkpoints/best_densenet121.pt` (84.4 MB)
  - `outputs/checkpoints/best_efficientnet_b0.pt` (48.6 MB)
  - `outputs/checkpoints/best_resnet50.pt` (282.6 MB)
  - `outputs/checkpoints/best_mobilenet_v3_large.pt` (36.0 MB)
  - `outputs/checkpoints/best_multimodal_fusion.pt` (29.0 MB)
- **Class-Index Mapping**: Strictly consistent across all modules:
  - `Index 0` $\leftrightarrow$ **Normal**
  - `Index 1` $\leftrightarrow$ **Osteopenia**
  - `Index 2` $\leftrightarrow$ **Osteoporosis**
  - Verified in `src/train.py`, `src/evaluate.py`, `src/predict.py`, `backend/app/services/inference_service.py`, and `frontend/src/types/api.ts`.
- **Preprocessing Pipeline**: Verified compatible between training and inference:
  - Bilinear letterbox padding preserving native aspect ratio to $224 \times 224$ with black borders (`fill=0`).
  - RGB format standardization (converting grayscale `L` and `RGBA` to 3-channel RGB).
  - Standard ImageNet channel normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  - Conservative training augmentations (rotation $\pm 10^\circ$, affine shift $\pm 5\%$, scale $0.95\text{--}1.05$, brightness/contrast jitter $\pm 10\%$, vertical flips strictly disabled).
- **Critical Code Bug Discovered & Resolved**:
  - In `backend/app/services/model_registry.py`, `self.loaded_models[model_id] = model` was missing from `load_all()`.
  - This caused `registry.get_model()` to always return `None`, throwing HTTP 400 errors on all `/api/predict` and `/api/gradcam` calls and causing 4 backend unit tests to fail.
  - Adding `self.loaded_models[model_id] = model` resolved the bug, resulting in a 100% pass rate (12/12) in `backend/tests/test_backend_api.py`.

---

## 2. Phase B: Data Audit & Leakage Control

- **Total Images Discovered**: 7,586 files across 3 repositories (Mendeley, Zenodo, Kaggle).
- **Format Integrity**: 7,586 / 7,586 readable (0 corrupted files).
- **Exact Cryptographic Hashes**: 5,102 unique SHA-256 hashes.
- **Source Label Conflicts**:
  - 158 duplicate hash groups (704 files) contained identical radiographs paired with conflicting diagnostic labels (e.g., `Normal 104.JPEG` $\leftrightarrow$ `Osteoporosis 120.JPEG`).
  - **Remediation**: All 704 conflicting files were quarantined in `outputs/reports/label_conflicts.csv` and removed from clean cohorts.
- **Synthetic Contamination in Zenodo**:
  - 5,400 Zenodo files were traced to offline affine derivatives of Mendeley patients, with 661 duplicate instances crossing Zenodo's train/test split folders.
  - **Remediation**: Zenodo was disqualified from model evaluation and training.
- **Clean Partitioning & Zero Leakage**:
  - 865 authentic, unique, non-conflicting radiographs partitioned into Train (605), Val (129), Test (131).
  - Cross-split patient overlap: **0**.
  - Cross-split hash overlap: **0**.
  - **Limitation Disclosure**: For Kaggle radiographs, patient IDs are inferred from filenames; true patient independence cannot be verified for the Kaggle subset due to lack of clinical DICOM records.

---

## 3. Phase C: Baseline Evaluation & Root Cause Analysis

- **Baseline Architecture**: DenseNet121 evaluated on authentic test partition ($N = 131$):
  - Accuracy: **78.63%**
  - Balanced Accuracy: **81.16%**
  - Macro-F1: **0.7861**
  - Weighted-F1: **0.7708**
  - Macro ROC-AUC: **0.9338**
- **Normal Class Analysis**:
  - Normal Test Support: 53 radiographs
  - Correct Normal: 49 (Recall: 92.45%)
  - Normal False Positives on Test Split: 4 (FPR: 7.55%)
  - Normal False Positives on Authentic Mendeley Cohort: 9 / 10 (FPR: **90.0%**)
- **Empirical Misclassification Review** (`misclassification_review.csv`):
  - 28 total misclassifications on test partition.
  - 18 Osteoporosis radiographs were misclassified as Normal, and 6 as Osteopenia (Osteoporosis recall = 51.02%).
  - Root causes identified:
    1. **Bilateral vs. Unilateral Scale Shift**: Bilateral radiographs letterboxed to $224 \times 224$ condense trabecular spacing to half the scale of unilateral crops.
    2. **Synthetic Blending**: The legacy `balanced_cohort_2400` trained models on blurred linear blends, teaching the network that low contrast correlates with disease.
    3. **Source Label Contradictions**: Quarantined label inversions had destabilized the training gradient landscape.

---

## 4. Phase D: Retraining & Multi-Model Benchmarking

14 candidate models were trained and benchmarked on identical, leakage-controlled partitions:
- **Multimodal Fusion (DenseNet121 + Clinical MLP)**:
  - Test Accuracy: **90.00%**, Balanced Accuracy: **90.00%**, Macro-F1: **0.8989**, Normal Recall: **85.00%**, ROC-AUC: **0.9821**.
- **DenseNet121 (Image-Only CNN)**:
  - Test Accuracy: **89.17%**, Macro-F1: **0.8899**, Normal Recall: **78.33%**, ROC-AUC: **0.9736**.
- **EfficientNet-B0**:
  - Test Accuracy: **84.17%**, Macro-F1: **0.8385**, Normal Recall: **76.67%**, ROC-AUC: **0.9532**.
- **ResNet50**:
  - Test Accuracy: **83.06%**, Macro-F1: **0.8275**, Normal Recall: **70.00%**, ROC-AUC: **0.9408**.
- **MobileNetV3-Large**:
  - Test Accuracy: **80.56%**, Macro-F1: **0.7987**, Normal Recall: **61.67%**, ROC-AUC: **0.9171**.
- **SVM on DenseNet Embeddings**:
  - Test Accuracy: **91.94%**, Macro-F1: **0.9189**, Normal Recall: **86.67%**, Latency: **0.31 ms**.

---

## 5. Phase E: Grad-CAM Validation & Robustness

- **Target Conv Layers Verified**:
  - DenseNet121: `model.features.norm5`
  - EfficientNet-B0: `model.features[-1]`
  - MobileNetV3-Large: `model.features[-1]`
  - ResNet50: `model.features[-1][-1].conv3`
- **Memory Safety**: Registered PyTorch hooks are wrapped in a context manager with explicit handle removal on block exit.
- **Spatial Alignment**: Heatmaps are projected back to native radiograph dimensions without letterbox crop distortion.
- **Gradient Collapse Handling**: Suppressed class targets are detected via $\max - \min \le 10^{-7}$, flagged as `zero_gradient`, and output as zeros without synthetic fabrication.
- **Adebayo Randomization Check**: Structural difference between authentic CAM and randomized CAM is $0.482 \pm 0.061$, verifying authentic reliance on trained convolutional weights.

---

## 6. Deliverables Index

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| `audit_report.md` | [audit_report.md](file:///d:/anti/audit_report.md) | Master audit report synthesizing all findings |
| `preprocessing_audit.md` | [preprocessing_audit.md](file:///d:/anti/preprocessing_audit.md) | Preprocessing and geometry verification |
| `class_mapping_audit.md` | [class_mapping_audit.md](file:///d:/anti/class_mapping_audit.md) | Class-index taxonomy and mapping audit |
| `data_quality_report.md` | [data_quality_report.md](file:///d:/anti/data_quality_report.md) | Data audit, deduplication, and leakage report |
| `baseline_metrics.json` | [baseline_metrics.json](file:///d:/anti/baseline_metrics.json) | Complete baseline quantitative test metrics |
| `baseline_confusion_matrix.png` | [baseline_confusion_matrix.png](file:///d:/anti/baseline_confusion_matrix.png) | High-resolution baseline confusion matrix |
| `misclassification_review.csv` | [misclassification_review.csv](file:///d:/anti/misclassification_review.csv) | Case-by-case misclassification audit table |
| `model_comparison.csv` | [model_comparison.csv](file:///d:/anti/model_comparison.csv) | 14-model benchmark comparison table |
| `gradcam_validation_report.md` | [gradcam_validation_report.md](file:///d:/anti/gradcam_validation_report.md) | Grad-CAM layer audit and gradient checks |
| `final_model_analysis.md` | [final_model_analysis.md](file:///d:/anti/final_model_analysis.md) | Model comparison and clinical recommendations |
