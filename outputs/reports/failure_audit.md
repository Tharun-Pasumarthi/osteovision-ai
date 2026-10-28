# Comprehensive Model & Pipeline Failure Audit

## Executive Summary
This audit investigates the major failure modes identified in the Knee X-Ray Osteoporosis Classification system:
1. **Clinical Prediction Failure**: Healthy-looking normal knee radiographs are systematically misclassified as **Osteopenia** or **Osteoporosis** (e.g., 80.6% misclassification on authentic Mendeley clinical examinations).
2. **Grad-CAM Visualization Failures**: Activation heatmaps are unreliable, accumulate hook memory leaks, target suboptimal layers (e.g. `nn.Sequential` for ResNet50, `BatchNorm2d` for DenseNet121), distort aspect ratios, and lack proper cleanup and gradient verification.

---

## 1. Empirical Reproduction of the Prediction Failure

Before modifying any pipeline code, the saved checkpoint (`best_densenet121.pt`) was evaluated against authentic radiograph cohorts using exact inference transforms (`KneeXRayEvalTransform(224)`):

### A. Reproduction on Authentic Mendeley Ground-Truth Cohort (`normal/`)
- **Total Tested**: 36 verified Normal knee radiographs from Mendeley Data (fxjm8fb6mw/2) where ground-truth diagnosis is documented via calcaneal QUS T-scores ($T > -1.0$).
- **Predicted as Normal**: 7 / 36 (**19.4%**)
- **Predicted as Osteopenia (False Positive)**: 18 / 36 (**50.0%**)
- **Predicted as Osteoporosis (False Positive)**: 11 / 36 (**30.6%**)
- **Total False-Positive Rate for Normal**: **80.6%**

### B. Individual Example Predictions
| Image File | Source Dataset | Documented True Label | Model Prediction | Softmax Probs (Normal / Osteopenia / Osteoporosis) | Failure Mode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `N1.JPEG` | Mendeley | Normal | **Osteopenia** | `[0.043, 0.918, 0.039]` | Severe false positive ($p=0.918$) |
| `N10.JPEG` | Mendeley | Normal | **Osteopenia** | `[0.309, 0.614, 0.077]` | False positive |
| `N14.jpg` | Mendeley | Normal | **Osteopenia** | `[0.019, 0.954, 0.027]` | Extreme false positive ($p=0.954$) |
| `N16.jpg` | Mendeley | Normal | **Osteopenia** | `[0.031, 0.936, 0.033]` | Extreme false positive ($p=0.936$) |
| `Normal 1.png` | Kaggle | Normal | **Osteoporosis** | `[0.081, 0.000, 0.919]` | Severe false positive ($p=0.919$) |
| `Normal 104.JPEG` | Kaggle | Normal | **Osteoporosis** | `[0.215, 0.020, 0.766]` | Label conflict in source dataset |

---

## 2. Identified Root Causes

### Root Cause 1: Contaminated "Balanced 2,400" Synthetic Data Pipeline
- The previous pipeline utilized `scripts/generate_balanced_dataset.py` to create `data/balanced_cohort_2400` with 800 images per class.
- **64.0% (1,535 / 2,400)** of all images in this cohort were generated via **intra-class convex manifold blending** (`0.5 * img_A + 0.5 * img_B`).
- Blending two distinct radiographs creates artificial double-exposure artifacts, diffused cortical margins, and washed-out trabecular textures. 
- In radiography, loss of sharp trabecular microarchitecture is the cardinal sign of Osteopenia and Osteoporosis. Consequently, the network learned that hazy, blended textures correlate with the target classes.
- Shuffling authentic and synthetic images across splits caused catastrophic test-set contamination: derivatives of training patients were present in the test set.

### Root Cause 2: Label Inversion and Conflict in Source Datasets
- Audit of exact file SHA-256 hashes identified **158 duplicate hash groups** (704 files) with contradictory diagnosis labels.
- In Kaggle (OS Collected Data), 79 image groups had direct internal label conflicts:
  - `Normal 104.JPEG` is identical in byte content to `Osteoporosis 120.JPEG` and `Osteoporosis 211.JPEG`.
  - `Normal 201.jpg` is identical in byte content to `Osteopenia 230.jpg`.
  - `Osteopenia 262.jpg` is identical in byte content to `Osteoporosis 347.jpg`.
- When training without strict deduplication and conflict quarantine, identical features were paired with contradictory gradient targets, destabilizing class boundaries.

### Root Cause 3: Bilateral vs. Unilateral Knee Scale Shift
- **Mendeley Dataset**: Full bilateral knee radiographs ($1024 \times 1024$ or $2180 \times 2660$) showing both knees in a single exposure.
- **Kaggle Dataset**: Unilateral crops ($525 \times 1024$ or $1128 \times 2660$) showing only a single knee joint.
- When letterbox padding bilateral radiographs into $224 \times 224$, each knee occupies only $\sim 50\%$ the area of a unilateral knee. The effective magnification and cortical thickness scales differ by a factor of 2, creating an uncalibrated domain shift.

### Root Cause 4: Overfitting on Stage-2 Fine-Tuning
- The saved model was fine-tuned for only 2 epochs on the synthetic dataset, optimizing validation macro-F1 on contaminated synthetic test samples.
- Softmax output probabilities were uncalibrated and overconfident, yielding $>0.90$ confidence on false-positive misclassifications.

---

## 3. Grad-CAM Root Cause Analysis

1. **Accumulating PyTorch Hooks**:
   - `GradCAM` registered forward and backward hooks inside `__init__` without returning removable handles or calling `handle.remove()`.
   - On every API call in `GradCAMService.generate_explanation`, a new `GradCAM` instance was created on the globally cached model, adding 2 new permanent hooks per request. This caused memory leaks and backward execution errors.
2. **Suboptimal Target Layers**:
   - ResNet50: Targeted `model.features[-1]` (`nn.Sequential` container `layer4`) rather than the final convolutional layer `layer4[-1].conv3`.
   - DenseNet121: Targeted `model.features.norm5` (`BatchNorm2d`) rather than the final dense block conv features before pooling.
3. **Heatmap Distortion and Padding Inconsistency**:
   - The original image was padded to $224 \times 224$, the heatmap resized to $224 \times 224$, and the blended result returned without projecting back to the patient's native image aspect ratio.
4. **Unsafe Normalization on Flat Gradients**:
   - Near-zero gradients resulted in zero-division risks and blank heatmaps without informative error signaling.

---

## 4. Remediation Plan
1. **Purge Synthetic Blends**: Retire `balanced_cohort_2400` completely from evaluation and training. Train strictly on authentic radiographs.
2. **Quarantine Conflicted Hashes**: Quarantine all 158 conflict groups.
3. **Patient-Isolated Clean Partitions**: Reconstruct clean train/val/test splits grouping all images from the same patient and deduplicating identical hashes.
4. **Refactor Grad-CAM**: Use context-managed hooks with explicit cleanup, verified target conv layers, native dimension projection, and proper zero-gradient guards.
5. **Calibrate Probabilities & Add Review State**: Implement validation-based temperature scaling and confidence thresholding for clinical decision support.
