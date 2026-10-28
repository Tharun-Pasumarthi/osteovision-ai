# Preprocessing Audit & Technical Verification

## 1. Preprocessing Specification & Audit

| Property | Training-Time Preprocessing (`KneeXRayTrainTransform`) | Inference-Time Preprocessing (`KneeXRayEvalTransform`) | API / Service Preprocessing (`inference_service.py`) | Status & Integrity |
| :--- | :--- | :--- | :--- | :--- |
| **Input Color Mode** | Converts any format (`L`, `RGBA`, `CMYK`) to `RGB` | Converts to `RGB` | Converts to `RGB` | **Identical** |
| **Geometry Handling** | Letterbox pad to square preserving aspect ratio (fill=0) | Letterbox pad to square preserving aspect ratio (fill=0) | Letterbox pad to square preserving aspect ratio (fill=0) | **Identical** |
| **Target Size** | $224 \times 224$ pixels | $224 \times 224$ pixels | $224 \times 224$ pixels | **Identical** |
| **Augmentation** | Conservative: Rotation ($\pm 10^\circ$), affine shift ($\pm 5\%$), scale (0.95–1.05), brightness/contrast jitter ($\pm 10\%$) | Deterministic (no stochastic transforms) | Deterministic (no stochastic transforms) | **Appropriate** |
| **Normalization** | ImageNet Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]` | ImageNet Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]` | ImageNet Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]` | **Identical** |
| **Tensor Dimensions** | `(B, 3, 224, 224)` float32 | `(1, 3, 224, 224)` float32 | `(1, 3, 224, 224)` float32 | **Identical** |

---

## 2. Identified Preprocessing Vulnerabilities & Discoveries

### A. Pre-Letterboxed Pipeline Artifacts
In the previous synthetic pipeline (`scripts/generate_balanced_dataset.py`), authentic images were pre-processed and written to disk as pre-letterboxed $224 \times 224$ PNG files (`data/balanced_cohort_2400`). When loaded into `KneeXRayTrainTransform`, this resulted in double-padding operations and fixed border artifacts.
- **Remedy**: Training pipelines must always load original native-resolution radiographs and apply on-the-fly letterbox padding directly.

### B. Aspect Ratio & Scale Discrepancy
- Mendeley images are square or near-square ($1024 \times 1024$ or $2180 \times 2660$) showing bilateral knees.
- Kaggle images are narrow rectangles ($525 \times 1024$ or $1128 \times 2660$) showing single knees with heavy black padding added during letterboxing ($525 \to 1024$ requires 250px black bars on left and right).
- On a 224x224 grid, the effective knee joint area of a unilateral radiograph is $115 \times 224$, whereas for a bilateral radiograph it is $224 \times 224$ containing two joints. The spatial frequencies of trabeculae are thus fundamentally different.

### C. Conservative Augmentation Protocol
Medical imaging requires anatomically plausible augmentations:
- **Vertical flips**: Strictly prohibited (`allow_vertical_flip: false`). Inverted gravity or upside-down femurs destroy anatomical orientation.
- **Horizontal flips**: Disabled by default to preserve laterality markers (`allow_horizontal_flip: false`).
- **Elastic deformations / Shears**: Avoided to prevent distorting joint space width and cortical boundaries.
