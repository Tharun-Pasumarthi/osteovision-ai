# Class Mapping & Output Activation Audit

## 1. Class Taxonomy & Canonical Index Mapping

The system classifies knee radiographs into three mutually exclusive diagnostic categories:
1. **Normal**: Calcaneal QUS T-Score proxy $T \ge -1.0$ (or healthy radiographic reference).
2. **Osteopenia**: Low bone mass proxy $-2.5 < T < -1.0$.
3. **Osteoporosis**: Severe bone mineral density depletion proxy $T \le -2.5$.

### Index Alignment Across All Subsystems
| Component | Class 0 | Class 1 | Class 2 | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| `src/train.py` (`CLASS_TO_IDX`) | `"Normal": 0` | `"Osteopenia": 1` | `"Osteoporosis": 2` | **Consistent** |
| `src/evaluate.py` (`CLASS_NAMES`) | `"Normal"` | `"Osteopenia"` | `"Osteoporosis"` | **Consistent** |
| `backend/app/services/inference_service.py` | `"Normal"` (idx 0) | `"Osteopenia"` (idx 1) | `"Osteoporosis"` (idx 2) | **Consistent** |
| `backend/app/services/gradcam_service.py` | `"Normal"` (idx 0) | `"Osteopenia"` (idx 1) | `"Osteoporosis"` (idx 2) | **Consistent** |
| `frontend/src/types/api.ts` | `"Normal"` | `"Osteopenia"` | `"Osteoporosis"` | **Consistent** |

No internal index inversions were found between training targets and model output indices.

---

## 2. Output Activation and Probability Conversion

### Forward Pass & Logits
- Neural network classifier heads terminate in an unconstrained linear projection:
  $$\mathbf{z} = [z_0, z_1, z_2] \in \mathbb{R}^3$$
- Training loss uses standard PyTorch `nn.CrossEntropyLoss(weight=class_weights)`, which applies $\log\text{Softmax}$ internally.

### Inference Conversion
- Softmax activation is applied to convert logits into empirical probabilities:
  $$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=0}^2 e^{z_j}}$$
- Predicted class is resolved via argmax:
  $$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} P(y = c \mid \mathbf{x})$$

### Calibration Note
- Empirical softmax scores from standard deep networks are often overconfident and uncalibrated under domain shift.
- Temperature scaling ($T > 0$):
  $$P_{\text{cal}}(y = c \mid \mathbf{x}) = \frac{e^{z_c / T}}{\sum_{j=0}^2 e^{z_j / T}}$$
  will be fitted exclusively on validation logits to prevent exaggerated confidence on ambiguous borderline cases.

---

## 3. Discovered Source-Level Label Inversions

While programmatic index mappings are consistent, **source dataset labels contain critical inversions**:
1. In Kaggle (Gobara), identical byte-for-byte radiographs are present in both the `Normal` and `Osteoporosis` directory trees (e.g. `Normal 104.JPEG` $\leftrightarrow$ `Osteoporosis 120.JPEG`).
2. In Mendeley, `N23.jpg` (labeled Normal, patient N23) is identical in image hash to `OP107.jpg` (labeled Osteopenia, patient OP107).
3. These contradictions represent ground-truth label pollution in public repositories. They must be quarantined and excluded from all clean partitions.
