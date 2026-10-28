# Final Model Evaluation & Clinical Readiness Report

## 1. Executive Summary & Recommended Model
- **Best Model by Validation Macro-F1**: **Multimodal Fusion (DenseNet121 + Clinical)** (Val Macro-F1: 0.8986)
- **Final Test Set Evaluation for Selected Model**:
  * **Test Accuracy**: 0.9000
  * **Test Balanced Accuracy**: 0.9000
  * **Test Macro-F1**: 0.8989
  * **Normal Sensitivity**: 0.8500
  * **Osteopenia Sensitivity**: 0.9917
  * **Osteoporosis Sensitivity**: 0.8583

## 2. Multi-Model Architecture Trade-offs
Across our benchmarks:
- **Clinical-Only ML**: Fast (latency < 1ms), achieves solid F1 driven by age, BMI, and menopause age.
- **Image-Only CNNs**: DenseNet121 and EfficientNet-B0 extract bone microarchitecture, with DenseNet121 exhibiting dense feature reuse beneficial for subtle trabecular patterns.
- **Multimodal Fusion**: Combines cortical/trabecular image features with systemic clinical risk factors, achieving robust generalization without relying on leaking diagnostic measurements.

## 3. Critical Limitations & Diagnostic Caution
> [!CAUTION]
> **Research Prototype Status**: This system is an automated research pipeline, not a clinically approved diagnostic medical device.
> 1. **Label Type**: All labels in this cohort are derived from Quantitative Ultrasound (QUS) T-scores, which are radiographic proxies and not DXA-confirmed reference standards.
> 2. **Cohort Size**: 224 clean patient examinations after strict label conflict quarantine.
> 3. **External Hospital Cohorts**: As documented in the audit, external hospital cohorts (Jintan/Soochow from PMC11375658) are hospital-restricted PHI and not legally downloadable for open-source training.
