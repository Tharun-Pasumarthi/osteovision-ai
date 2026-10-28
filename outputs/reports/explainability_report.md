# Model Explainability & Visual Attribution Report

## 1. Grad-CAM Radiograph Visualizations

Grad-CAM was applied to the final normalization layer (`features.norm5`) of the DenseNet121 image encoder. Heatmap gradients highlight regions of high activation responsible for predicted class logits.

- **Total Test Images Analyzed**: 131
- **Correct Predictions Visualized**: 105
- **Misclassified Cases Visualized**: 26

### Anatomical Alignment vs Artifact Checks
- **Bone Region Focus**: High activation clusters are predominantly centered over the distal femur, proximal tibia, and subchondral trabecular bone plate.
- **Acquisition Artifact Check**: Verified that heatmaps do not concentrate on periphery borders, text orientation labels ('L'/'R' markers), or background black padding.

### Sample Test Case Attributions

| patient_id   | true_label   | predicted_label   |   confidence | is_correct   |
|:-------------|:-------------|:------------------|-------------:|:-------------|
| OP79         | Osteopenia   | Osteopenia        |     0.62424  | True         |
| OP75         | Osteopenia   | Osteopenia        |     0.815806 | True         |
| OP74         | Osteopenia   | Osteopenia        |     0.736319 | True         |
| OP63         | Osteopenia   | Osteopenia        |     0.651371 | True         |
| OP59         | Osteopenia   | Osteopenia        |     0.810293 | True         |
| OP60         | Osteopenia   | Osteopenia        |     0.822557 | True         |
| OP71         | Osteopenia   | Osteopenia        |     0.828505 | True         |
| OP66         | Osteopenia   | Osteopenia        |     0.694429 | True         |
| OP67         | Osteopenia   | Osteopenia        |     0.945217 | True         |
| OP93         | Osteopenia   | Osteopenia        |     0.718741 | True         |

## 2. Clinical Feature Importance (Permutation Importance)

To explain the clinical branch independently without leakage, permutation feature importance was computed on validation data across 10 random permutations.

| feature_name                |   importance_mean |   importance_std |
|:----------------------------|------------------:|-----------------:|
| height_m                    |       0.246629    |       0.035245   |
| gender_female               |       0.0491724   |       0.0105669  |
| diabetic                    |       0           |       0          |
| smoker                      |       0           |       0          |
| obesity_obesity             |       0           |       0          |
| obesity_overweight          |       0           |       0          |
| obesity_normal weight       |       0           |       0          |
| obesity_under weight        |       0           |       0          |
| joint_pain                  |       0           |       0          |
| bmi                         |      -0.000630228 |       0.00725415 |
| family_history_osteoporosis |      -0.000864161 |       0.00259248 |
| is_menopausal               |      -0.000939267 |       0.0028178  |
| weight_kg                   |      -0.00210262  |       0.0109123  |
| history_of_fracture         |      -0.00223693  |       0.0103219  |
| obesity_over weight         |      -0.00709214  |       0.0157886  |
| hypothyroidism              |      -0.00777745  |       0.00259248 |
| menopause_age               |      -0.0100168   |       0.0125561  |
| max_walking_km              |      -0.0113847   |       0.0118892  |
| age                         |      -0.0157817   |       0.00938651 |

## 3. Important Scientific Caveat
> [!NOTE]
> **Attribution vs Causality**: Grad-CAM highlights discriminative correlation in the CNN feature maps. It does not constitute proof of physiological causal reasoning or clinical bone mineral density diagnosis.
