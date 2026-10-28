# Data Quality & Provenance Audit Report

## 1. Executive Summary & Cohort Overview

This audit rigorously evaluates the data integrity, duplicates, label contradictions, patient-level grouping, and leakage across all knee radiograph cohorts used for osteoporosis classification.

### High-Level Statistics
- **Total Images Audited**: 7,586 files across 3 public data repositories.
- **Readable Files**: 7,586 / 7,586 (100% format validity, 0 corrupted files).
- **Exact SHA-256 Duplicates**: 1,888 duplicate hash groups identified.
- **Conflicting Label Groups**: 158 duplicate hash groups (704 files) where identical images were assigned contradictory diagnostic labels.
- **Leakage-Controlled Clean Split**: 865 authentic, deduplicated radiographs partitioned into Train (605), Validation (129), and Test (131).

---

## 2. Dataset Provenance & Collection Characteristics

### A. Mendeley Data (fxjm8fb6mw/2)
- **Image Count**: 239 authentic radiograph files corresponding to 240 patient clinical records.
- **Label Distribution**:
  - Osteopenia: 154 (64.4%)
  - Osteoporosis: 49 (20.5%)
  - Normal: 36 (15.1%)
- **Diagnostic Ground Truth**:
  - Ground truth labels were derived from **Quantitative Ultrasound System (QUS) calcaneal bone mineral density (BMD) T-scores**, NOT gold-standard dual-energy X-ray absorptiometry (DXA).
  - T-Score thresholds applied: Normal ($T \ge -1.0$), Osteopenia ($-2.5 < T < -1.0$), Osteoporosis ($T \le -2.5$).
- **Radiograph Geometry**: Full bilateral anterior-posterior (AP) radiographs ($1024 \times 1024$ or $2180 \times 2660$) showing both knee joints in a single view.
- **Patient Identifiers**: Authentic patient IDs (`N1` to `N36`, `OP1` to `OP154`, `OS1` to `OS49`) mapped to clinical tabular records in `patient details.xlsx`.

### B. Zenodo Cohort (20053147)
- **Image Count**: 5,400 image files organized into synthetic Train (3,600), Validation (900), and Test (900) folders.
- **Audit Findings**:
  - 100% of Zenodo images are derived from offline affine transformations (rotations, flips, crops) of Mendeley patient radiographs.
  - Exactly 661 duplicate image groups cross Zenodo's train and test split directories.
  - **Action Taken**: **Quarantined completely**. The Zenodo synthetic collection was excluded from all model training and testing to prevent test-set contamination.

### C. Kaggle OS Collected Data (Gobara)
- **Image Count**: 1,947 raw radiograph files.
- **Label Distribution**:
  - Osteoporosis: 793 (40.7%)
  - Normal: 780 (40.1%)
  - Osteopenia: 374 (19.2%)
- **Radiograph Geometry**: Unilateral cropped views ($525 \times 1024$ or $1128 \times 2660$) showing only a single isolated knee joint.
- **Provenance & Patient ID Limitation**:
  - Source does not provide clinical DICOM metadata, institutional ethics board identifiers, or DXA scan reports.
  - **Limitation Disclosure**: Patient-level identifiers are unavailable for the Kaggle cohort. Patient IDs for Kaggle files are derived from filenames (e.g. `Normal 125`). True cross-image patient independence cannot be verified for Kaggle images; this limitation is explicitly disclosed.

---

## 3. Duplicate and Label Conflict Registry

### A. Exact SHA-256 Hash Matching
Full cryptographic hashing identified 5,102 unique image hashes across the 7,586 files:
- 1,888 duplicate groups contained 2 or more files with byte-for-byte identical pixel payloads.
- 704 files across 158 hash groups contained **conflicting diagnostic labels** across folders or datasets.

### B. Critical Label Conflict Examples (Quarantined)
| Image Filename A | Assigned Label A | Image Filename B | Assigned Label B | Source Dataset | Status |
| :--- | :---: | :--- | :---: | :--- | :--- |
| `Normal 104.JPEG` | Normal | `Osteoporosis 120.JPEG` | Osteoporosis | Kaggle | **Quarantined** |
| `Normal 201.jpg` | Normal | `Osteopenia 230.jpg` | Osteopenia | Kaggle | **Quarantined** |
| `Osteopenia 262.jpg` | Osteopenia | `Osteoporosis 347.jpg` | Osteoporosis | Kaggle | **Quarantined** |
| `N23.jpg` | Normal | `OP107.jpg` | Osteopenia | Mendeley | **Quarantined** |
| `OP136.jpg` | Osteopenia | `Osteoporosis 76.jpg` | Osteoporosis | Mendeley / Kaggle | **Quarantined** |

All 704 conflicting files were recorded in `outputs/reports/label_conflicts.csv` and excluded from the clean benchmark partitions.

---

## 4. Split Partitioning & Leakage Verification

### A. Clean Manifest Counts
After removing Zenodo synthetic duplicates and quarantining all 158 conflicting hash groups, 865 authentic, unique radiographs remained:

| Split Partition | Normal | Osteopenia | Osteoporosis | Total Images | Unique Patients | Unique SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (70%)** | 245 | 132 | 228 | **605** | 605 | 605 |
| **Validation (15%)** | 52 | 28 | 49 | **129** | 129 | 129 |
| **Test (15%)** | 53 | 29 | 49 | **131** | 131 | 131 |
| **Total Clean Cohort** | **350** | **189** | **326** | **865** | **865** | **865** |

### B. Automated Cross-Partition Leakage Verification
- **Patient ID Overlap**:
  - Train $\cap$ Validation: **0**
  - Train $\cap$ Test: **0**
  - Validation $\cap$ Test: **0**
- **Image SHA-256 Hash Overlap**:
  - Train $\cap$ Validation: **0**
  - Train $\cap$ Test: **0**
  - Validation $\cap$ Test: **0**
- **Test Set Isolation**: The test set was locked and untouched during model training, hyperparameter tuning, and probability calibration.

### C. Patient Identifier Independence Limitation Disclosure
- For Mendeley radiographs ($N = 153$ in clean split), patient IDs (`N*`, `OP*`, `OS*`) are genuine clinical identifiers verified against patient records.
- For Kaggle radiographs ($N = 712$ in clean split), patient identifiers are inferred from image filenames. While duplicate hashes are strictly clustered and isolated, **patient-level independence cannot be claimed with certainty for the Kaggle cohort** due to the absence of clinical patient registry numbers.
