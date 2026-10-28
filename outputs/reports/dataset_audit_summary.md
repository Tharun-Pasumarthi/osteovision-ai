# Dataset Quality & Label Audit Summary

## 1. Discovered Source Breakdown
- **Mendeley Data (fxjm8fb6mw/2)**: 239 knee radiographs, 240 patient clinical records. Diagnostic ground truth is derived from calcaneal Quantitative Ultrasound System (QUS) T-score proxies ($T > -1.0$ Normal, $-2.5 < T < -1.0$ Osteopenia, $T \le -2.5$ Osteoporosis).
- **Kaggle Gobara Dataset**: 1,947 raw radiographs. Provenance analysis showed extensive internal exact duplicates (1,003 duplicate files) and 79 internal label conflict clusters where identical images were placed in both Normal and Osteoporosis/Osteopenia folders.
- **Zenodo Collection (20053147)**: 5,400 images. Derived from Mendeley and Kaggle via offline rotations/flips. Identical patients cross raw train/val/test folders. Quarantined to prevent contamination.
- **Published Study Hospital Cohorts (PMC11375658)**: Hospital-restricted Protected Health Information (PHI) under retrospective IRB #2022098. OSF repository returned 404. Confirmed inaccessible; no synthetic records fabricated.

## 2. Duplicate and Conflict Audit Findings
- **Total Discovered Files**: 7586
- **Quarantined Label Conflicts**: 704 files (158 distinct SHA-256 groups) where identical byte contents were assigned contradictory class diagnoses across or within repositories.
- **Deduplicated Clean Cohort**: 865 authentic, unique, conflict-free knee radiographs (Mendeley + Kaggle).
- **Class Distribution in Clean Cohort**:
  - **Normal**: 350
  - **Osteopenia**: 189
  - **Osteoporosis**: 326

## 3. Visual Gallery for Manual Review
Saved to `outputs/figures/sample_review_gallery.png`.
