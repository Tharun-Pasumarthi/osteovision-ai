# Dataset Documentation & Source Register

## Data Source Register

| Source Name | Version / Date | Location / DOI | License | Original Class Counts | Patient IDs Available | Label Source & Provenance | Image Count | Duplicate / Augmentation Status | Accessibility Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mendeley Knee X-ray Database** | v2 (2021) | [10.17632/fxjm8fb6mw.2](https://data.mendeley.com/datasets/fxjm8fb6mw/2) | CC BY 4.0 | Normal: 36<br>Osteopenia: 154<br>Osteoporosis: 49 | Yes (`N1`..`N36`, `OP1`..`OP154`, `OS1`..`OS49`) | Quantitative Ultrasound System (QUS) T-score and diagnostic thresholds | 239 knee radiographs (243 participant metadata rows) | Ground-truth original examinations, unaugmented | **Accessible & Downloaded** (`data/raw/mendeley`) |
| **Zenodo Combined Collection** | Record 20053147 | [10.5281/zenodo.20053147](https://zenodo.org/records/20053147) | Creative Commons Attribution 4.0 | Normal: 1800<br>Osteopenia: 1800<br>Osteoporosis: 1800 (balanced via augmentation) | No individual patient identifiers preserved | Derived from Mendeley and Kaggle; labels inherited from QUS sources | 5,400 images across train (3780), val (1080), test (540) | Contains extensive rotation, flipping, and synthetic duplicates of Mendeley images | **Accessible & Downloaded** (`data/raw/zenodo`) |
| **Published Few-Shot Study** | PMC11375658 (2024) | [PMC11375658](https://pmc.ncbi.nlm.nih.gov/articles/PMC11375658/) | Authors / Sage Pubs | Jintan Hospital Cohort & Soochow University Cohort | Internal hospital MRN / accession | Dual-Energy X-ray Absorptiometry (DXA) at femoral neck & lumbar spine | Not public (Hospital retrospective review #2022098) | External hospital cohort | **Hospital-Restricted PHI** (Code repositories on OSF returned 404; raw radiographs are protected clinical data requiring institutional IRB agreement) |

## Key Research Protocols & Protections
1. **Label Provenance**: All accessible labels from Mendeley are **QUS-derived** (Quantitative Ultrasound System T-scores), *not* gold-standard DXA scans. This distinction is strictly documented across all reports and models.
2. **Leakage Control**: Raw Zenodo splits have data leakage because images were augmented before splitting. For rigorous research, we anchor our validation and testing on patient-isolated splits from Mendeley.
3. **No Synthetic Patients**: Clinical metadata features come strictly from `patient details.xlsx` without fabricating synthetic records.
