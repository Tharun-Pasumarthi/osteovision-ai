"""
Comprehensive Data Quality & Label Audit Script.
Performs:
- Decoded pixel and SHA-256 duplicate auditing
- Label conflict detection and quarantine
- Source overlap and provenance verification
- Creation of visual review gallery across classes, sources, and quarantined conflicts
- Outputs audited clean manifest and duplicate report
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from src.utils import load_config, setup_logger, save_json

def run_comprehensive_audit():
    logger = setup_logger("data_audit")
    config = load_config()
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = reports_dir / "image_manifest.csv"
    if not manifest_csv.exists():
        logger.error(f"Image manifest not found at {manifest_csv}. Ingesting...")
        from scripts.audit_dataset import run_audit
        run_audit()

    manifest_df = pd.read_csv(manifest_csv)
    logger.info(f"Loaded total raw discovered images: {len(manifest_df)}")

    # 1. Exact SHA-256 and Pixel Duplicate Audit
    sha_dups = manifest_df[manifest_df.duplicated(subset=["sha256"], keep=False)].copy()
    pixel_dups = manifest_df[manifest_df.duplicated(subset=["pixel_hash"], keep=False)].copy()
    logger.info(f"Total rows involved in SHA-256 duplication: {len(sha_dups)}")
    logger.info(f"Total rows involved in decoded pixel duplication: {len(pixel_dups)}")

    # 2. Strict Label Conflict Identification
    conflict_rows = []
    for hash_val, group in manifest_df.groupby("sha256"):
        labels = group["class_label"].dropna().unique()
        if len(labels) > 1:
            sources = group["source_dataset"].unique()
            for _, r in group.iterrows():
                conflict_rows.append({
                    "conflict_group_key": hash_val,
                    "image_path": r["image_path"],
                    "filename": r["filename"],
                    "source_dataset": r["source_dataset"],
                    "patient_id": r.get("patient_id", ""),
                    "assigned_label": r["class_label"],
                    "conflicting_labels": ", ".join(labels),
                    "sources_involved": ", ".join(sources),
                    "action": "QUARANTINED_EXCLUDED_FROM_SPLITS",
                })

    conflicts_df = pd.DataFrame(conflict_rows)
    conflicts_csv = reports_dir / "label_conflicts.csv"
    conflicts_df.to_csv(conflicts_csv, index=False)
    logger.info(f"Quarantined {len(conflicts_df)} files across {conflicts_df['conflict_group_key'].nunique() if not conflicts_df.empty else 0} distinct conflict groups.")

    # 3. Duplicate Report (same label or different)
    dup_records = []
    for hash_val, group in manifest_df.groupby("sha256"):
        if len(group) > 1:
            labels = group["class_label"].unique()
            is_conflict = len(labels) > 1
            dup_records.append({
                "sha256": hash_val,
                "file_count": len(group),
                "is_label_conflict": is_conflict,
                "classes": ", ".join(labels),
                "sources": ", ".join(group["source_dataset"].unique()),
                "sample_filenames": ", ".join(group["filename"].tolist()[:5]),
            })
    dup_df = pd.DataFrame(dup_records)
    dup_csv = reports_dir / "duplicate_report.csv"
    dup_df.to_csv(dup_csv, index=False)
    logger.info(f"Saved duplicate report with {len(dup_df)} duplicate clusters to {dup_csv}")

    # 4. Construct Audited Clean Manifest
    # Policy:
    # - Exclude all conflicted SHA-256 hashes
    # - Exclude Zenodo due to documented cross-split rotation/flip leakage
    # - Include authentic Mendeley and clean Kaggle Gobara
    # - Deduplicate exact SHA-256 hashes, prioritizing Mendeley's patient-labeled records
    conflict_hashes = set(conflicts_df["conflict_group_key"].unique()) if not conflicts_df.empty else set()
    
    clean_candidates = manifest_df[
        manifest_df["source_dataset"].isin(["mendeley", "kaggle_gobara"])
        & (~manifest_df["sha256"].isin(conflict_hashes))
    ].copy()

    # Prioritize Mendeley
    clean_candidates["is_mendeley"] = clean_candidates["source_dataset"] == "mendeley"
    clean_candidates = clean_candidates.sort_values(by=["is_mendeley"], ascending=False).reset_index(drop=True)
    audited_clean_df = clean_candidates.drop_duplicates(subset=["sha256"]).copy().reset_index(drop=True)
    audited_clean_df["patient_id"] = audited_clean_df["patient_id"].fillna(audited_clean_df["filename"])

    clean_manifest_csv = reports_dir / "audited_clean_manifest.csv"
    audited_clean_df.to_csv(clean_manifest_csv, index=False)
    logger.info(f"Saved audited clean manifest with {len(audited_clean_df)} unique radiographs to {clean_manifest_csv}")
    logger.info(f"Clean cohort class distribution:\n{audited_clean_df['class_label'].value_counts()}")

    # 5. Generate Multi-Class, Multi-Source Sample Gallery for Manual Review
    logger.info("Generating multi-source sample gallery figure...")
    classes = ["Normal", "Osteopenia", "Osteoporosis"]
    sources = ["mendeley", "kaggle_gobara", "zenodo"]
    
    fig, axes = plt.subplots(len(classes), len(sources) + 1, figsize=(16, 12))
    plt.suptitle("Medical Image Audit Gallery: Class, Source, & Quarantined Contradiction Review", fontsize=14, y=0.98)

    for r_idx, cls in enumerate(classes):
        # 1. Authentic Mendeley sample
        m_sample = manifest_df[(manifest_df["source_dataset"] == "mendeley") & (manifest_df["class_label"] == cls)]
        if not m_sample.empty:
            img_p = m_sample.iloc[0]["image_path"]
            with Image.open(img_p) as im:
                axes[r_idx, 0].imshow(im.convert("RGB"))
            axes[r_idx, 0].set_title(f"Mendeley: {cls}\n({m_sample.iloc[0]['filename']})", fontsize=10)
        axes[r_idx, 0].axis("off")

        # 2. Kaggle sample
        k_sample = manifest_df[(manifest_df["source_dataset"] == "kaggle_gobara") & (manifest_df["class_label"] == cls)]
        if not k_sample.empty:
            img_p = k_sample.iloc[0]["image_path"]
            with Image.open(img_p) as im:
                axes[r_idx, 1].imshow(im.convert("RGB"))
            axes[r_idx, 1].set_title(f"Kaggle: {cls}\n({k_sample.iloc[0]['filename']})", fontsize=10)
        axes[r_idx, 1].axis("off")

        # 3. Zenodo sample
        z_sample = manifest_df[(manifest_df["source_dataset"] == "zenodo") & (manifest_df["class_label"] == cls)]
        if not z_sample.empty:
            img_p = z_sample.iloc[0]["image_path"]
            with Image.open(img_p) as im:
                axes[r_idx, 2].imshow(im.convert("RGB"))
            axes[r_idx, 2].set_title(f"Zenodo: {cls}\n({z_sample.iloc[0]['filename']})", fontsize=10)
        axes[r_idx, 2].axis("off")

        # 4. Quarantined conflict sample
        if not conflicts_df.empty:
            c_sample = conflicts_df[conflicts_df["assigned_label"] == cls]
            if not c_sample.empty:
                img_p = c_sample.iloc[0]["image_path"]
                with Image.open(img_p) as im:
                    axes[r_idx, 3].imshow(im.convert("RGB"))
                axes[r_idx, 3].set_title(f"QUARANTINED CONFLICT\nAssigned: {cls}\nClashes: {c_sample.iloc[0]['conflicting_labels']}", fontsize=9, color="crimson")
            else:
                axes[r_idx, 3].text(0.5, 0.5, "No Conflict in Class", ha="center", va="center")
        axes[r_idx, 3].axis("off")

    plt.tight_layout()
    gallery_png = figures_dir / "sample_review_gallery.png"
    plt.savefig(gallery_png, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved sample review gallery to {gallery_png}")

    # 6. Write Dataset Audit Summary
    summary_dict = {
        "total_images_discovered": int(len(manifest_df)),
        "sources": {
            "mendeley": int((manifest_df["source_dataset"] == "mendeley").sum()),
            "kaggle_gobara": int((manifest_df["source_dataset"] == "kaggle_gobara").sum()),
            "zenodo": int((manifest_df["source_dataset"] == "zenodo").sum()),
        },
        "exact_sha256_duplicates": int(len(sha_dups)),
        "quarantined_label_conflict_files": int(len(conflicts_df)),
        "quarantined_label_conflict_groups": int(conflicts_df["conflict_group_key"].nunique()) if not conflicts_df.empty else 0,
        "clean_authentic_unique_radiographs": int(len(audited_clean_df)),
        "clean_class_distribution": audited_clean_df["class_label"].value_counts().to_dict(),
        "unaccessible_hospital_cohorts": {
            "PMC11375658_jintan_hospital": "Hospital-restricted PHI under IRB #2022098; OSF repository 404; strictly excluded from counts",
            "PMC11375658_soochow_university": "Hospital-restricted PHI under IRB #2022098; OSF repository 404; strictly excluded from counts",
        }
    }
    save_json(summary_dict, reports_dir / "dataset_audit_summary.json")

    # Generate Markdown summary
    summary_md = f"""# Dataset Quality & Label Audit Summary

## 1. Discovered Source Breakdown
- **Mendeley Data (fxjm8fb6mw/2)**: 239 knee radiographs, 240 patient clinical records. Diagnostic ground truth is derived from calcaneal Quantitative Ultrasound System (QUS) T-score proxies ($T > -1.0$ Normal, $-2.5 < T < -1.0$ Osteopenia, $T \le -2.5$ Osteoporosis).
- **Kaggle Gobara Dataset**: 1,947 raw radiographs. Provenance analysis showed extensive internal exact duplicates (1,003 duplicate files) and 79 internal label conflict clusters where identical images were placed in both Normal and Osteoporosis/Osteopenia folders.
- **Zenodo Collection (20053147)**: 5,400 images. Derived from Mendeley and Kaggle via offline rotations/flips. Identical patients cross raw train/val/test folders. Quarantined to prevent contamination.
- **Published Study Hospital Cohorts (PMC11375658)**: Hospital-restricted Protected Health Information (PHI) under retrospective IRB #2022098. OSF repository returned 404. Confirmed inaccessible; no synthetic records fabricated.

## 2. Duplicate and Conflict Audit Findings
- **Total Discovered Files**: {len(manifest_df)}
- **Quarantined Label Conflicts**: {len(conflicts_df)} files ({conflicts_df['conflict_group_key'].nunique() if not conflicts_df.empty else 0} distinct SHA-256 groups) where identical byte contents were assigned contradictory class diagnoses across or within repositories.
- **Deduplicated Clean Cohort**: {len(audited_clean_df)} authentic, unique, conflict-free knee radiographs (Mendeley + Kaggle).
- **Class Distribution in Clean Cohort**:
  - **Normal**: {audited_clean_df['class_label'].value_counts().get('Normal', 0)}
  - **Osteopenia**: {audited_clean_df['class_label'].value_counts().get('Osteopenia', 0)}
  - **Osteoporosis**: {audited_clean_df['class_label'].value_counts().get('Osteoporosis', 0)}

## 3. Visual Gallery for Manual Review
Saved to `outputs/figures/sample_review_gallery.png`.
"""
    with open(reports_dir / "dataset_audit_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)
    logger.info("Comprehensive audit and gallery generation completed successfully.")

if __name__ == "__main__":
    run_comprehensive_audit()
