"""
Dataset auditing executable script.
Discovers images, checks integrity, calculates SHA256 and perceptual hashes,
detects duplicates and label conflicts, and generates audit reports.
"""
import os
import sys
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath("."))

from src.utils import load_config, setup_logger, save_json
from src.data_ingestion import ingest_mendeley_dataset, ingest_kaggle_dataset, ingest_zenodo_dataset, ingest_clinical_metadata
from src.deduplication import (
    detect_exact_duplicates,
    detect_pixel_duplicates,
    detect_label_conflicts,
    compute_perceptual_near_duplicates,
    audit_mendeley_zenodo_overlap,
)
from src.dataset_audit import (
    audit_dataset_integrity,
    generate_missing_metadata_report,
    generate_markdown_audit_report,
)

def run_audit():
    logger = setup_logger("audit")
    config = load_config()

    mendeley_dir = config["paths"]["mendeley_dir"]
    kaggle_dir = config["paths"].get("kaggle_dir")
    zenodo_dir = config["paths"]["zenodo_dir"]
    metadata_excel = config["paths"]["metadata_excel"]
    reports_dir = config["paths"]["reports_dir"]

    os.makedirs(reports_dir, exist_ok=True)

    logger.info("Starting dataset discovery and integrity inspection...")
    df_mendeley = ingest_mendeley_dataset(mendeley_dir)
    logger.info(f"Ingested Mendeley: {len(df_mendeley)} images")

    df_kaggle = ingest_kaggle_dataset(kaggle_dir) if kaggle_dir else pd.DataFrame()
    logger.info(f"Ingested Kaggle: {len(df_kaggle)} images")

    df_zenodo = ingest_zenodo_dataset(zenodo_dir)
    logger.info(f"Ingested Zenodo: {len(df_zenodo)} images")

    # Combine into unified raw manifest
    manifest_df = pd.concat([df_mendeley, df_kaggle, df_zenodo], ignore_index=True)
    manifest_csv = os.path.join(reports_dir, "image_manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    logger.info(f"Saved complete image manifest ({len(manifest_df)} rows) to {manifest_csv}")

    # Ingest clinical metadata
    metadata_df = ingest_clinical_metadata(metadata_excel)
    logger.info(f"Ingested clinical metadata: {len(metadata_df)} rows")

    # Audit metadata missingness
    missing_report_csv = os.path.join(reports_dir, "missing_metadata_report.csv")
    generate_missing_metadata_report(metadata_df, missing_report_csv)
    logger.info(f"Saved missing metadata report to {missing_report_csv}")

    # Deduplication and conflict audits
    logger.info("Running duplicate detection and cross-source comparison...")
    exact_dups = detect_exact_duplicates(manifest_df)
    pixel_dups = detect_pixel_duplicates(manifest_df)
    conflicts_df = detect_label_conflicts(manifest_df)
    
    conflicts_csv = os.path.join(reports_dir, "label_conflicts.csv")
    conflicts_df.to_csv(conflicts_csv, index=False)
    logger.info(f"Label conflicts detected: {len(conflicts_df)}. Saved to {conflicts_csv}")

    overlap_df = audit_mendeley_zenodo_overlap(manifest_df)
    logger.info(f"Mendeley-to-Zenodo direct image overlaps: {len(overlap_df)}")

    # Sample perceptual near duplicates on Mendeley to evaluate intra-cohort similarity
    near_dups_df = compute_perceptual_near_duplicates(df_mendeley, hamming_threshold=4, max_comparisons=30000)
    dup_review_csv = os.path.join(reports_dir, "duplicate_review.csv")
    near_dups_df.to_csv(dup_review_csv, index=False)
    logger.info(f"Perceptual near-duplicates saved to {dup_review_csv}")

    # Overall dataset audit summary
    audit_summary = audit_dataset_integrity(manifest_df, metadata_df)
    audit_json = os.path.join(reports_dir, "dataset_audit.json")
    save_json(audit_summary, audit_json)
    logger.info(f"Saved dataset audit JSON to {audit_json}")

    audit_md = os.path.join(reports_dir, "dataset_audit_report.md")
    generate_markdown_audit_report(audit_summary, conflicts_df, near_dups_df, overlap_df, audit_md)
    logger.info(f"Saved Markdown audit report to {audit_md}")

    logger.info("Dataset audit completed successfully.")

if __name__ == "__main__":
    run_audit()
