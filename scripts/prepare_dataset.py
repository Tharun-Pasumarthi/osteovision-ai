"""
Dataset preparation executable script.
Performs patient-grouped partitioning with zero-leakage verification,
fits clinical preprocessor strictly on training data, and generates quality control figures.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.utils import load_config, setup_logger, save_json, set_seed
from src.data_splitting import create_patient_grouped_splits, verify_zero_leakage
from src.metadata_preprocessing import ClinicalFeaturePreprocessor, build_feature_dictionary
from src.image_transforms import generate_preprocessing_qc_figure

def prepare_data():
    logger = setup_logger("prepare")
    config = load_config()
    set_seed(config.get("project", {}).get("random_seed", 42))

    reports_dir = config["paths"]["reports_dir"]
    checkpoints_dir = config["paths"]["checkpoints_dir"]
    figures_dir = config["paths"]["figures_dir"]
    manifest_csv = os.path.join(reports_dir, "image_manifest.csv")

    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}. Run audit_dataset.py first.")

    manifest_df = pd.read_csv(manifest_csv)
    conflicts_csv = os.path.join(reports_dir, "label_conflicts.csv")
    conflicts_df = pd.read_csv(conflicts_csv) if os.path.exists(conflicts_csv) else pd.DataFrame()
    conflict_keys = set(conflicts_df["conflict_group_key"].unique()) if not conflicts_df.empty else set()

    # Select authentic Mendeley and clean Kaggle candidate images (excluding Zenodo due to cross-split leakage)
    candidates = manifest_df[
        manifest_df["source_dataset"].isin(["mendeley", "kaggle_gobara"])
        & (~manifest_df["sha256"].isin(conflict_keys))
    ].copy()

    # Prioritize Mendeley so when deduplicating identical image hashes, Mendeley's patient_id is preserved
    candidates["is_mendeley"] = candidates["source_dataset"] == "mendeley"
    candidates = candidates.sort_values(by=["is_mendeley"], ascending=False).reset_index(drop=True)
    clean_unique_df = candidates.drop_duplicates(subset=["sha256"]).copy().reset_index(drop=True)
    
    # Ensure patient_id is populated for all images
    clean_unique_df["patient_id"] = clean_unique_df["patient_id"].fillna(clean_unique_df["filename"])
    logger.info(f"Loaded {len(clean_unique_df)} clean, unique, non-conflicting knee radiographs (Mendeley + Kaggle).")

    # Ingest clinical metadata
    metadata_df = pd.read_excel(config["paths"]["metadata_excel"])
    metadata_df = metadata_df.dropna(subset=["Patient Id"]).copy()
    metadata_df["patient_id"] = metadata_df["Patient Id"].astype(str).str.strip().str.upper()

    # Merge clinical data
    merged_df = pd.merge(clean_unique_df, metadata_df, on="patient_id", how="left", suffixes=("", "_excel"))
    merged_df["has_clinical_metadata"] = merged_df["Patient Id"].notna()
    logger.info(f"Merged cohort: {len(merged_df)} total ({merged_df['has_clinical_metadata'].sum()} with clinical metadata).")

    # 1. Patient-grouped stratified split (70% train, 15% val, 15% test)
    logger.info("Generating patient-isolated stratified partitions (70/15/15)...")
    split_df = create_patient_grouped_splits(
        merged_df,
        patient_col="patient_id",
        class_col="class_label",
        train_ratio=config["splitting"]["train_ratio"],
        val_ratio=config["splitting"]["val_ratio"],
        test_ratio=config["splitting"]["test_ratio"],
        random_seed=config["project"]["random_seed"],
    )

    # 2. Automated leakage checks
    logger.info("Executing automated cross-partition leakage assertions...")
    leakage_report = verify_zero_leakage(split_df, patient_col="patient_id", hash_col="sha256")
    logger.info(f"Leakage check passed! Split distribution: {leakage_report['split_counts']}")
    logger.info(f"Class by split: {leakage_report['class_by_split']}")

    split_manifest_csv = os.path.join(reports_dir, "split_manifest.csv")
    split_df.to_csv(split_manifest_csv, index=False)
    logger.info(f"Saved verified split manifest to {split_manifest_csv}")

    # 3. Fit clinical preprocessor strictly on training partition
    train_df = split_df[split_df["split"] == "train"]
    logger.info(f"Fitting ClinicalFeaturePreprocessor on {len(train_df)} training records...")
    preprocessor = ClinicalFeaturePreprocessor()
    preprocessor.fit(train_df)

    preproc_path = os.path.join(checkpoints_dir, "clinical_preprocessor.pkl")
    preprocessor.save(preproc_path)
    logger.info(f"Saved fitted clinical preprocessor to {preproc_path}")

    # Save feature dictionary
    feat_dict = build_feature_dictionary()
    feat_dict_json = os.path.join(reports_dir, "metadata_dictionary.json")
    save_json(feat_dict, feat_dict_json)
    logger.info(f"Saved metadata dictionary to {feat_dict_json}")

    # Generate feature availability report
    features_avail = []
    for feat in preprocessor.feature_names:
        features_avail.append({
            "feature_name": feat,
            "training_median_or_mode": preprocessor.medians.get(feat, "N/A"),
            "is_standardized": feat in preprocessor.numerical_cols,
        })
    df_avail = pd.DataFrame(features_avail)
    avail_csv = os.path.join(reports_dir, "feature_availability_report.csv")
    df_avail.to_csv(avail_csv, index=False)
    logger.info(f"Saved feature availability report to {avail_csv}")

    # 4. Generate visual preprocessing QC figure
    sample_img = split_df.iloc[0]["image_path"]
    qc_png = os.path.join(figures_dir, "preprocessing_qc.png")
    generate_preprocessing_qc_figure(sample_img, qc_png, target_size=224)
    logger.info(f"Saved visual preprocessing quality control figure to {qc_png}")

    logger.info("Dataset preparation completed successfully.")

if __name__ == "__main__":
    prepare_data()
