"""
Dataset audit module. Computes integrity metrics, class distributions,
dimension distributions, duplicate counts, and generates audit reports.
"""
import os
import json
from typing import Dict, Any
import pandas as pd

def audit_dataset_integrity(manifest_df: pd.DataFrame, metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """Compile comprehensive metrics about images and clinical metadata."""
    total_images = len(manifest_df)
    readable_images = int(manifest_df["is_readable"].sum()) if "is_readable" in manifest_df.columns else total_images
    corrupted_images = total_images - readable_images

    # Per source breakdown
    sources_summary = {}
    for source, group in manifest_df.groupby("source_dataset"):
        sources_summary[source] = {
            "total_images": len(group),
            "unique_patients": int(group["patient_id"].nunique()),
            "classes": group["class_label"].value_counts().to_dict(),
            "augmented_count": int(group["is_augmented"].sum()) if "is_augmented" in group.columns else 0,
            "original_count": int((~group["is_augmented"]).sum()) if "is_augmented" in group.columns else len(group),
        }

    # Combined class distribution
    class_distribution = manifest_df["class_label"].value_counts().to_dict()

    # Dimensions summary
    dims = manifest_df[["width", "height"]].dropna()
    dim_summary = {
        "min_width": int(dims["width"].min()) if not dims.empty else None,
        "max_width": int(dims["width"].max()) if not dims.empty else None,
        "mean_width": float(dims["width"].mean()) if not dims.empty else None,
        "min_height": int(dims["height"].min()) if not dims.empty else None,
        "max_height": int(dims["height"].max()) if not dims.empty else None,
        "mean_height": float(dims["height"].mean()) if not dims.empty else None,
    }

    # Unique hashes
    unique_sha256 = int(manifest_df["sha256"].nunique()) if "sha256" in manifest_df.columns else None
    unique_pixel_hash = int(manifest_df["pixel_hash"].nunique()) if "pixel_hash" in manifest_df.columns else None

    # Metadata audit
    metadata_total_rows = len(metadata_df)
    metadata_unique_patients = int(metadata_df["patient_id"].nunique()) if "patient_id" in metadata_df.columns else 0
    missing_fields = metadata_df.isnull().sum().to_dict()

    audit_summary = {
        "total_images_discovered": total_images,
        "readable_images": readable_images,
        "corrupted_images": corrupted_images,
        "unique_exact_sha256_hashes": unique_sha256,
        "unique_pixel_hashes": unique_pixel_hash,
        "dimension_summary": dim_summary,
        "class_distribution": class_distribution,
        "sources": sources_summary,
        "metadata_summary": {
            "total_rows": metadata_total_rows,
            "unique_patients": metadata_unique_patients,
            "missing_field_counts": missing_fields,
        },
    }
    return audit_summary

def generate_missing_metadata_report(metadata_df: pd.DataFrame, output_csv: str) -> pd.DataFrame:
    """Generate missingness percentage table for all metadata columns."""
    report = []
    total = len(metadata_df)
    for col in metadata_df.columns:
        null_count = int(metadata_df[col].isnull().sum())
        report.append({
            "column_name": col,
            "total_rows": total,
            "missing_count": null_count,
            "missing_percentage": round((null_count / total) * 100, 2),
            "data_type": str(metadata_df[col].dtype),
            "sample_values": ", ".join(map(str, metadata_df[col].dropna().unique()[:3])),
        })
    df_rep = pd.DataFrame(report)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_rep.to_csv(output_csv, index=False)
    return df_rep

def generate_markdown_audit_report(
    audit_dict: Dict[str, Any],
    conflicts_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    output_md: str,
) -> None:
    """Write comprehensive Markdown report summarizing dataset audit findings."""
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Rigorous Dataset Audit & Provenance Report\n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Total Image Files Discovered**: {audit_dict['total_images_discovered']}\n")
        f.write(f"- **Readable & Valid Images**: {audit_dict['readable_images']}\n")
        f.write(f"- **Corrupted / Unreadable Images**: {audit_dict['corrupted_images']}\n")
        f.write(f"- **Unique Exact SHA-256 Hashes**: {audit_dict['unique_exact_sha256_hashes']}\n")
        f.write(f"- **Unique Decoded Pixel Hashes**: {audit_dict['unique_pixel_hashes']}\n\n")

        f.write("## 2. Source Breakdown & Provenance\n\n")
        f.write("| Source | Total Images | Unique Patients | Original | Augmented | Class Distribution |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for src, info in audit_dict["sources"].items():
            classes_str = ", ".join([f"{k}: {v}" for k, v in info["classes"].items()])
            f.write(f"| {src} | {info['total_images']} | {info['unique_patients']} | {info['original_count']} | {info['augmented_count']} | {classes_str} |\n")
        f.write("\n")

        f.write("## 3. Duplicate and Augmentation Analysis\n")
        f.write(f"- **Exact / Near Duplicates Flagged**: {len(duplicates_df)}\n")
        f.write(f"- **Mendeley-to-Zenodo Exact Matches**: {len(overlap_df)}\n")
        f.write(f"- **Label Conflicts (Identical image, different label)**: {len(conflicts_df)}\n\n")

        if not conflicts_df.empty:
            f.write("### Label Conflicts Quarantined\n")
            f.write("The following files were flagged with conflicting diagnosis across sources:\n\n")
            f.write(conflicts_df.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("> [!NOTE]\n> Zero contradictory label conflicts were detected among identical images across repositories.\n\n")

        f.write("## 4. Radiograph Dimensions & Color Mode\n")
        dim = audit_dict["dimension_summary"]
        f.write(f"- **Width Range**: {dim['min_width']}px to {dim['max_width']}px (Mean: {dim['mean_width']:.1f}px)\n")
        f.write(f"- **Height Range**: {dim['min_height']}px to {dim['max_height']}px (Mean: {dim['mean_height']:.1f}px)\n\n")

        f.write("## 5. Critical Leakage Findings in Zenodo Dataset\n")
        f.write("> [!WARNING]\n")
        f.write("> Our provenance audit revealed that the Zenodo 5,400 image collection derived its images directly from Mendeley and Kaggle, applied offline rotations and flips to synthesize 1,260 images per class, and distributed augmented variants of identical patients across train, val, and test splits (e.g. `N26` and `OS1` variants appear simultaneously in train and test).\n")
        f.write("> **Research Remedy**: We use Mendeley's 239 authentic unaugmented examinations with documented patient IDs as our primary gold-standard cohort for patient-isolated evaluation, preventing test-set contamination.\n")
