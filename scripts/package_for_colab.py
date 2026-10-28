"""
Utility script to package dataset or project files for Google Colab upload.
Creates clean zip archives ready to drag-and-drop into Colab.
"""
import os
import sys
import zipfile
import argparse
from pathlib import Path

def package_mendeley_dataset(output_zip: str = "mendeley_dataset.zip"):
    src_dir = "data/raw/mendeley"
    if not os.path.exists(src_dir):
        print(f"Directory not found: {src_dir}")
        return
    print(f"Packaging {src_dir} into {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, "data/raw")
                zf.write(full_path, arcname=rel_path)
    print(f"Created {output_zip} ({os.path.getsize(output_zip) / 1e6:.2f} MB)")

def package_balanced_dataset(output_zip: str = "balanced_cohort_dataset.zip"):
    src_dir = "data/balanced_cohort_2400"
    if not os.path.exists(src_dir):
        print(f"Directory not found: {src_dir}")
        return
    print(f"Packaging {src_dir} into {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, "data")
                zf.write(full_path, arcname=rel_path)
        # Add manifest if present
        manifest = "outputs/reports/balanced_2400_manifest.csv"
        if os.path.exists(manifest):
            zf.write(manifest, arcname="balanced_2400_manifest.csv")
    print(f"Created {output_zip} ({os.path.getsize(output_zip) / 1e6:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package datasets for Google Colab")
    parser.add_argument("--cohort", choices=["mendeley", "balanced", "both"], default="both")
    args = parser.parse_args()

    if args.cohort in ["mendeley", "both"]:
        package_mendeley_dataset()
    if args.cohort in ["balanced", "both"]:
        package_balanced_dataset()
