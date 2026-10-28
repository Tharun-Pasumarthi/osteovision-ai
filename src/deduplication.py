"""
Deduplication and conflict detection module.
Performs exact SHA-256 matching, decoded pixel comparison, perceptual hash distance,
cross-source duplicate tracking, and label conflict quarantine.
"""
from typing import Dict, List, Tuple, Any
import pandas as pd
import imagehash

def detect_exact_duplicates(df: pd.DataFrame, hash_col: str = "sha256") -> pd.DataFrame:
    """Find exact file hash duplicates."""
    dups = df[df.duplicated(subset=[hash_col], keep=False)].copy()
    if not dups.empty:
        dups["duplicate_type"] = "exact_file_hash"
    return dups

def detect_pixel_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Find identical decoded pixel duplicates across different file compressions."""
    dups = df[df.duplicated(subset=["pixel_hash"], keep=False)].copy()
    if not dups.empty:
        dups["duplicate_type"] = "exact_pixel_hash"
    return dups

def detect_label_conflicts(df: pd.DataFrame, group_col: str = "sha256") -> pd.DataFrame:
    """
    Quarantine images with identical file hash or pixel content but conflicting class labels.
    """
    conflicts = []
    # Check sha256
    for key, group in df.groupby(group_col):
        labels = group["class_label"].dropna().unique()
        if len(labels) > 1:
            for _, row in group.iterrows():
                conflicts.append({
                    "conflict_group_key": key,
                    "image_path": row["image_path"],
                    "filename": row.get("filename", ""),
                    "source_dataset": row.get("source_dataset", ""),
                    "patient_id": row.get("patient_id", ""),
                    "class_label": row["class_label"],
                    "all_conflicting_labels": ", ".join(labels),
                    "action": "quarantined_for_investigation",
                })
    return pd.DataFrame(conflicts)

def compute_perceptual_near_duplicates(
    df: pd.DataFrame,
    phash_col: str = "phash",
    hamming_threshold: int = 4,
    max_comparisons: int = 50000,
) -> pd.DataFrame:
    """
    Compare perceptual hashes (pHash) to find compressed, resized, or augmented copies.
    """
    records = []
    valid = df.dropna(subset=[phash_col]).reset_index(drop=True)
    n = len(valid)
    comparisons = 0

    for i in range(n):
        row_i = valid.iloc[i]
        h_i = imagehash.hex_to_hash(row_i[phash_col])
        for j in range(i + 1, n):
            comparisons += 1
            if comparisons > max_comparisons:
                break
            row_j = valid.iloc[j]
            h_j = imagehash.hex_to_hash(row_j[phash_col])
            dist = h_i - h_j
            if dist <= hamming_threshold:
                records.append({
                    "image_1": row_i["image_path"],
                    "filename_1": row_i.get("filename", ""),
                    "source_1": row_i.get("source_dataset", ""),
                    "class_1": row_i.get("class_label", ""),
                    "patient_1": row_i.get("patient_id", ""),
                    "image_2": row_j["image_path"],
                    "filename_2": row_j.get("filename", ""),
                    "source_2": row_j.get("source_dataset", ""),
                    "class_2": row_j.get("class_label", ""),
                    "patient_2": row_j.get("patient_id", ""),
                    "phash_hamming_distance": dist,
                    "same_label": (row_i.get("class_label") == row_j.get("class_label")),
                    "relation_type": "near_duplicate" if dist > 0 else "perceptual_identical",
                })
        if comparisons > max_comparisons:
            break

    return pd.DataFrame(records)

def audit_mendeley_zenodo_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify specific overlap where Zenodo images derive from Mendeley originals.
    """
    mendeley_df = df[df["source_dataset"] == "mendeley"]
    zenodo_df = df[df["source_dataset"] == "zenodo"]
    overlap_records = []

    m_hash_map = {row["sha256"]: row for _, row in mendeley_df.iterrows()}
    m_pixel_map = {row["pixel_hash"]: row for _, row in mendeley_df.iterrows()}

    for _, z_row in zenodo_df.iterrows():
        z_sha = z_row["sha256"]
        z_px = z_row["pixel_hash"]
        match_type = None
        matched_mendeley = None

        if z_sha in m_hash_map:
            match_type = "exact_sha256_match"
            matched_mendeley = m_hash_map[z_sha]
        elif z_px in m_pixel_map:
            match_type = "exact_pixel_match"
            matched_mendeley = m_pixel_map[z_px]

        if matched_mendeley is not None:
            overlap_records.append({
                "zenodo_image_path": z_row["image_path"],
                "zenodo_filename": z_row["filename"],
                "zenodo_split": z_row.get("zenodo_raw_split", "unknown"),
                "mendeley_image_path": matched_mendeley["image_path"],
                "mendeley_filename": matched_mendeley["filename"],
                "patient_id": matched_mendeley["patient_id"],
                "class_label": matched_mendeley["class_label"],
                "match_type": match_type,
            })

    return pd.DataFrame(overlap_records)
