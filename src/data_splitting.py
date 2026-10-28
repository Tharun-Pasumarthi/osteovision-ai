"""
Patient-grouped, leakage-resistant dataset partitioning module.
Groups connected patients sharing identical image hashes, quarantines label conflicts,
and ensures strict isolation across train, validation, and test splits.
"""
from typing import Dict, List, Tuple, Any, Set
import pandas as pd
import numpy as np

def build_patient_clusters(df: pd.DataFrame, patient_col: str = "patient_id", hash_col: str = "sha256") -> Dict[str, str]:
    """
    Build connected components of patients who share identical image hashes.
    Maps each patient_id to a cluster_id.
    """
    # Graph adjacency via hash
    patient_to_cluster: Dict[str, str] = {}
    hash_to_patients: Dict[str, Set[str]] = {}

    for _, row in df.iterrows():
        pid = str(row[patient_col])
        h = str(row[hash_col])
        if h not in hash_to_patients:
            hash_to_patients[h] = set()
        hash_to_patients[h].add(pid)

    # Union-find / connected components
    parent: Dict[str, str] = {}
    def find(p):
        if p not in parent:
            parent[p] = p
        if parent[p] != p:
            parent[p] = find(parent[p])
        return parent[p]

    def union(p1, p2):
        root1 = find(p1)
        root2 = find(p2)
        if root1 != root2:
            parent[root2] = root1

    for p_set in hash_to_patients.values():
        p_list = list(p_set)
        for i in range(1, len(p_list)):
            union(p_list[0], p_list[i])

    for pid in df[patient_col].unique():
        patient_to_cluster[str(pid)] = f"cluster_{find(str(pid))}"

    return patient_to_cluster

def create_patient_grouped_splits(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    class_col: str = "class_label",
    hash_col: str = "sha256",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    quarantine_conflicts: bool = True,
) -> pd.DataFrame:
    """
    Split dataset by patient cluster with class stratification.
    Identical images and linked patients are strictly kept in the exact same partition.
    """
    working_df = df.copy()

    # 1. Identify and quarantine label conflicts (same image hash, differing class labels)
    quarantined_mask = pd.Series(False, index=working_df.index)
    if quarantine_conflicts:
        hash_classes = working_df.groupby(hash_col)[class_col].nunique()
        conflict_hashes = set(hash_classes[hash_classes > 1].index)
        quarantined_mask = working_df[hash_col].isin(conflict_hashes)

    clean_df = working_df[~quarantined_mask].copy()

    # 2. Cluster patients sharing identical images
    patient_to_cluster = build_patient_clusters(clean_df, patient_col=patient_col, hash_col=hash_col)
    clean_df["cluster_id"] = clean_df[patient_col].map(patient_to_cluster)

    # 3. Stratified cluster assignment
    rng = np.random.RandomState(random_seed)
    cluster_classes = (
        clean_df.groupby("cluster_id")[class_col]
        .agg(lambda s: s.mode().iloc[0] if not s.empty else "Normal")
        .reset_index()
    )

    train_clusters, val_clusters, test_clusters = [], [], []

    for cls, group in cluster_classes.groupby(class_col):
        cids = group["cluster_id"].values.copy()
        rng.shuffle(cids)
        n = len(cids)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        n_test = n - n_train - n_val
        if n_test <= 0 and n >= 3:
            n_test = 1
            n_train -= 1

        train_c = cids[:n_train]
        val_c = cids[n_train:n_train + n_val]
        test_c = cids[n_train + n_val:]

        train_clusters.extend(train_c)
        val_clusters.extend(val_c)
        test_clusters.extend(test_c)

    train_set = set(train_clusters)
    val_set = set(val_clusters)
    test_set = set(test_clusters)

    def assign_split(cid: str) -> str:
        if cid in train_set:
            return "train"
        elif cid in val_set:
            return "val"
        elif cid in test_set:
            return "test"
        return "unassigned"

    clean_df["split"] = clean_df["cluster_id"].map(assign_split)

    # Attach quarantined rows with split = "quarantine"
    quarantined_df = working_df[quarantined_mask].copy()
    if not quarantined_df.empty:
        quarantined_df["cluster_id"] = "quarantined_conflict"
        quarantined_df["split"] = "quarantine"
        final_df = pd.concat([clean_df, quarantined_df], ignore_index=True)
    else:
        final_df = clean_df

    return final_df

def verify_zero_leakage(
    split_df: pd.DataFrame,
    patient_col: str = "patient_id",
    hash_col: str = "sha256",
) -> Dict[str, Any]:
    """
    Automated assertion verifying zero overlap of patients or image hashes across splits.
    Quarantined rows are excluded from training/evaluation sets.
    """
    valid_splits = split_df[split_df["split"].isin(["train", "val", "test"])]
    train_df = valid_splits[valid_splits["split"] == "train"]
    val_df = valid_splits[valid_splits["split"] == "val"]
    test_df = valid_splits[valid_splits["split"] == "test"]

    train_patients = set(train_df[patient_col].dropna().unique())
    val_patients = set(val_df[patient_col].dropna().unique())
    test_patients = set(test_df[patient_col].dropna().unique())

    # Patient overlap checks
    train_val_patients = train_patients.intersection(val_patients)
    train_test_patients = train_patients.intersection(test_patients)
    val_test_patients = val_patients.intersection(test_patients)

    # Hash overlap checks
    train_hashes = set(train_df[hash_col].dropna().unique()) if hash_col in train_df.columns else set()
    val_hashes = set(val_df[hash_col].dropna().unique()) if hash_col in val_df.columns else set()
    test_hashes = set(test_df[hash_col].dropna().unique()) if hash_col in test_df.columns else set()

    train_val_hashes = train_hashes.intersection(val_hashes)
    train_test_hashes = train_hashes.intersection(test_hashes)
    val_test_hashes = val_hashes.intersection(test_hashes)

    leakage_detected = (
        len(train_val_patients) > 0 or
        len(train_test_patients) > 0 or
        len(val_test_patients) > 0 or
        len(train_val_hashes) > 0 or
        len(train_test_hashes) > 0 or
        len(val_test_hashes) > 0
    )

    report = {
        "is_leakage_free": not leakage_detected,
        "train_patients_count": len(train_patients),
        "val_patients_count": len(val_patients),
        "test_patients_count": len(test_patients),
        "patient_overlap_train_val": list(train_val_patients),
        "patient_overlap_train_test": list(train_test_patients),
        "patient_overlap_val_test": list(val_test_patients),
        "hash_overlap_train_val": len(train_val_hashes),
        "hash_overlap_train_test": len(train_test_hashes),
        "hash_overlap_val_test": len(val_test_hashes),
        "split_counts": split_df["split"].value_counts().to_dict(),
        "class_by_split": pd.crosstab(valid_splits["split"], valid_splits["class_label"]).to_dict(),
    }

    if leakage_detected:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Overlapping patients: train-test={train_test_patients}, "
            f"train-val={train_val_patients}. Overlapping hashes: train-test={len(train_test_hashes)}, "
            f"train-val={len(train_val_hashes)}, val-test={len(val_test_hashes)}."
        )

    return report
