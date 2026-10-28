"""
Dataset ingestion module for knee X-ray images and clinical metadata.
Discovers images, checks integrity, calculates exact and perceptual hashes,
normalizes class labels, links patient metadata, and builds a comprehensive manifest.
Optimized with multi-threaded execution for fast large-scale auditing.
"""
import os
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from PIL import Image
import imagehash

CANONICAL_CLASSES = ["Normal", "Osteopenia", "Osteoporosis"]

LABEL_MAP = {
    "normal": "Normal",
    "Normal": "Normal",
    "NORMAL": "Normal",
    "osteopenia": "Osteopenia",
    "Osteopenia": "Osteopenia",
    "OSTEOPENIA": "Osteopenia",
    "osteoporosis": "Osteoporosis",
    "Osteoporosis": "Osteoporosis",
    "OSTEOPOROSIS": "Osteoporosis",
}

def normalize_label(raw_label: str) -> Optional[str]:
    """Map raw class string to canonical label or return None if unrecognized."""
    if not isinstance(raw_label, str):
        return None
    cleaned = raw_label.strip()
    return LABEL_MAP.get(cleaned, None)

def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of raw file bytes."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def extract_patient_and_aug_info(filename: str) -> Tuple[Optional[str], bool, Optional[int]]:
    """
    Extract base patient ID and augmentation status from filename.
    Examples:
        'N10.JPEG' -> ('N10', False, None)
        'N10_aug_3.jpeg' -> ('N10', True, 3)
        'OP100.jpg' -> ('OP100', False, None)
        'OS5_aug_0.jpeg' -> ('OS5', True, 0)
    """
    stem = os.path.splitext(filename)[0]
    aug_match = re.search(r"^(N\d+|OP\d+|OS\d+)_aug_(\d+)$", stem, re.IGNORECASE)
    if aug_match:
        return aug_match.group(1).upper(), True, int(aug_match.group(2))

    patient_match = re.search(r"^(N\d+|OP\d+|OS\d+)$", stem, re.IGNORECASE)
    if patient_match:
        return patient_match.group(1).upper(), False, None

    return stem, False, None

def inspect_image_file(filepath: str, compute_perceptual: bool = True) -> Dict[str, Any]:
    """Inspect image integrity, dimensions, color mode, and hashes in a single pass."""
    info = {
        "file_size_bytes": os.path.getsize(filepath),
        "is_readable": False,
        "width": None,
        "height": None,
        "channels": None,
        "color_mode": None,
        "sha256": compute_file_hash(filepath),
        "pixel_hash": None,
        "phash": None,
        "error_message": None,
    }
    try:
        with Image.open(filepath) as img:
            img.load()
            info["width"], info["height"] = img.size
            info["color_mode"] = img.mode
            info["channels"] = len(img.getbands())
            info["is_readable"] = True
            if compute_perceptual:
                info["phash"] = str(imagehash.phash(img))
                gray = img.convert("L")
                info["pixel_hash"] = hashlib.sha256(np.array(gray).tobytes()).hexdigest()
    except Exception as e:
        info["error_message"] = str(e)
    return info

def _process_single_image(args: Tuple[str, str, str, str, str, str, Optional[str]]) -> Dict[str, Any]:
    """Helper worker for multi-threaded ingestion."""
    file_path, file, source_dataset, source_version, folder_name, norm_class, raw_split = args
    patient_id, is_aug, aug_idx = extract_patient_and_aug_info(file)
    img_info = inspect_image_file(file_path, compute_perceptual=True)
    record = {
        "image_path": file_path.replace("\\", "/"),
        "filename": file,
        "source_dataset": source_dataset,
        "source_version": source_version,
        "source_image_id": file,
        "patient_id": patient_id,
        "is_augmented": is_aug,
        "augmentation_index": aug_idx,
        "raw_class": folder_name,
        "class_label": norm_class,
        "label_source": "QUS_tscore" if source_dataset == "mendeley" else ("kaggle_gobara_published" if source_dataset == "kaggle_gobara" else "zenodo_reorganized_QUS"),
        "label_confidence_or_status": "documented" if source_dataset == "mendeley" else "derived",
        **img_info,
    }
    if raw_split is not None:
        record["zenodo_raw_split"] = raw_split
    return record

def ingest_mendeley_dataset(mendeley_dir: str, max_workers: int = 8) -> pd.DataFrame:
    """Ingest images from Mendeley raw directory structure."""
    if not os.path.exists(mendeley_dir):
        return pd.DataFrame()

    tasks = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for root, _, files in os.walk(mendeley_dir):
        folder_name = os.path.basename(root)
        norm_class = normalize_label(folder_name)
        if norm_class is None:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                file_path = os.path.join(root, file)
                tasks.append((file_path, file, "mendeley", "v2", folder_name, norm_class, None))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(executor.map(_process_single_image, tasks))

    return pd.DataFrame(records)

def ingest_kaggle_dataset(kaggle_dir: str, max_workers: int = 8) -> pd.DataFrame:
    """Ingest images from Kaggle (Mohamed Gobara) raw directory structure."""
    if not os.path.exists(kaggle_dir):
        return pd.DataFrame()

    tasks = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for root, _, files in os.walk(kaggle_dir):
        folder_name = os.path.basename(root)
        norm_class = normalize_label(folder_name)
        if norm_class is None:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                file_path = os.path.join(root, file)
                tasks.append((file_path, file, "kaggle_gobara", "v1_2024", folder_name, norm_class, None))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(executor.map(_process_single_image, tasks))

    return pd.DataFrame(records)

def ingest_zenodo_dataset(zenodo_dir: str, max_workers: int = 8) -> pd.DataFrame:
    """Ingest images from Zenodo raw directory structure."""
    if not os.path.exists(zenodo_dir):
        return pd.DataFrame()

    tasks = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for root, _, files in os.walk(zenodo_dir):
        folder_name = os.path.basename(root)
        norm_class = normalize_label(folder_name)
        if norm_class is None:
            continue
        rel = os.path.relpath(root, zenodo_dir).replace("\\", "/")
        parts = rel.split("/")
        raw_split = parts[0] if len(parts) > 1 else "unknown"
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                file_path = os.path.join(root, file)
                tasks.append((file_path, file, "zenodo", "20053147", folder_name, norm_class, raw_split))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(executor.map(_process_single_image, tasks))

    return pd.DataFrame(records)

def ingest_clinical_metadata(excel_path: str) -> pd.DataFrame:
    """Ingest and normalize patient details from Excel."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Metadata file not found: {excel_path}")
    df = pd.read_excel(excel_path)
    # Drop rows without Patient Id
    if "Patient Id" in df.columns:
        df = df.dropna(subset=["Patient Id"]).copy()
        df["patient_id"] = df["Patient Id"].astype(str).str.strip().str.upper()
    return df

