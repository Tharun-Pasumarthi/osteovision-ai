"""
Unit tests for multimodal knee X-ray osteoporosis classification pipeline.
Tests class mapping, transforms, leakage isolation, models, inference, and explainability.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pytest
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.data_ingestion import normalize_label, compute_file_hash, extract_patient_and_aug_info
from src.deduplication import detect_exact_duplicates, detect_label_conflicts
from src.metadata_preprocessing import ClinicalFeaturePreprocessor
from src.data_splitting import build_patient_clusters, create_patient_grouped_splits, verify_zero_leakage
from src.image_transforms import resize_and_pad_radiograph, KneeXRayEvalTransform, KneeXRayTrainTransform
from src.models import KneeXRayCNNClassifier, MultimodalFusionClassifier, count_parameters
from src.predict import predict_single_image, predict_multimodal
from src.gradcam import GradCAM

def test_class_label_normalization():
    """Verify that case variations map cleanly to canonical 3 classes."""
    assert normalize_label("normal") == "Normal"
    assert normalize_label("Normal") == "Normal"
    assert normalize_label("osteopenia") == "Osteopenia"
    assert normalize_label("Osteopenia") == "Osteopenia"
    assert normalize_label("osteoporosis") == "Osteoporosis"
    assert normalize_label("Osteoporosis") == "Osteoporosis"
    assert normalize_label("unknown_class") is None
    assert normalize_label(None) is None

def test_extract_patient_and_aug_info():
    """Verify regex extraction of patient ID and augmentation flags."""
    pid, is_aug, idx = extract_patient_and_aug_info("N10.JPEG")
    assert pid == "N10" and is_aug is False and idx is None

    pid, is_aug, idx = extract_patient_and_aug_info("N10_aug_4.jpeg")
    assert pid == "N10" and is_aug is True and idx == 4

    pid, is_aug, idx = extract_patient_and_aug_info("OP123.jpg")
    assert pid == "OP123" and is_aug is False

    pid, is_aug, idx = extract_patient_and_aug_info("OS9_aug_1.jpeg")
    assert pid == "OS9" and is_aug is True and idx == 1

def test_image_transforms_and_shape():
    """Verify that letterbox padding and eval transforms preserve proportions and produce (3, 224, 224)."""
    # Create synthetic rectangular grayscale image
    dummy_img = Image.new("L", (400, 200), color=128)
    padded = resize_and_pad_radiograph(dummy_img, target_size=224)
    assert padded.size == (224, 224)

    eval_transform = KneeXRayEvalTransform(target_size=224)
    tensor = eval_transform(dummy_img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

    train_transform = KneeXRayTrainTransform(target_size=224)
    train_tensor = train_transform(dummy_img)
    assert train_tensor.shape == (3, 224, 224)

def test_patient_cluster_and_leakage_isolation():
    """Verify that patients sharing image hashes are clustered and isolated across splits."""
    dummy_data = pd.DataFrame([
        {"patient_id": "P1", "sha256": "hash_A", "class_label": "Normal"},
        {"patient_id": "P2", "sha256": "hash_A", "class_label": "Normal"}, # Duplicate image
        {"patient_id": "P3", "sha256": "hash_B", "class_label": "Osteopenia"},
        {"patient_id": "P4", "sha256": "hash_C", "class_label": "Osteoporosis"},
        {"patient_id": "P5", "sha256": "hash_D", "class_label": "Osteopenia"},
        {"patient_id": "P6", "sha256": "hash_E", "class_label": "Normal"},
    ])

    clusters = build_patient_clusters(dummy_data, patient_col="patient_id", hash_col="sha256")
    # P1 and P2 must be in the exact same cluster
    assert clusters["P1"] == clusters["P2"]

    split_df = create_patient_grouped_splits(
        dummy_data,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=42,
    )
    # P1 and P2 must have the exact same split
    p1_split = split_df[split_df["patient_id"] == "P1"]["split"].iloc[0]
    p2_split = split_df[split_df["patient_id"] == "P2"]["split"].iloc[0]
    assert p1_split == p2_split

    # Leakage check must pass
    report = verify_zero_leakage(split_df, patient_col="patient_id", hash_col="sha256")
    assert report["is_leakage_free"] is True

def test_label_conflict_quarantine():
    """Verify that identical images with conflicting labels are quarantined."""
    conflict_data = pd.DataFrame([
        {"patient_id": "P1", "sha256": "conflict_hash", "class_label": "Normal", "image_path": "path1"},
        {"patient_id": "P2", "sha256": "conflict_hash", "class_label": "Osteopenia", "image_path": "path2"},
        {"patient_id": "P3", "sha256": "clean_hash", "class_label": "Osteoporosis", "image_path": "path3"},
    ])
    conflicts = detect_label_conflicts(conflict_data, group_col="sha256")
    assert len(conflicts) == 2
    assert "quarantined_for_investigation" in conflicts["action"].values

def test_metadata_preprocessor_training_only_fit():
    """Verify clinical preprocessor transforms features and handles missing values."""
    train_data = pd.DataFrame([
        {"patient_id": "P1", "Age": 60, "height  (meter)": 1.60, "Weight (KG) ": 60.0, "Gender": "female", "Joint Pain:": "yes", "Diabetic": "no"},
        {"patient_id": "P2", "Age": 70, "height  (meter)": 1.70, "Weight (KG) ": 75.0, "Gender": "male", "Joint Pain:": "no", "Diabetic": "yes"},
    ])
    preproc = ClinicalFeaturePreprocessor()
    preproc.fit(train_data)
    assert preproc.is_fitted is True

    test_data = pd.DataFrame([
        {"patient_id": "P3", "Age": np.nan, "height  (meter)": 1.65, "Weight (KG) ": 65.0, "Gender": "female", "Joint Pain:": "yes", "Diabetic": "no"},
    ])
    feat_mat, feat_names = preproc.transform(test_data)
    assert feat_mat.shape == (1, len(feat_names))
    # No NaNs allowed in output
    assert not np.isnan(feat_mat).any()

def test_model_output_shape_and_probability_sum():
    """Verify that CNN and Multimodal models output 3 class logits and probabilities sum to 1.0."""
    model = KneeXRayCNNClassifier("densenet121", num_classes=3, pretrained=False)
    dummy_x = torch.randn(2, 3, 224, 224)
    logits = model(dummy_x)
    assert logits.shape == (2, 3)

    probs = torch.softmax(logits, dim=1).detach().numpy()
    np.testing.assert_allclose(np.sum(probs, axis=1), [1.0, 1.0], atol=1e-5)

def test_multimodal_missing_clinical_fallback():
    """Verify multimodal model functions when clinical features are missing / None."""
    multimodal = MultimodalFusionClassifier(
        clinical_input_dim=19,
        num_classes=3,
    )
    dummy_img = torch.randn(2, 3, 224, 224)
    
    # 1. With clinical vector
    dummy_clin = torch.randn(2, 19)
    out_with = multimodal(dummy_img, dummy_clin)
    assert out_with.shape == (2, 3)

    # 2. Without clinical vector (fallback)
    out_fallback = multimodal(dummy_img, None)
    assert out_fallback.shape == (2, 3)

def test_gradcam_generation():
    """Verify Grad-CAM produces normalized heatmap of shape (H, W)."""
    model = KneeXRayCNNClassifier("densenet121", num_classes=3, pretrained=False)
    target_layer = model.features.norm5
    cam = GradCAM(model, target_layer)
    dummy_x = torch.randn(1, 3, 224, 224)
    heatmap, pred_class, probs, status = cam.generate_heatmap(dummy_x)

    assert status == "success"
    assert heatmap.shape == (7, 7) or len(heatmap.shape) == 2
    assert pred_class in [0, 1, 2]
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
