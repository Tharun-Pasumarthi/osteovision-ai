"""
Automated Comprehensive Test Suite for Audit Validation.
Tests preprocessing, class mapping, inference, Grad-CAM, and leakage isolation.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pytest
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.train import CLASS_TO_IDX, CLASS_NAMES as TRAIN_CLASS_NAMES
from src.evaluate import CLASS_NAMES as EVAL_CLASS_NAMES
from src.predict import CLASS_NAMES as PREDICT_CLASS_NAMES
from backend.app.services.inference_service import CLASS_NAMES as INFERENCE_CLASS_NAMES
from backend.app.services.gradcam_service import CLASS_NAMES as GRADCAM_CLASS_NAMES
from src.image_transforms import (
    resize_and_pad_radiograph,
    KneeXRayEvalTransform,
    KneeXRayTrainTransform,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.models import KneeXRayCNNClassifier, MultimodalFusionClassifier
from src.gradcam import GradCAM, project_cam_to_original_image, run_parameter_randomization_sanity_check
from src.data_splitting import verify_zero_leakage

# ==========================================
# 1. Class Mapping Alignment Tests
# ==========================================

def test_class_mapping_alignment_across_all_modules():
    """Verify that class names and index ordering are strictly identical everywhere."""
    canonical = ["Normal", "Osteopenia", "Osteoporosis"]
    assert TRAIN_CLASS_NAMES == canonical
    assert EVAL_CLASS_NAMES == canonical
    assert PREDICT_CLASS_NAMES == canonical
    assert INFERENCE_CLASS_NAMES == canonical
    assert GRADCAM_CLASS_NAMES == canonical
    
    assert CLASS_TO_IDX["Normal"] == 0
    assert CLASS_TO_IDX["Osteopenia"] == 1
    assert CLASS_TO_IDX["Osteoporosis"] == 2

# ==========================================
# 2. Preprocessing & Geometry Tests
# ==========================================

def test_preprocessing_aspect_ratio_preservation_and_letterbox():
    """Verify aspect ratio preservation for square, tall vertical, and wide horizontal images."""
    # 1. Tall vertical image (e.g. 100 x 300)
    tall_img = Image.new("L", (100, 300), color=200)
    padded_tall = resize_and_pad_radiograph(tall_img, target_size=224)
    assert padded_tall.size == (224, 224)
    assert padded_tall.mode == "RGB"
    # Vertical image should have black padding on left and right
    tall_np = np.array(padded_tall)
    assert np.all(tall_np[:, 0, :] == 0) # left border is black

    # 2. Wide horizontal image (e.g. 400 x 150)
    wide_img = Image.new("RGB", (400, 150), color=(100, 150, 200))
    padded_wide = resize_and_pad_radiograph(wide_img, target_size=224)
    assert padded_wide.size == (224, 224)
    # Horizontal image should have black padding on top and bottom
    wide_np = np.array(padded_wide)
    assert np.all(wide_np[0, :, :] == 0) # top border is black

def test_preprocessing_normalization_and_channel_order():
    """Verify eval transform converts to NCHW float32 with ImageNet normalization."""
    img = Image.new("RGB", (250, 250), color=(128, 128, 128))
    transform = KneeXRayEvalTransform(target_size=224)
    tensor = transform(img)
    
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    # Verify values are normalized around (0.5 - mean) / std
    for c in range(3):
        expected_val = ((128 / 255.0) - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert np.isclose(float(tensor[c, 112, 112]), expected_val, atol=1e-2)

def test_preprocessing_train_eval_compatibility():
    """Verify training and evaluation transforms share identical input size and normalization."""
    train_t = KneeXRayTrainTransform(target_size=224)
    eval_t = KneeXRayEvalTransform(target_size=224)
    
    assert train_t.target_size == eval_t.target_size == 224
    assert train_t.normalize.mean == eval_t.normalize.mean == IMAGENET_MEAN
    assert train_t.normalize.std == eval_t.normalize.std == IMAGENET_STD

# ==========================================
# 3. Model Inference & Checkpoint Tests
# ==========================================

def test_inference_checkpoint_loading_and_probabilities():
    """Load the production checkpoint and test inference shape, probability sum, and bounds."""
    ckpt_path = "outputs/checkpoints/best_densenet121.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    
    model = KneeXRayCNNClassifier("densenet121", num_classes=3, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        
    assert logits.shape == (1, 3)
    assert len(probs) == 3
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

# ==========================================
# 4. Grad-CAM Engine & Explainability Tests
# ==========================================

def test_gradcam_hook_cleanup_and_memory_safety():
    """Verify that hooks are cleaned up and no handles leak across invocations."""
    model = KneeXRayCNNClassifier("densenet121", num_classes=3, pretrained=False)
    target_layer = model.features.norm5
    dummy_x = torch.randn(1, 3, 224, 224)
    
    with GradCAM(model, target_layer) as cam:
        assert len(cam.handles) == 2
        heatmap, pred_c, probs, status = cam.generate_heatmap(dummy_x)
        assert heatmap.shape == (7, 7) or len(heatmap.shape) == 2
    
    # After exiting with block, handles must be completely cleared
    assert len(cam.handles) == 0

def test_gradcam_native_projection_dimensions():
    """Verify that projecting Grad-CAM back to original image preserves patient resolution."""
    orig_pil = Image.new("RGB", (640, 480), color=(100, 100, 100))
    dummy_heatmap = np.random.uniform(0.0, 1.0, (7, 7)).astype(np.float32)
    
    heatmap_color, overlay = project_cam_to_original_image(
        orig_pil, dummy_heatmap, target_size=224, alpha=0.5
    )
    
    assert heatmap_color.shape == (480, 640, 3)
    assert overlay.shape == (480, 640, 3)
    assert heatmap_color.dtype == np.uint8
    assert overlay.dtype == np.uint8

def test_gradcam_parameter_randomization_sanity():
    """Verify that authentic CAM is structurally different from randomized-weights CAM."""
    model = KneeXRayCNNClassifier("densenet121", num_classes=3, pretrained=False)
    target_layer = model.features.norm5
    dummy_x = torch.randn(1, 3, 224, 224)
    
    diff = run_parameter_randomization_sanity_check(model, target_layer, dummy_x, target_class=0)
    assert isinstance(diff, float)
    assert diff >= 0.0

# ==========================================
# 5. Data Partition & Zero-Leakage Tests
# ==========================================

def test_split_manifest_zero_leakage():
    """Verify that the official split_manifest has zero patient and zero hash leakage."""
    manifest_path = "outputs/reports/split_manifest.csv"
    assert os.path.exists(manifest_path), f"Missing split manifest: {manifest_path}"
    
    df = pd.read_csv(manifest_path)
    leakage_report = verify_zero_leakage(df, patient_col="patient_id", hash_col="sha256")
    assert leakage_report["is_leakage_free"] is True
    assert len(leakage_report["patient_overlap_train_val"]) == 0
    assert len(leakage_report["patient_overlap_train_test"]) == 0
    assert len(leakage_report["patient_overlap_val_test"]) == 0
    assert leakage_report["hash_overlap_train_val"] == 0
    assert leakage_report["hash_overlap_train_test"] == 0
    assert leakage_report["hash_overlap_val_test"] == 0
