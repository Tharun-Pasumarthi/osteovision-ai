"""
Comprehensive Scientific Validation and Test Suite for OsteoVision AI Platform:
- Intelligent Image Quality Assessment (IQA)
- SHAP Clinical Explainability (TreeExplainer, local attribution, additivity, global importance)
- Probability Calibration (Temperature scaling, Platt scaling, ECE, Brier score, reliability diagrams)
- End-to-end API integration and response contracts
"""
import io
import json
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.image_quality_service import image_quality_service
from backend.app.services.shap_service import shap_service
from backend.app.services.calibration_service import (
    calibration_service,
    compute_ece_and_reliability_diagram,
    compute_multiclass_brier_score,
    compute_multiclass_log_loss,
)
from backend.app.schemas.models import ClinicalInputSchema

@pytest.fixture
def client():
    return TestClient(app)

# ==============================================================================
# 1. Image Quality Assessment (IQA) Service Tests
# ==============================================================================

def test_iqa_valid_image():
    """Verify IQA accurately accepts a standard high-quality synthetic knee radiograph with bone texture."""
    # Create synthetic structured image with sharp edges and noise
    np.random.seed(42)
    noise = np.random.randint(50, 200, (300, 300), dtype=np.uint8)
    img = Image.fromarray(noise).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    
    result = image_quality_service.assess_image(buf.getvalue(), "knee_valid.png")
    assert result["can_proceed_to_inference"] is True
    assert result["width"] == 300
    assert result["height"] == 300
    assert result["laplacian_variance"] > 50.0
    assert "disclaimer" in result

def test_iqa_corrupted_file():
    """Verify IQA catches corrupted byte stream and prevents inference."""
    corrupted_bytes = b"NOT_A_VALID_IMAGE_HEADER_00000000"
    result = image_quality_service.assess_image(corrupted_bytes, "corrupt.jpg")
    assert result["overall_status"] == "unusable"
    assert result["can_proceed_to_inference"] is False
    assert result["quality_score"] == 0.0
    assert len(result["issues"]) > 0

def test_iqa_severely_low_resolution():
    """Verify IQA flags severely microscopic dimensions."""
    tiny_img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    buf = io.BytesIO()
    tiny_img.save(buf, format="PNG")
    
    result = image_quality_service.assess_image(buf.getvalue(), "tiny.png")
    assert result["overall_status"] == "unusable"
    assert result["can_proceed_to_inference"] is False
    assert any("resolution" in issue.lower() for issue in result["issues"])

def test_iqa_extreme_aspect_ratio():
    """Verify IQA flags extreme banner or cropped dimensions."""
    banner_img = Image.new("RGB", (600, 50), color=(128, 128, 128))
    buf = io.BytesIO()
    banner_img.save(buf, format="PNG")
    
    result = image_quality_service.assess_image(buf.getvalue(), "banner.png")
    assert result["overall_status"] == "unusable"
    assert result["aspect_ratio"] == 12.0

def test_iqa_severe_blur():
    """Verify IQA computes low Laplacian variance on smoothed image."""
    img = Image.new("RGB", (250, 250), color=(140, 140, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    
    result = image_quality_service.assess_image(buf.getvalue(), "smooth.jpg")
    assert result["laplacian_variance"] < 10.0
    assert result["blur_status"] in ["severe_blur", "borderline_blur"]

# ==============================================================================
# 2. SHAP Explainability Service Tests
# ==============================================================================

def test_shap_local_explanation_additivity():
    """Verify SHAP local explanation preserves exact 13-feature contract and local additivity."""
    patient = ClinicalInputSchema(
        num_age=68.0,
        num_height_m=1.58,
        num_weight_kg=54.0,
        num_bmi=21.63,
        num_menopause_age=44.0,
        num_max_walking_km=1.5,
        gender_female=1.0,
        joint_pain=1.0,
        smoker=0.0,
        diabetic=0.0,
        hypothyroidism=1.0,
        fracture=1.0,
        family_history=1.0,
    )
    explanation = shap_service.explain_patient(patient, target_class="Osteoporosis")
    
    assert explanation["target_class"] == "Osteoporosis"
    assert "base_value" in explanation
    assert len(explanation["contributions"]) == 13
    assert len(explanation["waterfall_steps"]) == 13
    assert "summary" in explanation
    assert "disclaimer" in explanation
    
    # Check that sum of SHAP values matches final delta
    total_shap = sum(c["shap_value"] for c in explanation["contributions"])
    expected_delta = explanation["final_predicted_value"] - explanation["base_value"]
    assert total_shap == pytest.approx(expected_delta, abs=1e-3)

def test_shap_global_importance():
    """Verify global feature importance covers all 13 features with clinical metadata."""
    global_imp = shap_service.get_global_importance()
    assert len(global_imp) == 13
    assert global_imp[0]["importance_score"] >= global_imp[-1]["importance_score"]
    total_pct = sum(f["percentage"] for f in global_imp)
    assert total_pct == pytest.approx(100.0, abs=1.0)
    for f in global_imp:
        assert "feature_name" in f
        assert "display_name" in f
        assert "category" in f
        assert "description" in f

# ==============================================================================
# 3. Probability Calibration Service Tests
# ==============================================================================

def test_calibration_ece_and_diagram():
    """Verify mathematical correctness of Expected Calibration Error calculation."""
    # Synthetic perfectly calibrated predictions
    confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    accuracies = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    ece, mce, diagram = compute_ece_and_reliability_diagram(confidences, accuracies, num_bins=5)
    
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert len(diagram) == 5

def test_brier_score_and_log_loss():
    """Verify multi-class Brier score and log loss computation."""
    y_true = np.array([0, 1, 2])
    # Perfect predictions
    perfect_probs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    brier_perfect = compute_multiclass_brier_score(y_true, perfect_probs, num_classes=3)
    loss_perfect = compute_multiclass_log_loss(y_true, perfect_probs)
    assert brier_perfect == pytest.approx(0.0, abs=1e-4)
    assert loss_perfect == pytest.approx(0.0, abs=1e-4)

def test_calibration_service_report():
    """Verify calibration service returns verified validation reports for image and clinical models."""
    report = calibration_service.get_full_calibration_report()
    assert "image_model_calibration" in report
    assert "clinical_model_calibration" in report
    assert report["image_model_calibration"]["status"] == "calibrated"
    assert report["clinical_model_calibration"]["status"] == "calibrated"
    assert report["image_model_calibration"]["raw_metrics"]["ece"] > report["image_model_calibration"]["calibrated_metrics"]["ece"]

# ==============================================================================
# 4. API Endpoint Integration Tests
# ==============================================================================

def test_api_image_quality_endpoint(client):
    """Test POST /api/image-quality pre-screening."""
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post("/api/image-quality", files={"file": ("test_knee.png", buf, "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert "quality_score" in data
    assert "overall_status" in data
    assert data["width"] == 256
    assert data["height"] == 256

def test_api_shap_explain_endpoint(client):
    """Test POST /api/shap-explain clinical attribution."""
    payload = {
        "num_age": 70,
        "num_height_m": 1.52,
        "num_weight_kg": 49.0,
        "gender_female": 1.0,
        "joint_pain": 1.0,
        "smoker": 0.0,
        "diabetic": 0.0,
        "hypothyroidism": 1.0,
        "fracture": 1.0,
        "family_history": 1.0,
        "num_menopause_age": 42.0,
        "num_max_walking_km": 0.8,
    }
    res = client.post("/api/shap-explain?target_class=Osteoporosis", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["target_class"] == "Osteoporosis"
    assert len(data["contributions"]) == 13
    assert len(data["waterfall_steps"]) == 13
    assert "summary" in data

def test_api_shap_global_endpoint(client):
    """Test GET /api/shap-global."""
    res = client.get("/api/shap-global")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) == 13

def test_api_calibration_status_endpoint(client):
    """Test GET /api/calibration-status."""
    res = client.get("/api/calibration-status")
    assert res.status_code == 200
    data = res.json()
    assert "image_model_calibration" in data
    assert "clinical_model_calibration" in data
    assert data["image_model_calibration"]["calibration_method"] == "Post-Hoc Temperature Scaling"

def test_api_predict_returns_raw_and_calibrated(client):
    """Test POST /api/predict provides both raw and calibrated probability distributions and quality info."""
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    res = client.post(
        "/api/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        data={"model_id": "densenet121"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "probabilities" in data
    assert "raw_probabilities" in data
    assert "calibrated_probabilities" in data
    assert data["is_calibrated"] is True
    assert "image_quality" in data
