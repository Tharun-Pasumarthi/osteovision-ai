"""
Backend API Tests using FastAPI TestClient.
Tests endpoints, schemas, real inference, Grad-CAM, error handling, and validations.
"""
import io
import json
import os
import sys
from pathlib import Path

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.model_registry import registry

@pytest.fixture(scope="module")
def client():
    # Trigger startup lifespan
    with TestClient(app) as test_client:
        yield test_client

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert len(data["models_loaded"]) > 0

def test_models_list_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 5
    model_ids = [m["id"] for m in data["models"]]
    assert "densenet121" in model_ids
    assert "multimodal_fusion" in model_ids
    assert "efficientnet_b0" in model_ids

def test_metrics_endpoint(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0
    # Ensure best model is noted
    assert "best_validation_model" in data

def test_dataset_summary_endpoint(client):
    response = client.get("/api/dataset-summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_images_discovered"] > 0
    assert "class_distribution" in data
    assert "sources" in data
    assert "provenance_notes" in data

def test_clinical_importance_endpoint(client):
    response = client.get("/api/clinical-importance")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert len(data["features"]) > 0

def test_sample_images_endpoint(client):
    response = client.get("/api/sample-images")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0

def test_predict_real_image_image_only(client):
    # Create valid synthetic grayscale test radiograph
    img = Image.new("RGB", (300, 300), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict",
        files={"file": ("test_knee.jpg", buf, "image/jpeg")},
        data={"model_id": "densenet121"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] in ["Normal", "Osteopenia", "Osteoporosis"]
    assert "probabilities" in data
    assert sum(data["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert data["inference_duration_ms"] > 0
    assert data["model_id"] == "densenet121"

def test_predict_multimodal_with_clinical(client):
    img = Image.new("RGB", (300, 300), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    clinical_payload = json.dumps({
        "age": 65,
        "height_m": 1.55,
        "weight_kg": 52.0,
        "gender": "female",
        "joint_pain": True,
        "smoker": False,
        "diabetic": False,
        "hypothyroidism": True,
        "history_of_fracture": True,
        "family_history_osteoporosis": True,
        "menopause_age": 45,
        "max_walking_km": 1.0,
        "obesity_category": "normal weight"
    })

    response = client.post(
        "/api/predict",
        files={"file": ("test_knee_mm.jpg", buf, "image/jpeg")},
        data={
            "model_id": "multimodal_fusion",
            "clinical_data": clinical_payload,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] in ["Normal", "Osteopenia", "Osteoporosis"]
    assert data["model_id"] == "multimodal_fusion"
    assert data["clinical_features_used"] is not None
    assert len(data["clinical_features_used"]) == 13

def test_predict_invalid_extension(client):
    buf = io.BytesIO(b"fake data")
    response = client.post(
        "/api/predict",
        files={"file": ("fake.txt", buf, "text/plain")},
        data={"model_id": "densenet121"},
    )
    assert response.status_code == 400

def test_predict_corrupted_image(client):
    buf = io.BytesIO(b"this is not an image file")
    response = client.post(
        "/api/predict",
        files={"file": ("corrupt.jpg", buf, "image/jpeg")},
        data={"model_id": "densenet121"},
    )
    assert response.status_code == 400

def test_gradcam_generation_endpoint(client):
    img = Image.new("RGB", (250, 250), color=(150, 150, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/gradcam",
        files={"file": ("knee_xray.png", buf, "image/png")},
        data={"model_id": "densenet121", "alpha": 0.5},
    )
    assert response.status_code == 200
    data = response.json()
    assert "original_image_base64" in data
    assert "heatmap_base64" in data
    assert "overlay_base64" in data
    assert data["alpha"] == 0.5
    assert data["target_class"] in ["Normal", "Osteopenia", "Osteoporosis"]
