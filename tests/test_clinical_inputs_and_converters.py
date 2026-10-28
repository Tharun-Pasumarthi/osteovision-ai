"""
Comprehensive automated tests for clinical inputs, unit conversions, BMI validation,
menopausal status semantics, and walking distance estimation.
"""
import pytest
import math
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.models import ClinicalInputSchema
from backend.app.services.inference_service import inference_service, CLASS_NAMES

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

# =====================================================================
# 1. Height Conversion & Normalization Tests
# =====================================================================

def test_height_units_conversion_metric():
    """Verify standard height unit conversions to canonical metres."""
    # 163 cm -> 1.63 m
    cm = 163.0
    m_from_cm = round(cm / 100.0, 4)
    assert m_from_cm == 1.63

    schema = ClinicalInputSchema(
        num_age=50,
        num_height_m=m_from_cm,
        num_weight_kg=60,
    )
    assert schema.num_height_m == 1.63

def test_height_units_conversion_imperial():
    """Verify imperial feet & inches to metres conversion."""
    # 5 ft 8 in -> 68 inches * 0.0254 = 1.7272 m
    feet = 5
    inches = 8
    total_inches = feet * 12 + inches
    metres = round(total_inches * 0.0254, 4)
    assert metres == 1.7272

    schema = ClinicalInputSchema(
        num_age=50,
        num_height_m=round(metres, 2),
        num_weight_kg=65,
    )
    assert schema.num_height_m == 1.73

def test_height_out_of_range():
    """Verify height outside [0.5, 2.5] is rejected."""
    with pytest.raises(Exception):
        ClinicalInputSchema(num_age=50, num_height_m=0.3, num_weight_kg=60)
    with pytest.raises(Exception):
        ClinicalInputSchema(num_age=50, num_height_m=2.8, num_weight_kg=60)

# =====================================================================
# 2. BMI Calculation & Validation Tests
# =====================================================================

def test_bmi_auto_calculation_accuracy():
    """Verify BMI formula: weight_kg / (height_m ** 2)."""
    # 64 kg, 1.58 m -> 64 / 2.4964 = 25.63689 -> rounded to 25.64
    schema = ClinicalInputSchema(
        num_age=60,
        num_height_m=1.58,
        num_weight_kg=64.0,
    )
    assert schema.num_bmi == 25.64

def test_bmi_preserves_explicit_valid_value():
    """Verify that an explicitly supplied valid BMI is preserved."""
    schema = ClinicalInputSchema(
        num_age=60,
        num_height_m=1.58,
        num_weight_kg=64.0,
        num_bmi=26.10,
    )
    assert schema.num_bmi == 26.10

def test_bmi_zero_or_empty_is_recalculated_not_stored_as_zero():
    """Verify that zero BMI is NOT treated as a true 0, but recalculated from height and weight."""
    schema = ClinicalInputSchema(
        num_age=60,
        num_height_m=1.60,
        num_weight_kg=64.0,
        num_bmi=0.0,
    )
    # Recalculated: 64 / (1.60^2) = 25.0
    assert schema.num_bmi == 25.0

def test_bmi_out_of_range_rejected():
    """Verify BMI outside [10, 70] raises validation error."""
    with pytest.raises(Exception):
        ClinicalInputSchema(num_age=50, num_bmi=5.0)
    with pytest.raises(Exception):
        ClinicalInputSchema(num_age=50, num_bmi=85.0)

# =====================================================================
# 3. Menopausal Status & Onset Age Semantics
# =====================================================================

def test_menopause_age_male_forced_none():
    """Verify menopause age for male subjects (gender_female=0.0) is always set to None."""
    schema = ClinicalInputSchema(
        num_age=65,
        gender_female=0.0,
        num_menopause_age=48.0,  # Should be overridden to None for males
    )
    assert schema.num_menopause_age is None

def test_menopause_age_zero_sanitized_to_none():
    """Verify menopause age entered as 0 or empty is sanitized to None rather than failing or fabricating zero."""
    schema = ClinicalInputSchema(
        num_age=65,
        gender_female=1.0,
        num_menopause_age=0.0,
    )
    assert schema.num_menopause_age is None

def test_menopause_age_empty_string_sanitized_to_none():
    """Verify menopause age passed as empty string is coerced to None."""
    raw_data = {
        "num_age": 62,
        "gender_female": 1.0,
        "num_menopause_age": "",
    }
    schema = ClinicalInputSchema(**raw_data)
    assert schema.num_menopause_age is None

def test_menopause_age_valid_female_preserved():
    """Verify valid menopause onset age for females is preserved."""
    schema = ClinicalInputSchema(
        num_age=65,
        gender_female=1.0,
        num_menopause_age=46.0,
    )
    assert schema.num_menopause_age == 46.0

def test_menopause_age_out_of_range_rejected():
    """Verify menopause onset age outside [20, 80] is rejected."""
    with pytest.raises(Exception):
        ClinicalInputSchema(num_age=65, gender_female=1.0, num_menopause_age=15.0)
    with pytest.raises(Exception):
        ClinicalInputSchema(num_age=65, gender_female=1.0, num_menopause_age=90.0)

# =====================================================================
# 4. Walking Distance & Step Conversion Tests
# =====================================================================

def test_walking_distance_metres_to_km():
    """Verify conversion of metres to canonical kilometres."""
    metres = 2500
    km = metres / 1000.0
    assert km == 2.5

    schema = ClinicalInputSchema(
        num_age=50,
        num_max_walking_km=km,
    )
    assert schema.num_max_walking_km == 2.5

def test_walking_distance_steps_to_km_estimation():
    """Verify steps to km estimation with assumed 0.75m stride."""
    steps = 4000
    stride_m = 0.75
    estimated_km = round((steps * stride_m) / 1000.0, 2)
    assert estimated_km == 3.0

    schema = ClinicalInputSchema(
        num_age=50,
        num_max_walking_km=estimated_km,
    )
    assert schema.num_max_walking_km == 3.0

def test_walking_distance_out_of_range():
    """Verify walking distance outside [0, 50] is rejected."""
    with pytest.raises(Exception):
        ClinicalInputSchema(num_age=50, num_max_walking_km=-1.0)
    with pytest.raises(Exception):
        ClinicalInputSchema(num_age=50, num_max_walking_km=60.0)

# =====================================================================
# 5. API Endpoint Regression & Error Handling Tests
# =====================================================================

def test_api_predict_clinical_valid_female_postmenopausal(client):
    """Test clinical prediction endpoint with full female postmenopausal profile."""
    payload = {
        "num_age": 62,
        "num_height_m": 1.58,
        "num_weight_kg": 64.0,
        "num_menopause_age": 46.0,
        "num_max_walking_km": 2.5,
        "gender_female": 1.0,
        "joint_pain": 1.0,
        "smoker": 0.0,
        "diabetic": 0.0,
        "hypothyroidism": 0.0,
        "fracture": 0.0,
        "family_history": 1.0,
    }
    response = client.post("/api/predict-clinical", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] in CLASS_NAMES
    assert "probabilities" in data
    assert sum(data["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert data["input_type"] == "clinical_only"

def test_api_predict_clinical_male_with_null_menopause(client):
    """Test clinical prediction endpoint with male profile where menopause age is null."""
    payload = {
        "num_age": 70,
        "num_height_m": 1.72,
        "num_weight_kg": 75.0,
        "num_menopause_age": None,
        "num_max_walking_km": 4.0,
        "gender_female": 0.0,
        "joint_pain": 0.0,
        "smoker": 1.0,
        "diabetic": 0.0,
        "hypothyroidism": 0.0,
        "fracture": 1.0,
        "family_history": 0.0,
    }
    response = client.post("/api/predict-clinical", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] in CLASS_NAMES

def test_api_predict_clinical_auto_calculates_bmi_when_omitted(client):
    """Test clinical prediction endpoint computes BMI automatically when omitted."""
    payload = {
        "num_age": 55,
        "num_height_m": 1.60,
        "num_weight_kg": 64.0,
        # num_bmi omitted
        "num_max_walking_km": 3.0,
        "gender_female": 1.0,
    }
    response = client.post("/api/predict-clinical", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] in CLASS_NAMES

def test_api_predict_clinical_empty_payload_fails_validation(client):
    """Test that malformed/empty payload returns 422 Unprocessable Entity."""
    # Invalid age (below 18)
    response = client.post("/api/predict-clinical", json={"num_age": 10})
    assert response.status_code == 422

