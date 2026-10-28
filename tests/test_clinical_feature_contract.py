"""
Automated Comprehensive Test Suite for the 13 Clinical Features Contract.
Validates:
1. Exact 13 features, names, units, and order matching the training contract:
   1. num_age
   2. num_height_m
   3. num_weight_kg
   4. num_bmi
   5. num_menopause_age
   6. num_max_walking_km
   7. gender_female
   8. joint_pain
   9. smoker
   10. diabetic
   11. hypothyroidism
   12. fracture
   13. family_history
2. Form-generated feature vector vs training pipeline expected input schema.
3. BMI calculation convention: weight_kg / (height_m ** 2).
4. Menopause age dataset semantics and missing-value policy.
5. Gender and binary features exact [1.0, 0.0] encoding.
6. Clinical model 3-class prediction alignment (Normal, Osteopenia, Osteoporosis).
7. Separate analysis pathways isolation (no ad-hoc probability averaging).
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pytest
import numpy as np
import pandas as pd
import torch

from src.metadata_preprocessing import ClinicalFeaturePreprocessor, build_feature_dictionary
from backend.app.schemas.models import ClinicalInputSchema, PredictionResponse
from backend.app.services.inference_service import inference_service, CLASS_NAMES
from backend.app.services.model_registry import registry

EXPECTED_13_FEATURES = [
    "num_age",
    "num_height_m",
    "num_weight_kg",
    "num_bmi",
    "num_menopause_age",
    "num_max_walking_km",
    "gender_female",
    "joint_pain",
    "smoker",
    "diabetic",
    "hypothyroidism",
    "fracture",
    "family_history",
]

# =====================================================================
# 1. Feature Contract Order, Names, and Count Verification
# =====================================================================

def test_exact_13_features_order_and_names():
    """Verify preprocessor feature names and ordering strictly match the 13 features contract."""
    prep = ClinicalFeaturePreprocessor()
    assert prep.feature_names == EXPECTED_13_FEATURES
    assert len(prep.feature_names) == 13

    # Verify documented feature dictionary
    feat_dict = build_feature_dictionary()
    dict_features = list(feat_dict["features"].keys())
    assert dict_features == EXPECTED_13_FEATURES
    for i, name in enumerate(EXPECTED_13_FEATURES, start=1):
        assert feat_dict["features"][name]["order"] == i

def test_schema_generates_exact_13_features_dict():
    """Verify ClinicalInputSchema.to_feature_dict produces exact 13 keys in order."""
    schema = ClinicalInputSchema(
        num_age=62.0,
        num_height_m=1.58,
        num_weight_kg=64.0,
        num_bmi=25.64,
        num_menopause_age=46.0,
        num_max_walking_km=2.5,
        gender_female=1.0,
        joint_pain=1.0,
        smoker=0.0,
        diabetic=0.0,
        hypothyroidism=0.0,
        fracture=1.0,
        family_history=1.0,
    )
    feature_dict = schema.to_feature_dict()
    assert list(feature_dict.keys()) == EXPECTED_13_FEATURES
    assert len(feature_dict) == 13
    assert feature_dict["num_age"] == 62.0
    assert feature_dict["gender_female"] == 1.0
    assert feature_dict["fracture"] == 1.0

# =====================================================================
# 2. BMI Calculation Convention Tests
# =====================================================================

def test_bmi_calculation_convention():
    """Verify BMI is computed as weight_kg / (height_m ** 2) matching training convention."""
    # Test case 1: 64 kg, 1.58 m -> 64 / (1.58^2) = 25.6368... -> 25.64
    schema1 = ClinicalInputSchema(
        num_height_m=1.58,
        num_weight_kg=64.0,
    )
    expected_bmi_1 = round(64.0 / (1.58 ** 2), 2)
    assert schema1.num_bmi == expected_bmi_1

    # Test case 2: 78 kg, 1.68 m -> 78 / (1.68^2) = 27.636... -> 27.64
    schema2 = ClinicalInputSchema(
        num_height_m=1.68,
        num_weight_kg=78.0,
    )
    expected_bmi_2 = round(78.0 / (1.68 ** 2), 2)
    assert schema2.num_bmi == expected_bmi_2

    # Test case 3: Preprocessor extraction computes same BMI convention
    prep = ClinicalFeaturePreprocessor()
    df_raw = pd.DataFrame([{
        "Age": 55,
        "height  (meter)": 1.60,
        "Weight (KG) ": 64.0,
        "Gender": "female",
    }])
    clean = prep._extract_clean_dataframe(df_raw)
    assert np.isclose(clean["bmi"].iloc[0], 64.0 / (1.60 ** 2), atol=1e-3)

# =====================================================================
# 3. Menopause Age Semantics and Missing-Value Policy Tests
# =====================================================================

def test_menopause_age_semantics_for_males_and_females():
    """Verify menopause age is validated per dataset semantics: null for males, median-imputed on transform."""
    # Male patient: menopause age must be null regardless of input
    schema_male = ClinicalInputSchema(
        num_age=65.0,
        gender_female=0.0,
        num_menopause_age=45.0, # should be neutralized to None for males
    )
    assert schema_male.num_menopause_age is None

    # Pre-menopausal female: menopause age is None
    schema_pre_female = ClinicalInputSchema(
        num_age=35.0,
        gender_female=1.0,
        num_menopause_age=None,
    )
    assert schema_pre_female.num_menopause_age is None

    # Post-menopausal female: menopause age is preserved
    schema_post_female = ClinicalInputSchema(
        num_age=58.0,
        gender_female=1.0,
        num_menopause_age=48.0,
    )
    assert schema_post_female.num_menopause_age == 48.0

def test_missing_value_imputation_policy():
    """Verify missing numerical features are imputed strictly with training medians."""
    ckpt_path = "outputs/checkpoints/clinical_preprocessor.pkl"
    assert os.path.exists(ckpt_path), f"Missing preprocessor checkpoint: {ckpt_path}"
    prep = ClinicalFeaturePreprocessor.load(ckpt_path)

    # Empty payload except required indicators
    schema = ClinicalInputSchema(
        gender_female=1.0,
    )
    df = pd.DataFrame([schema.to_feature_dict()])
    feat_matrix, feat_names = prep.transform(df)

    assert feat_matrix.shape == (1, 13)
    assert feat_names == EXPECTED_13_FEATURES
    assert not np.isnan(feat_matrix).any(), "Transformed feature vector must contain zero NaN values!"

    # Verify that num_menopause_age (index 4) was imputed with training median
    expected_median = prep.medians["menopause_age"]
    expected_scaled = (expected_median - prep.scaler.mean_[4]) / prep.scaler.scale_[4]
    scaled_menopause = feat_matrix[0, 4]
    assert np.isclose(scaled_menopause, expected_scaled, atol=1e-5), f"Expected standardized median {expected_scaled}, got {scaled_menopause}"

# =====================================================================
# 4. Binary Features Encoding Tests
# =====================================================================

def test_binary_features_exact_encoding():
    """Verify gender_female and all binary indicators are strictly encoded as 1.0 or 0.0."""
    schema = ClinicalInputSchema(
        gender_female=1.0,
        joint_pain=True,
        smoker=False,
        diabetic=1.0,
        hypothyroidism=0.0,
        fracture="yes",
        family_history="no",
    )
    fdict = schema.to_feature_dict()
    assert fdict["gender_female"] == 1.0
    assert fdict["joint_pain"] == 1.0
    assert fdict["smoker"] == 0.0
    assert fdict["diabetic"] == 1.0
    assert fdict["hypothyroidism"] == 0.0
    assert fdict["fracture"] == 1.0
    assert fdict["family_history"] == 0.0

    # Ensure no unsupported features (such as obesity_category) are present
    assert "obesity_category" not in fdict
    assert "obesity_normal weight" not in fdict

# =====================================================================
# 5. Form Vector vs Training Pipeline Schema Alignment Test
# =====================================================================

def test_form_generated_vector_matches_training_pipeline_schema():
    """Verify that form-generated feature vector matches the training pipeline expected input shape and types."""
    ckpt_path = "outputs/checkpoints/clinical_preprocessor.pkl"
    prep = ClinicalFeaturePreprocessor.load(ckpt_path)

    # Sample form input
    form_data = {
        "num_age": 62.0,
        "num_height_m": 1.58,
        "num_weight_kg": 64.0,
        "num_bmi": 25.64,
        "num_menopause_age": 46.0,
        "num_max_walking_km": 2.5,
        "gender_female": 1.0,
        "joint_pain": 1.0,
        "smoker": 0.0,
        "diabetic": 0.0,
        "hypothyroidism": 0.0,
        "fracture": 1.0,
        "family_history": 1.0,
    }

    schema = ClinicalInputSchema(**form_data)
    df = pd.DataFrame([schema.to_feature_dict()])
    features, feature_names = prep.transform(df)

    # Assertions
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert features.shape == (1, 13)
    assert feature_names == EXPECTED_13_FEATURES
    assert list(df.columns) == EXPECTED_13_FEATURES

# =====================================================================
# 6. Clinical Model 3-Class Prediction Alignment Tests
# =====================================================================

def test_clinical_model_three_class_prediction():
    """Verify clinical model predicts strictly Normal, Osteopenia, Osteoporosis with valid probabilities."""
    registry.load_all()
    assert registry.clinical_model is not None, "Clinical model must be loaded in registry!"

    schema = ClinicalInputSchema(
        num_age=74.0,
        num_height_m=1.52,
        num_weight_kg=48.0,
        num_menopause_age=44.0,
        num_max_walking_km=0.8,
        gender_female=1.0,
        joint_pain=1.0,
        smoker=0.0,
        diabetic=0.0,
        hypothyroidism=1.0,
        fracture=1.0,
        family_history=1.0,
    )

    pred: PredictionResponse = inference_service.predict_clinical_only(schema)

    # Canonical class name and mapping checks
    assert pred.predicted_label in ["Normal", "Osteopenia", "Osteoporosis"]
    assert CLASS_NAMES == ["Normal", "Osteopenia", "Osteoporosis"]
    assert pred.model_id == "clinical_13feat_model"
    assert pred.model_name == "Clinical Model Prediction"

    # Probabilities verification
    probs = pred.probabilities
    assert set(probs.keys()) == {"Normal", "Osteopenia", "Osteoporosis"}
    prob_sum = probs["Normal"] + probs["Osteopenia"] + probs["Osteoporosis"]
    assert np.isclose(prob_sum, 1.0, atol=1e-3)
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.confidence == probs[pred.predicted_label]

    # Clinical features used must match exact 13 features
    assert pred.clinical_features_used == EXPECTED_13_FEATURES

# =====================================================================
# 7. Separate Analysis Pathways Isolation (No Probability Averaging)
# =====================================================================

def test_separate_analysis_pathways_isolation():
    """Verify clinical analysis and image analysis remain strictly separated without output averaging."""
    registry.load_all()

    # Clinical prediction
    schema = ClinicalInputSchema(
        num_age=58.0,
        num_height_m=1.58,
        num_weight_kg=60.0,
        num_menopause_age=46.0,
        num_max_walking_km=2.0,
        gender_female=1.0,
        joint_pain=1.0,
        fracture=0.0,
        family_history=1.0,
    )
    clin_pred = inference_service.predict_clinical_only(schema)
    assert clin_pred.input_type == "clinical_only"
    assert clin_pred.gradcam is None
    assert clin_pred.image_meta is None

    # Verify that clinical confidence comes directly from the 13-feature model
    tensor, _ = inference_service.prepare_clinical_tensor(schema)
    raw_probs = registry.clinical_model.predict_proba(tensor.cpu().numpy())[0]
    expected_clin_conf = float(np.max(raw_probs))
    assert np.isclose(clin_pred.confidence, expected_clin_conf, atol=1e-3)
