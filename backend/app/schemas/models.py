"""
Pydantic Schemas for Osteoporosis Classification API.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, model_validator

class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_version: str
    device: str
    models_loaded: List[str]
    checkpoints_found: Dict[str, bool]

class ModelMetadata(BaseModel):
    id: str
    name: str
    family: str
    input_type: str  # "image_only" | "image_plus_clinical" | "clinical_only"
    supports_clinical: bool
    gradcam_supported: bool
    checkpoint_available: bool
    checkpoint_path: Optional[str] = None
    test_accuracy: Optional[float] = None
    test_macro_f1: Optional[float] = None
    val_macro_f1: Optional[float] = None
    normal_recall: Optional[float] = None
    osteopenia_recall: Optional[float] = None
    osteoporosis_recall: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    parameter_count: Optional[float] = None
    description: str

class ModelsListResponse(BaseModel):
    models: List[ModelMetadata]
    default_model_id: str

class ClinicalInputSchema(BaseModel):
    """
    Exact clinical model feature contract:
    1. num_age: Age in years
    2. num_height_m: Height in metres
    3. num_weight_kg: Weight in kilograms
    4. num_bmi: Body Mass Index (kg/m^2)
    5. num_menopause_age: Age at menopause onset (years, post-menopausal female)
    6. num_max_walking_km: Daily walking distance (kilometres)
    7. gender_female: 1.0 for female, 0.0 for male
    8. joint_pain: 1.0 for yes, 0.0 for no
    9. smoker: 1.0 for yes, 0.0 for no
    10. diabetic: 1.0 for yes, 0.0 for no
    11. hypothyroidism: 1.0 for yes, 0.0 for no
    12. fracture: 1.0 for prior fragility fracture, 0.0 for no
    13. family_history: 1.0 for family history of osteoporosis, 0.0 for no
    """
    num_age: Optional[float] = Field(default=None, ge=18.0, le=120.0, description="Patient age in years")
    num_height_m: Optional[float] = Field(default=None, ge=0.5, le=2.5, description="Height in metres (e.g., 1.62)")
    num_weight_kg: Optional[float] = Field(default=None, ge=20.0, le=250.0, description="Weight in kilograms (e.g., 64.0)")
    num_bmi: Optional[float] = Field(default=None, ge=10.0, le=70.0, description="Body Mass Index in kg/m^2 (convention: weight / height^2)")
    num_menopause_age: Optional[float] = Field(default=None, ge=20.0, le=80.0, description="Age at onset of menopause in years for post-menopausal female subjects")
    num_max_walking_km: Optional[float] = Field(default=None, ge=0.0, le=50.0, description="Maximum daily walking distance in kilometres")
    gender_female: float = Field(default=1.0, ge=0.0, le=1.0, description="1.0 for female, 0.0 for male")
    joint_pain: float = Field(default=0.0, ge=0.0, le=1.0, description="1.0 for yes, 0.0 for no")
    smoker: float = Field(default=0.0, ge=0.0, le=1.0, description="1.0 for yes, 0.0 for no")
    diabetic: float = Field(default=0.0, ge=0.0, le=1.0, description="1.0 for yes, 0.0 for no")
    hypothyroidism: float = Field(default=0.0, ge=0.0, le=1.0, description="1.0 for yes, 0.0 for no")
    fracture: float = Field(default=0.0, ge=0.0, le=1.0, description="1.0 for prior fragility fracture history, 0.0 for no")
    family_history: float = Field(default=0.0, ge=0.0, le=1.0, description="1.0 for family history of osteoporosis, 0.0 for no")

    @model_validator(mode="before")
    @classmethod
    def standardize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalize legacy field names to 13-feature contract
            if "age" in data and "num_age" not in data:
                data["num_age"] = data.pop("age")
            if "height_m" in data and "num_height_m" not in data:
                data["num_height_m"] = data.pop("height_m")
            if "weight_kg" in data and "num_weight_kg" not in data:
                data["num_weight_kg"] = data.pop("weight_kg")
            if "bmi" in data and "num_bmi" not in data:
                data["num_bmi"] = data.pop("bmi")
            if "menopause_age" in data and "num_menopause_age" not in data:
                data["num_menopause_age"] = data.pop("menopause_age")
            if "max_walking_km" in data and "num_max_walking_km" not in data:
                data["num_max_walking_km"] = data.pop("max_walking_km")
            if "gender" in data and "gender_female" not in data:
                g = str(data.pop("gender")).lower().strip()
                data["gender_female"] = 1.0 if g == "female" else 0.0
            if "history_of_fracture" in data and "fracture" not in data:
                data["fracture"] = 1.0 if data.pop("history_of_fracture") else 0.0
            if "family_history_osteoporosis" in data and "family_history" not in data:
                data["family_history"] = 1.0 if data.pop("family_history_osteoporosis") else 0.0

            # Convert empty string / null placeholders for numerical fields to None
            for num_key in ["num_age", "num_height_m", "num_weight_kg", "num_bmi", "num_menopause_age", "num_max_walking_km"]:
                if num_key in data:
                    val = data[num_key]
                    if val is None or val == "" or str(val).strip().lower() in ["none", "null", "nan", "undefined"]:
                        data[num_key] = None
                    else:
                        try:
                            data[num_key] = float(val)
                        except (ValueError, TypeError):
                            pass

            # Convert boolean inputs to 1.0 / 0.0
            for bin_key in ["gender_female", "joint_pain", "smoker", "diabetic", "hypothyroidism", "fracture", "family_history"]:
                if bin_key in data and data[bin_key] is not None:
                    if isinstance(data[bin_key], bool):
                        data[bin_key] = 1.0 if data[bin_key] else 0.0
                    elif isinstance(data[bin_key], str):
                        data[bin_key] = 1.0 if data[bin_key].lower().strip() in ["yes", "y", "1", "true", "female"] else 0.0
                    else:
                        data[bin_key] = float(data[bin_key])

            # Auto calculate BMI if height and weight present and BMI not provided or zero/negative
            h = data.get("num_height_m")
            w = data.get("num_weight_kg")
            bmi = data.get("num_bmi")
            if bmi is not None and bmi <= 0:
                data["num_bmi"] = None
                bmi = None

            if bmi is None and h is not None and w is not None:
                try:
                    fh = float(h)
                    fw = float(w)
                    if fh > 0.5 and fw > 10.0:
                        data["num_bmi"] = round(fw / (fh ** 2), 2)
                except Exception:
                    pass

            # Dataset semantics: menopause age is strictly for post-menopausal females.
            # A value of 0 or <= 0 is clinically invalid as an age and denotes inapplicable/missing.
            ma = data.get("num_menopause_age")
            if ma is not None and ma <= 0:
                data["num_menopause_age"] = None

            # For males, menopause age is not applicable (set to None)
            gf = data.get("gender_female", 1.0)
            if gf == 0.0:
                data["num_menopause_age"] = None
        return data

    @property
    def age(self) -> Optional[float]:
        return self.num_age

    @property
    def height_m(self) -> Optional[float]:
        return self.num_height_m

    @property
    def weight_kg(self) -> Optional[float]:
        return self.num_weight_kg

    @property
    def bmi(self) -> Optional[float]:
        return self.num_bmi

    @property
    def menopause_age(self) -> Optional[float]:
        return self.num_menopause_age

    @property
    def max_walking_km(self) -> Optional[float]:
        return self.num_max_walking_km

    @property
    def history_of_fracture(self) -> bool:
        return bool(self.fracture >= 0.5)

    @property
    def family_history_osteoporosis(self) -> bool:
        return bool(self.family_history >= 0.5)

    @property
    def gender(self) -> str:
        return "female" if self.gender_female >= 0.5 else "male"

    def to_feature_dict(self) -> Dict[str, Any]:
        """Return the dictionary with exact 13 feature names in contract order."""
        return {
            "num_age": self.num_age,
            "num_height_m": self.num_height_m,
            "num_weight_kg": self.num_weight_kg,
            "num_bmi": self.num_bmi,
            "num_menopause_age": self.num_menopause_age,
            "num_max_walking_km": self.num_max_walking_km,
            "gender_female": self.gender_female,
            "joint_pain": self.joint_pain,
            "smoker": self.smoker,
            "diabetic": self.diabetic,
            "hypothyroidism": self.hypothyroidism,
            "fracture": self.fracture,
            "family_history": self.family_history,
        }

class PredictionResponse(BaseModel):
    predicted_label: str  # "Normal" | "Osteopenia" | "Osteoporosis"
    confidence: float
    probabilities: Dict[str, float]
    model_id: str
    model_name: str
    input_type: str
    inference_duration_ms: float
    is_calibrated: bool = False
    calibrated_probabilities: Optional[Dict[str, float]] = None
    temperature_applied: Optional[float] = None
    calibration_note: str
    is_uncertain: bool = False
    review_needed: bool = False
    review_rationale: Optional[str] = None
    model_version: str = "2.1.0-audited"
    clinical_features_used: Optional[List[str]] = None
    vision_probabilities: Optional[Dict[str, float]] = None
    clinical_probabilities: Optional[Dict[str, float]] = None
    image_meta: Optional[Dict[str, Any]] = None
    gradcam: Optional[Any] = None
    disclaimer: str

class GradCAMResponse(BaseModel):
    model_id: str
    model_name: str
    target_class: str
    predicted_class: str
    probabilities: Dict[str, float]
    original_image_base64: str
    heatmap_base64: str
    overlay_base64: str
    alpha: float
    notes: str
    disclaimer: str

class DatasetSummaryResponse(BaseModel):
    total_images_discovered: int
    readable_images: int
    unique_exact_sha256_hashes: int
    dimension_summary: Dict[str, Any]
    class_distribution: Dict[str, int]
    sources: Dict[str, Any]
    metadata_summary: Dict[str, Any]
    patient_split_counts: Dict[str, int]
    provenance_notes: Dict[str, str]

class SampleImageInfo(BaseModel):
    id: str
    filename: str
    true_label: str
    patient_id: str
    source: str
    clinical_data: Optional[Dict[str, Any]] = None
    image_url: str

class EvaluateRequest(BaseModel):
    dataset_name: str = "mendeley_test"
    model_id: str = "densenet121"

class EvaluateResponse(BaseModel):
    model_id: str
    samples_evaluated: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class_recall: Dict[str, float]
    confusion_matrix: List[List[int]]
    disclaimer: str
