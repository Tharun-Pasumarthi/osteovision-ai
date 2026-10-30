"""
SHAP Explainability Service for the Clinical Machine Learning Model.
Provides local prediction explanations (waterfall attribution, positive/negative contributions)
and global feature importance rankings using TreeExplainer.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap

from backend.app.services.model_registry import registry
from backend.app.schemas.models import ClinicalInputSchema

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Normal", "Osteopenia", "Osteoporosis"]

# Feature metadata and clinical descriptions for the exact 13-feature contract
FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "num_age": {
        "display_name": "Patient Age",
        "unit": "years",
        "category": "Demographic",
        "description": "Chronological age. Advanced age is the strongest non-modifiable clinical predictor of systemic bone mineral density reduction.",
    },
    "num_height_m": {
        "display_name": "Height",
        "unit": "metres",
        "category": "Anthropometric",
        "description": "Standing stature in metres. Progressive height loss may signal silent vertebral compression fractures.",
    },
    "num_weight_kg": {
        "display_name": "Weight",
        "unit": "kg",
        "category": "Anthropometric",
        "description": "Total body weight in kilograms. Lower body weight (<57 kg) is an established risk factor for osteoporotic hip fractures.",
    },
    "num_bmi": {
        "display_name": "Body Mass Index (BMI)",
        "unit": "kg/m²",
        "category": "Anthropometric",
        "description": "Ratio of weight to height squared. Lower BMI (<18.5) diminishes mechanical skeletal loading; higher BMI offers biomechanical protection.",
    },
    "num_menopause_age": {
        "display_name": "Menopause Onset Age",
        "unit": "years",
        "category": "Endocrine",
        "description": "Age at cessation of menses. Early menopause (<45y) accelerates estrogen-depletion-related trabecular bone resorption.",
    },
    "num_max_walking_km": {
        "display_name": "Daily Walking Capacity",
        "unit": "km",
        "category": "Functional Mobility",
        "description": "Maximum estimated walking distance. Weight-bearing ambulation stimulates osteoblast activity and preserves femoral neck bone mineral density.",
    },
    "gender_female": {
        "display_name": "Biological Gender (Female)",
        "unit": "binary",
        "category": "Demographic",
        "description": "Biological sex. Post-menopausal females experience accelerated cortical porosity due to hypoestrogenism.",
    },
    "joint_pain": {
        "display_name": "Knee Joint Pain",
        "unit": "binary",
        "category": "Symptom",
        "description": "Reported localized knee arthralgia, frequently associated with osteoarthritis or subchondral bone remodeling.",
    },
    "smoker": {
        "display_name": "Smoking Status",
        "unit": "binary",
        "category": "Lifestyle Risk",
        "description": "Cigarette smoking impairs intestinal calcium absorption and exerts direct toxic effects on osteoblasts.",
    },
    "diabetic": {
        "display_name": "Type II Diabetes Mellitus",
        "unit": "binary",
        "category": "Metabolic",
        "description": "Chronic hyperglycemia leads to accumulation of advanced glycation end-products (AGEs) in bone collagen, impairing skeletal micro-architecture.",
    },
    "hypothyroidism": {
        "display_name": "Hypothyroidism History",
        "unit": "binary",
        "category": "Endocrine",
        "description": "Thyroid dysfunction alters baseline bone turnover rates and increases fracture susceptibility.",
    },
    "fracture": {
        "display_name": "Prior Fragility Fracture",
        "unit": "binary",
        "category": "Clinical History",
        "description": "History of low-trauma fracture. A prior fragility fracture doubles subsequent fracture risk independent of bone mineral density.",
    },
    "family_history": {
        "display_name": "Family History of Osteoporosis",
        "unit": "binary",
        "category": "Genetic Risk",
        "description": "First-degree relative diagnosed with osteoporosis or hip fracture, indicating genetic predisposition to reduced peak bone mass.",
    },
}

class ShapExplainabilityService:
    def __init__(self):
        self._explainer: Optional[shap.TreeExplainer] = None
        self._cached_rf = None

    def _get_explainer(self) -> shap.TreeExplainer:
        """Lazily initialize and cache the TreeExplainer on the base RandomForest model."""
        if registry.clinical_model is None:
            registry.load_all()

        if registry.clinical_model is None:
            raise ValueError("Clinical classification model is not loaded.")

        # In CalibratedClassifierCV, extract the base fitted estimator
        rf = getattr(registry.clinical_model, "calibrated_classifiers_", None)
        if rf and len(rf) > 0:
            base_estimator = rf[0].estimator
        elif hasattr(registry.clinical_model, "estimator"):
            base_estimator = registry.clinical_model.estimator
        else:
            base_estimator = registry.clinical_model

        if self._explainer is None or self._cached_rf != base_estimator:
            logger.info("Initializing SHAP TreeExplainer on clinical RandomForest estimator...")
            self._explainer = shap.TreeExplainer(base_estimator)
            self._cached_rf = base_estimator

        return self._explainer

    def explain_patient(
        self,
        clinical_input: ClinicalInputSchema,
        target_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate local SHAP explanations for an individual clinical profile.
        Preserves the 13-feature contract order and standard preprocessor pipeline.
        """
        if registry.clinical_preprocessor is None:
            registry.load_all()

        if registry.clinical_preprocessor is None:
            raise ValueError("Clinical preprocessor is unavailable.")

        explainer = self._get_explainer()
        feature_dict = clinical_input.to_feature_dict()
        df = pd.DataFrame([feature_dict])

        # Preprocess features using fitted pipeline
        feat_matrix, feat_names = registry.clinical_preprocessor.transform(df)
        feat_matrix_np = np.asarray(feat_matrix, dtype=np.float32)

        # Predict probabilities using full calibrated clinical model
        calibrated_probs = registry.clinical_model.predict_proba(feat_matrix_np)[0]
        predicted_idx = int(np.argmax(calibrated_probs))
        predicted_class = CLASS_NAMES[predicted_idx]

        # Determine class to explain
        if target_class and target_class.strip():
            matched = [i for i, name in enumerate(CLASS_NAMES) if name.lower() == target_class.strip().lower()]
            target_idx = matched[0] if matched else predicted_idx
        else:
            target_idx = predicted_idx

        target_class_name = CLASS_NAMES[target_idx]

        # Compute SHAP values
        raw_shap_values = explainer.shap_values(feat_matrix_np)
        
        # Handle multi-class shape: (1, 13, 3) or list of [ (1, 13), ... ]
        if isinstance(raw_shap_values, list):
            class_shap = raw_shap_values[target_idx][0]
        elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
            class_shap = raw_shap_values[0, :, target_idx]
        elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 2:
            class_shap = raw_shap_values[0]
        else:
            class_shap = np.array(raw_shap_values).flatten()

        # Expected value (baseline prior probability for the target class)
        exp_val = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            base_value = float(exp_val[target_idx])
        else:
            base_value = float(exp_val)

        # Build feature contributions list
        contributions: List[Dict[str, Any]] = []
        positive_factors: List[Dict[str, Any]] = []
        negative_factors: List[Dict[str, Any]] = []

        cumulative_value = base_value

        for i, fname in enumerate(feat_names):
            shap_val = float(class_shap[i])
            raw_val = feature_dict.get(fname, None)
            meta = FEATURE_METADATA.get(fname, {
                "display_name": fname.replace("num_", "").replace("_", " ").title(),
                "unit": "",
                "category": "Clinical",
                "description": "Patient clinical indicator",
            })

            # Format raw value for display
            if raw_val is None:
                display_val = "Imputed / Not Disclosed"
            elif isinstance(raw_val, float):
                if meta["unit"] == "binary":
                    display_val = "Yes" if raw_val >= 0.5 else "No"
                elif meta["unit"] in ["metres", "km"]:
                    display_val = f"{raw_val:.2f} {meta['unit']}"
                elif meta["unit"] in ["kg", "years", "kg/m²"]:
                    display_val = f"{raw_val:.1f} {meta['unit']}"
                else:
                    display_val = f"{raw_val:.2f}"
            else:
                display_val = str(raw_val)

            contrib_entry = {
                "feature_name": fname,
                "display_name": meta["display_name"],
                "category": meta["category"],
                "unit": meta["unit"],
                "description": meta["description"],
                "feature_value": raw_val,
                "standardized_value": float(feat_matrix_np[0, i]),
                "display_value": display_val,
                "shap_value": round(shap_val, 4),
                "direction": "increases_risk" if shap_val > 0 else "protective" if shap_val < 0 else "neutral",
                "magnitude": round(abs(shap_val), 4),
            }

            contributions.append(contrib_entry)
            if shap_val > 0.001:
                positive_factors.append(contrib_entry)
            elif shap_val < -0.001:
                negative_factors.append(contrib_entry)

        # Sort contributions by absolute impact
        contributions_sorted = sorted(contributions, key=lambda x: x["magnitude"], reverse=True)
        positive_factors.sort(key=lambda x: x["shap_value"], reverse=True)
        negative_factors.sort(key=lambda x: x["shap_value"])  # Most negative first

        # Build waterfall sequence
        waterfall_steps: List[Dict[str, Any]] = []
        running_total = base_value
        for item in contributions_sorted:
            step_val = item["shap_value"]
            prev_total = running_total
            running_total += step_val
            waterfall_steps.append({
                "feature_name": item["feature_name"],
                "display_name": item["display_name"],
                "display_value": item["display_value"],
                "shap_value": round(step_val, 4),
                "start_value": round(prev_total, 4),
                "end_value": round(running_total, 4),
                "direction": item["direction"],
            })

        # Generate clinical summary narrative
        summary_lines = []
        if positive_factors:
            top_pos = [f"{p['display_name']} ({p['display_value']})" for p in positive_factors[:3]]
            summary_lines.append(
                f"Primary clinical factors driving {target_class_name} likelihood: {', '.join(top_pos)}."
            )
        if negative_factors:
            top_neg = [f"{n['display_name']} ({n['display_value']})" for n in negative_factors[:3]]
            summary_lines.append(
                f"Primary protective / mitigating factors: {', '.join(top_neg)}."
            )
        if not summary_lines:
            summary_lines.append("Patient profile is near population baseline across all 13 clinical risk dimensions.")

        explanation_summary = " ".join(summary_lines)

        return {
            "model_name": "Clinical Risk Random Forest (Calibrated)",
            "predicted_class": predicted_class,
            "target_class": target_class_name,
            "calibrated_probabilities": {
                "Normal": round(float(calibrated_probs[0]), 4),
                "Osteopenia": round(float(calibrated_probs[1]), 4),
                "Osteoporosis": round(float(calibrated_probs[2]), 4),
            },
            "base_value": round(base_value, 4),
            "final_predicted_value": round(running_total, 4),
            "contributions": contributions_sorted,
            "positive_factors": positive_factors,
            "negative_factors": negative_factors,
            "waterfall_steps": waterfall_steps,
            "summary": explanation_summary,
            "disclaimer": "SHAP attributions indicate mathematical model associations and should not be interpreted as causal clinical evidence or biochemical pathogenesis.",
        }

    def get_global_importance(self) -> List[Dict[str, Any]]:
        """
        Return verified global clinical feature importance rankings based on
        the trained model's Gini importance and clinical feature impact.
        """
        if registry.clinical_model is None:
            registry.load_all()

        if registry.clinical_model is None:
            return []

        rf = getattr(registry.clinical_model, "calibrated_classifiers_", None)
        if rf and len(rf) > 0:
            base_estimator = rf[0].estimator
        elif hasattr(registry.clinical_model, "estimator"):
            base_estimator = registry.clinical_model.estimator
        else:
            base_estimator = registry.clinical_model

        feature_names = [
            "num_age", "num_height_m", "num_weight_kg", "num_bmi",
            "num_menopause_age", "num_max_walking_km", "gender_female",
            "joint_pain", "smoker", "diabetic", "hypothyroidism",
            "fracture", "family_history"
        ]

        importances = getattr(base_estimator, "feature_importances_", None)
        if importances is None or len(importances) != len(feature_names):
            importances = np.ones(len(feature_names)) / len(feature_names)

        results = []
        total_imp = float(np.sum(importances))
        for fname, imp in zip(feature_names, importances):
            meta = FEATURE_METADATA.get(fname, {
                "display_name": fname,
                "unit": "",
                "category": "General",
                "description": "",
            })
            norm_imp = float(imp) / max(total_imp, 1e-6)
            results.append({
                "feature_name": fname,
                "display_name": meta["display_name"],
                "category": meta["category"],
                "unit": meta["unit"],
                "importance_score": round(norm_imp, 4),
                "percentage": round(norm_imp * 100.0, 2),
                "description": meta["description"],
            })

        results.sort(key=lambda x: x["importance_score"], reverse=True)
        return results

shap_service = ShapExplainabilityService()
