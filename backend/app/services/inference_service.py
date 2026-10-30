"""
Inference Service running actual PyTorch model predictions.
"""
import io
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import numpy as np
import pandas as pd
import torch

from src.image_transforms import KneeXRayEvalTransform
from backend.app.services.model_registry import registry
from backend.app.schemas.models import PredictionResponse, ClinicalInputSchema
from backend.app.services.image_quality_service import image_quality_service

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Normal", "Osteopenia", "Osteoporosis"]

class InferenceService:
    def __init__(self):
        self.transform = KneeXRayEvalTransform(target_size=224)

    def prepare_clinical_tensor(self, clinical_input: Optional[ClinicalInputSchema]) -> Tuple[Optional[torch.Tensor], List[str]]:
        """
        Convert structured clinical input into the exact 13-dimensional float32 feature vector
        using the training-fitted ClinicalFeaturePreprocessor.
        Features in exact order:
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
        """
        if clinical_input is None or registry.clinical_preprocessor is None:
            return None, []

        feature_dict = clinical_input.to_feature_dict()
        df = pd.DataFrame([feature_dict])
        try:
            feat_matrix, feat_names = registry.clinical_preprocessor.transform(df)
            tensor = torch.tensor(feat_matrix, dtype=torch.float32, device=registry.device)
            return tensor, feat_names
        except Exception as e:
            logger.error(f"Error transforming clinical features: {e}")
            return None, []

    def predict_clinical_only(
        self,
        clinical_input: ClinicalInputSchema,
    ) -> PredictionResponse:
        """
        Run clinical model prediction based strictly on the 13 clinical features.
        Preserves the 3-class index mapping: 0=Normal, 1=Osteopenia, 2=Osteoporosis.
        """
        start_time = time.perf_counter()
        
        # Extract 13 features
        tensor, features_used = self.prepare_clinical_tensor(clinical_input)
        if tensor is None or registry.clinical_model is None:
            raise ValueError("Clinical classification model or preprocessor is unavailable.")

        feat_np = tensor.cpu().numpy()
        probs = registry.clinical_model.predict_proba(feat_np)[0]
        # Extract uncalibrated raw tree voting probabilities
        rf = getattr(registry.clinical_model, "calibrated_classifiers_", None)
        if rf and len(rf) > 0:
            raw_tree_probs = rf[0].estimator.predict_proba(feat_np)[0]
        else:
            raw_tree_probs = probs

        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        is_uncertain = confidence < 0.50
        review_needed = pred_label in ["Osteopenia", "Osteoporosis"] or is_uncertain
        rationale_parts = []
        if pred_label in ["Osteopenia", "Osteoporosis"]:
            rationale_parts.append(
                f"Clinical model prediction indicates {pred_label} (confidence: {confidence*100:.1f}%) based on patient non-imaging systemic indicators."
            )
            rationale_parts.append(
                "A knee X-ray radiograph or dual-energy X-ray absorptiometry (DXA) scan is recommended for structural bone mineral confirmation."
            )
        else:
            rationale_parts.append(
                "Patient profile shows lower systemic risk factors consistent with Normal bone density expectations."
            )

        return PredictionResponse(
            predicted_label=pred_label,
            confidence=confidence,
            probabilities={
                "Normal": float(probs[0]),
                "Osteopenia": float(probs[1]),
                "Osteoporosis": float(probs[2]),
            },
            raw_probabilities={
                "Normal": float(raw_tree_probs[0]),
                "Osteopenia": float(raw_tree_probs[1]),
                "Osteoporosis": float(raw_tree_probs[2]),
            },
            model_id="clinical_13feat_model",
            model_name="Clinical Model Prediction",
            input_type="clinical_only",
            inference_duration_ms=round(duration_ms, 2),
            is_calibrated=True,
            calibrated_probabilities={
                "Normal": float(probs[0]),
                "Osteopenia": float(probs[1]),
                "Osteoporosis": float(probs[2]),
            },
            temperature_applied=1.0,
            calibration_note="Calibrated via 3-fold cross-validation Platt scaling on authentic clinical patient records.",
            is_uncertain=is_uncertain,
            review_needed=review_needed,
            review_rationale=" ".join(rationale_parts),
            model_version="2.1.0-audited",
            clinical_features_used=features_used,
            vision_probabilities=None,
            clinical_probabilities={
                "Normal": float(probs[0]),
                "Osteopenia": float(probs[1]),
                "Osteoporosis": float(probs[2]),
            },
            image_meta=None,
            gradcam=None,
            disclaimer="Research prototype only. Model outputs are experimental and do not establish a clinical diagnosis or prescribe medical treatment. Professional DXA bone mineral density assessment is recommended.",
        )

    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        model_id: str,
        clinical_input: Optional[ClinicalInputSchema] = None,
    ) -> PredictionResponse:
        """Run actual trained model inference on a radiograph image, fusing clinical data when present."""
        model = registry.get_model(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' is not loaded or does not exist.")

        # 1. Pre-inference Image Quality Screening
        quality_res = image_quality_service.assess_image(image_bytes, filename)
        if not quality_res.get("can_proceed_to_inference", True):
            issues_str = "; ".join(quality_res.get("issues", ["Severe image degradation detected"]))
            raise ValueError(f"Image Quality Screening failed ({issues_str}). Pre-inference check prevented processing.")

        # Decode image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            width, height = pil_img.size
            img_format = pil_img.format or "UNKNOWN"
        except Exception as e:
            raise ValueError(f"Failed to decode image file '{filename}': {str(e)}")

        # Prepare image tensor
        img_tensor = self.transform(pil_img).unsqueeze(0).to(registry.device)

        # Prepare clinical tensor and calculate clinical component probabilities if provided
        clin_tensor, features_used = None, []
        vision_prob_dict: Optional[Dict[str, float]] = None
        clinical_prob_dict: Optional[Dict[str, float]] = None

        if clinical_input is not None:
            clin_tensor, features_used = self.prepare_clinical_tensor(clinical_input)
            if clin_tensor is not None and registry.clinical_model is not None:
                feat_np = clin_tensor.cpu().numpy()
                clin_probs = registry.clinical_model.predict_proba(feat_np)[0]
                clinical_prob_dict = {
                    "Normal": float(clin_probs[0]),
                    "Osteopenia": float(clin_probs[1]),
                    "Osteoporosis": float(clin_probs[2]),
                }

        input_type = "image_only"
        if clin_tensor is not None:
            input_type = "image_plus_clinical"

        # Run inference and measure duration
        start_time = time.perf_counter()
        calib_note = ""
        with torch.inference_mode():
            if model_id in ("multimodal_fusion", "cm_graf_v2"):
                # Forward pass through multimodal architecture
                logits = model(img_tensor, clin_tensor)
                temp = registry.get_temperature(model_id)
                calib_logits = logits / max(temp, 1e-4)
                vision_probs = torch.softmax(calib_logits, dim=1).cpu().numpy()[0]
                vision_prob_dict = {
                    "Normal": float(vision_probs[0]),
                    "Osteopenia": float(vision_probs[1]),
                    "Osteoporosis": float(vision_probs[2]),
                }

                if clinical_prob_dict is not None:
                    clin_probs_arr = np.array([
                        clinical_prob_dict["Normal"],
                        clinical_prob_dict["Osteopenia"],
                        clinical_prob_dict["Osteoporosis"],
                    ])
                    # Multimodal Fusion:
                    # 60% Radiographic cortical/trabecular microarchitecture + 40% validated 13-feature clinical risk factors
                    fused_probs = 0.60 * vision_probs + 0.40 * clin_probs_arr
                    fused_probs = fused_probs / fused_probs.sum()
                    raw_probs = fused_probs
                    calib_probs = fused_probs
                    calib_note = (
                        "Multimodal Fusion: 60% radiograph bone microarchitecture + "
                        "40% calibrated 13-feature systemic clinical risk indicators."
                    )
                else:
                    raw_probs = vision_probs
                    calib_probs = vision_probs
                    calib_note = f"Post-hoc temperature scaling (T={temp:.3f}) applied (clinical data not supplied)."

            elif model_id == "adaptive_fusion":
                logits = model(img_tensor)
                temp = registry.get_temperature(model_id)
                calib_logits = logits / max(temp, 1e-4)
                vision_probs = torch.softmax(calib_logits, dim=1).cpu().numpy()[0]
                vision_prob_dict = {
                    "Normal": float(vision_probs[0]),
                    "Osteopenia": float(vision_probs[1]),
                    "Osteoporosis": float(vision_probs[2]),
                }

                if clinical_prob_dict is not None:
                    clin_probs_arr = np.array([
                        clinical_prob_dict["Normal"],
                        clinical_prob_dict["Osteopenia"],
                        clinical_prob_dict["Osteoporosis"],
                    ])
                    fused_probs = 0.60 * vision_probs + 0.40 * clin_probs_arr
                    fused_probs = fused_probs / fused_probs.sum()
                else:
                    fused_probs = vision_probs
                raw_probs = fused_probs
                # Multi-class validation threshold calibration multipliers
                weights = np.array([1.08, 1.00, 0.96])
                weighted = fused_probs * weights
                calib_probs = weighted / weighted.sum()
                temp = 1.0
                calib_note = "Multi-class validation threshold calibration applied with 60% vision + 40% clinical risk integration."
            else:
                logits = model(img_tensor)
                raw_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                temp = registry.get_temperature(model_id)
                calib_logits = logits / max(temp, 1e-4)
                calib_probs = torch.softmax(calib_logits, dim=1).cpu().numpy()[0]
                vision_prob_dict = {
                    "Normal": float(calib_probs[0]),
                    "Osteopenia": float(calib_probs[1]),
                    "Osteoporosis": float(calib_probs[2]),
                }
                calib_note = f"Post-hoc temperature scaling (T={temp:.3f}) applied to map raw logits to calibrated probabilities."

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000.0

        # We use calibrated probabilities for final inference confidence and decision
        pred_idx = int(np.argmax(calib_probs))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(calib_probs[pred_idx])

        # Uncertainty analysis
        sorted_probs = np.sort(calib_probs)[::-1]
        top1 = float(sorted_probs[0])
        top2 = float(sorted_probs[1])
        margin = top1 - top2

        is_uncertain = False
        review_needed = False
        rationale_parts = []

        if top1 < 0.50:
            is_uncertain = True
            review_needed = True
            rationale_parts.append(f"Maximum class confidence ({top1*100:.1f}%) is below 50.0% certainty threshold.")

        if margin < 0.12:
            is_uncertain = True
            review_needed = True
            rationale_parts.append(f"Narrow confidence margin between top-2 predictions ({margin*100:.1f}%).")

        if pred_label in ["Osteopenia", "Osteoporosis"]:
            review_needed = True
            rationale_parts.append(f"Presumptive {pred_label} detected. DXA bone mineral density assessment recommended for definitive diagnostic confirmation.")

        review_rationale = " ".join(rationale_parts) if review_needed else "High-confidence prediction consistent with normal bone mineral density features."

        del img_tensor
        if clin_tensor is not None:
            del clin_tensor
        registry.release_system_memory()

        # Get friendly model name
        model_name = registry.checkpoints_config.get(model_id, {}).get("name", model_id)

        return PredictionResponse(
            predicted_label=pred_label,
            confidence=confidence,
            probabilities={
                "Normal": float(calib_probs[0]),
                "Osteopenia": float(calib_probs[1]),
                "Osteoporosis": float(calib_probs[2]),
            },
            raw_probabilities={
                "Normal": float(raw_probs[0]),
                "Osteopenia": float(raw_probs[1]),
                "Osteoporosis": float(raw_probs[2]),
            },
            model_id=model_id,
            model_name=model_name,
            input_type=input_type,
            inference_duration_ms=round(duration_ms, 2),
            is_calibrated=True,
            calibrated_probabilities={
                "Normal": float(calib_probs[0]),
                "Osteopenia": float(calib_probs[1]),
                "Osteoporosis": float(calib_probs[2]),
            },
            temperature_applied=round(temp, 4),
            calibration_note=calib_note,
            is_uncertain=is_uncertain,
            review_needed=review_needed,
            review_rationale=review_rationale,
            model_version="2.1.0-audited",
            clinical_features_used=features_used if len(features_used) > 0 else None,
            vision_probabilities=vision_prob_dict,
            clinical_probabilities=clinical_prob_dict,
            image_meta={
                "filename": filename,
                "width": width,
                "height": height,
                "format": img_format,
            },
            image_quality=quality_res,
            disclaimer="Research prototype only. Model outputs are experimental and are not a substitute for professional clinical DXA assessment.",
        )

inference_service = InferenceService()
