"""
Unified Explainability Service coordinating Image Grad-CAM and Clinical SHAP Explanations.
Maintains strict separation between the radiograph deep learning pathway and clinical tabular pathway.
"""
import logging
from typing import Dict, Any, Optional

from backend.app.services.gradcam_service import gradcam_service
from backend.app.services.shap_service import shap_service
from backend.app.schemas.models import ClinicalInputSchema, GradCAMResponse

logger = logging.getLogger(__name__)

class UnifiedExplainabilityService:
    def explain_image(
        self,
        image_bytes: bytes,
        model_id: str,
        target_class: Optional[str] = None,
        alpha: float = 0.45,
        colormap_name: str = "jet",
        threshold: float = 0.0,
    ) -> GradCAMResponse:
        """Generate Grad-CAM visualization for radiograph convolutional activations."""
        return gradcam_service.generate_explanation(
            image_bytes=image_bytes,
            model_id=model_id,
            target_class=target_class,
            alpha=alpha,
            colormap_name=colormap_name,
            threshold=threshold,
        )

    def explain_clinical(
        self,
        clinical_input: ClinicalInputSchema,
        target_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate SHAP local attribution and waterfall breakdown for clinical indicators."""
        return shap_service.explain_patient(
            clinical_input=clinical_input,
            target_class=target_class,
        )

    def get_clinical_global_importance(self):
        """Retrieve global SHAP feature importance rankings across the 13 clinical indicators."""
        return shap_service.get_global_importance()

explainability_service = UnifiedExplainabilityService()

