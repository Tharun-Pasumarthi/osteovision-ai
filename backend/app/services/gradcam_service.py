"""
Grad-CAM Explainability Service generating authentic layer activations,
custom clinical colormaps (JET, TURBO, VIRIDIS, INFERNO, BONE), and non-distorting overlays.
"""
import io
import base64
import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from src.image_transforms import resize_and_pad_radiograph, IMAGENET_MEAN, IMAGENET_STD
from src.gradcam import GradCAM, project_cam_to_original_image, COLORMAPS
from backend.app.services.model_registry import registry
from backend.app.schemas.models import GradCAMResponse

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Normal", "Osteopenia", "Osteoporosis"]

def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"

def numpy_to_base64(arr: np.ndarray, format: str = "PNG") -> str:
    img = Image.fromarray(arr)
    return pil_to_base64(img, format)

class GradCAMService:
    def generate_explanation(
        self,
        image_bytes: bytes,
        model_id: str,
        target_class: Optional[str] = None,
        alpha: float = 0.45,
        colormap_name: str = "jet",
        threshold: float = 0.0,
    ) -> GradCAMResponse:
        """
        Compute authentic Grad-CAM heatmap and blended overlay with guaranteed hook cleanup,
        non-distorting native coordinate projection, and verified target conv layers.
        """
        model = registry.get_model(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' is not loaded.")

        target_layer = registry.get_target_layer(model_id)
        if target_layer is None:
            raise ValueError(f"Grad-CAM target layer is unavailable for model '{model_id}'.")

        # Decode image
        try:
            orig_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")

        # Letterbox pad to 224x224 for neural inference
        padded_pil = resize_and_pad_radiograph(orig_pil, 224)

        # Standard ImageNet normalization
        arr = np.array(padded_pil).astype(np.float32) / 255.0
        for c in range(3):
            arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(registry.device)

        # Resolve target class index
        target_class_idx = None
        if target_class is not None and target_class.strip():
            matched = [i for i, name in enumerate(CLASS_NAMES) if name.lower() == target_class.strip().lower()]
            if matched:
                target_class_idx = matched[0]

        # Context-managed GradCAM execution with guaranteed hook removal
        with GradCAM(model, target_layer) as cam:
            heatmap, pred_idx, probs, cam_status = cam.generate_heatmap(tensor, target_class=target_class_idx)

        explained_class_idx = target_class_idx if target_class_idx is not None else pred_idx
        explained_class_name = CLASS_NAMES[explained_class_idx]
        pred_class_name = CLASS_NAMES[pred_idx]

        # Resolve selected colormap
        cv_colormap = COLORMAPS.get(colormap_name.lower().strip(), cv2.COLORMAP_JET)

        # Project heatmap back to native radiograph dimensions without letterbox distortion
        clamped_alpha = max(0.05, min(0.95, float(alpha)))
        clamped_thresh = max(0.0, min(0.85, float(threshold)))
        heatmap_color, overlay = project_cam_to_original_image(
            orig_pil,
            heatmap,
            target_size=224,
            colormap=cv_colormap,
            alpha=clamped_alpha,
            threshold=clamped_thresh,
        )

        # Convert to base64
        orig_base64 = pil_to_base64(orig_pil)
        heatmap_base64 = numpy_to_base64(heatmap_color)
        overlay_base64 = numpy_to_base64(overlay)

        model_name = registry.checkpoints_config.get(model_id, {}).get("name", model_id)
        
        notes = (
            f"Grad-CAM computes spatial gradients of the target class ('{explained_class_name}') with respect to "
            f"the final convolutional layer ({target_layer.__class__.__name__}). High-intensity areas indicate anatomical radiograph features "
            f"contributing most strongly to the model's logit."
        )
        if cam_status == "zero_gradient":
            notes += " Notice: Spatial gradients were near zero for this class, indicating low localized feature attribution."
        if clamped_thresh > 0.0:
            notes += f" Activation threshold filter applied: {int(clamped_thresh * 100)}% cutoff."
        if model_id == "multimodal_fusion":
            notes += " Note: For multimodal fusion, Grad-CAM visualizes activations in the image branch only and does not depict clinical tabular contributions."

        return GradCAMResponse(
            model_id=model_id,
            model_name=model_name,
            target_class=explained_class_name,
            predicted_class=pred_class_name,
            probabilities={
                "Normal": float(probs[0]),
                "Osteopenia": float(probs[1]),
                "Osteoporosis": float(probs[2]),
            },
            original_image_base64=orig_base64,
            heatmap_base64=heatmap_base64,
            overlay_base64=overlay_base64,
            alpha=clamped_alpha,
            notes=notes,
            disclaimer="Grad-CAM visualizations reflect convolutional neural correlations and should not be construed as verified histological or clinical pathology markers.",
        )

gradcam_service = GradCAMService()
