"""
Inference module for trained knee X-ray models and multimodal pipelines.
Loads saved checkpoints and outputs calibrated 3-class probabilities summing to 1.0.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.models import KneeXRayCNNClassifier, MultimodalFusionClassifier
from src.image_transforms import KneeXRayEvalTransform
from src.metadata_preprocessing import ClinicalFeaturePreprocessor

CLASS_NAMES = ["Normal", "Osteopenia", "Osteoporosis"]

def predict_single_image(
    model: torch.nn.Module,
    image_path: str,
    device: Optional[torch.device] = None,
    target_size: int = 224,
) -> Dict[str, Any]:
    """Perform image-only inference on a single radiograph."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = KneeXRayEvalTransform(target_size)
    with Image.open(image_path) as img:
        tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    return {
        "image_path": image_path,
        "predicted_label": pred_label,
        "confidence": float(probs[pred_idx]),
        "probabilities": {cname: float(p) for cname, p in zip(CLASS_NAMES, probs)},
        "probability_sum": float(np.sum(probs)),
    }

def predict_multimodal(
    model: MultimodalFusionClassifier,
    image_path: str,
    clinical_features: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    target_size: int = 224,
) -> Dict[str, Any]:
    """Perform multimodal inference on a radiograph and clinical feature vector."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = KneeXRayEvalTransform(target_size)
    with Image.open(image_path) as img:
        img_tensor = transform(img).unsqueeze(0).to(device)

    clin_tensor = None
    if clinical_features is not None:
        clin_tensor = torch.tensor(clinical_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor, clin_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    return {
        "image_path": image_path,
        "predicted_label": pred_label,
        "confidence": float(probs[pred_idx]),
        "probabilities": {cname: float(p) for cname, p in zip(CLASS_NAMES, probs)},
        "probability_sum": float(np.sum(probs)),
    }
