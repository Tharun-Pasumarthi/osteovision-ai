"""
Intelligent Image Quality Assessment (IQA) Service for Pre-Inference Knee Radiographs.
Evaluates blurriness, resolution, exposure (brightness/contrast), file integrity, and field-of-view completeness.
"""
import io
import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class ImageQualityAssessmentService:
    """
    Automated pre-inference radiographic image quality screening service.
    Screening parameters are tuned specifically for knee plain-film radiographs.
    """
    # Quality thresholds
    MIN_WIDTH: int = 150
    MIN_HEIGHT: int = 150
    RECOMMENDED_MIN_DIM: int = 224
    MIN_ASPECT_RATIO: float = 0.33
    MAX_ASPECT_RATIO: float = 3.0
    
    # Laplacian variance for blurriness
    BLUR_UNUSABLE_THRESHOLD: float = 40.0
    BLUR_BORDERLINE_THRESHOLD: float = 120.0
    
    # Brightness (mean pixel intensity 0-255)
    UNDEREXPOSED_UNUSABLE: float = 25.0
    UNDEREXPOSED_WARNING: float = 45.0
    OVEREXPOSED_WARNING: float = 215.0
    OVEREXPOSED_UNUSABLE: float = 240.0
    
    # Contrast (standard deviation of pixel intensity)
    LOW_CONTRAST_UNUSABLE: float = 18.0
    LOW_CONTRAST_WARNING: float = 32.0

    def assess_image(self, image_bytes: bytes, filename: str = "radiograph.jpg") -> Dict[str, Any]:
        """
        Assess radiographic image quality before inference.
        Returns metrics, detected issues, recommendations, quality score (0-100), and overall status.
        """
        issues: List[str] = []
        recommendations: List[str] = []
        scores: List[float] = []

        # 1. Decoding & File Integrity Check
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()
            # Reopen after verify() because verify() invalidates the image instance
            pil_img = Image.open(io.BytesIO(image_bytes))
            width, height = pil_img.size
            img_format = (pil_img.format or "UNKNOWN").upper()
        except Exception as e:
            return {
                "overall_status": "unusable",
                "quality_score": 0.0,
                "is_acceptable": False,
                "can_proceed_to_inference": False,
                "filename": filename,
                "format": "CORRUPTED",
                "width": 0,
                "height": 0,
                "aspect_ratio": 0.0,
                "laplacian_variance": 0.0,
                "mean_brightness": 0.0,
                "contrast_std": 0.0,
                "blur_status": "corrupted",
                "exposure_status": "corrupted",
                "contrast_status": "corrupted",
                "issues": [f"File decoding error or corrupted radiograph: {str(e)}"],
                "recommendations": ["Re-export the digital radiograph from the PACS or scanner and ensure the file is not truncated."],
                "disclaimer": "Heuristic image quality screening only; does not replace formal radiographic positioning and exposure QA certification.",
            }

        # Convert to numpy grayscale array for computer vision metrics
        try:
            rgb_img = pil_img.convert("RGB")
            np_rgb = np.array(rgb_img)
            gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            return {
                "overall_status": "unusable",
                "quality_score": 0.0,
                "is_acceptable": False,
                "can_proceed_to_inference": False,
                "filename": filename,
                "format": img_format,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / max(height, 1), 2),
                "laplacian_variance": 0.0,
                "mean_brightness": 0.0,
                "contrast_std": 0.0,
                "blur_status": "unusable",
                "exposure_status": "unusable",
                "contrast_status": "unusable",
                "issues": [f"Array conversion failed: {str(e)}"],
                "recommendations": ["Ensure the image is a valid 8-bit or standard RGB radiograph."],
                "disclaimer": "Heuristic image quality screening only; does not replace formal radiographic positioning and exposure QA certification.",
            }

        # 2. Dimensions & Resolution Assessment
        aspect_ratio = width / max(height, 1)
        dim_score = 100.0
        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            issues.append(f"Severely low resolution ({width}x{height}px). Minimum acceptable is {self.MIN_WIDTH}x{self.MIN_HEIGHT}px.")
            recommendations.append("Provide a higher resolution Knee radiograph to resolve trabecular bone microarchitecture.")
            dim_score = 20.0
        elif width < self.RECOMMENDED_MIN_DIM or height < self.RECOMMENDED_MIN_DIM:
            issues.append(f"Sub-optimal resolution ({width}x{height}px). Recommended minimum is {self.RECOMMENDED_MIN_DIM}x{self.RECOMMENDED_MIN_DIM}px.")
            recommendations.append("Image resolution is slightly below ideal; model will upsample with letterbox padding.")
            dim_score = 70.0
        else:
            dim_score = 100.0

        if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
            issues.append(f"Abnormal aspect ratio ({aspect_ratio:.2f}:1). May represent an extreme crop or banner.")
            recommendations.append("Ensure the entire knee joint (distal femur and proximal tibia) is framed.")
            dim_score = min(dim_score, 40.0)

        scores.append(dim_score)

        # 3. Blurriness Assessment (Laplacian Variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_status = "sharp"
        if laplacian_var < self.BLUR_UNUSABLE_THRESHOLD:
            blur_status = "severe_blur"
            issues.append(f"Severe motion or focal blur detected (Laplacian variance: {laplacian_var:.1f}). Fine cortical boundaries are obscured.")
            recommendations.append("Acquire a stationary radiograph free of patient movement.")
            blur_score = 20.0
        elif laplacian_var < self.BLUR_BORDERLINE_THRESHOLD:
            blur_status = "borderline_blur"
            issues.append(f"Moderate blur detected (Laplacian variance: {laplacian_var:.1f}). Visual sharpness is sub-optimal.")
            recommendations.append("Inspect image sharpness around the subchondral bone plate.")
            blur_score = 65.0
        else:
            blur_score = min(100.0, 75.0 + min(25.0, (laplacian_var - 120.0) / 20.0))
        scores.append(blur_score)

        # 4. Brightness / Exposure Assessment
        mean_brightness = float(np.mean(gray))
        exposure_status = "normal"
        if mean_brightness < self.UNDEREXPOSED_UNUSABLE:
            exposure_status = "severe_underexposure"
            issues.append(f"Severe underexposure / dark image (mean brightness: {mean_brightness:.1f}/255). Anatomical bone structures are hidden in shadow.")
            recommendations.append("Increase exposure factors (mAs/kVp) or window leveling.")
            exp_score = 20.0
        elif mean_brightness < self.UNDEREXPOSED_WARNING:
            exposure_status = "underexposed"
            issues.append(f"Moderate underexposure (mean brightness: {mean_brightness:.1f}/255).")
            recommendations.append("Consider adjusting radiograph windowing/leveling for clearer trabecular contrast.")
            exp_score = 65.0
        elif mean_brightness > self.OVEREXPOSED_UNUSABLE:
            exposure_status = "severe_overexposure"
            issues.append(f"Severe overexposure / burnout (mean brightness: {mean_brightness:.1f}/255). Anatomical bone areas are clipped/saturated.")
            recommendations.append("Reduce radiation exposure or re-adjust radiograph window center.")
            exp_score = 20.0
        elif mean_brightness > self.OVEREXPOSED_WARNING:
            exposure_status = "overexposed"
            issues.append(f"Moderate overexposure (mean brightness: {mean_brightness:.1f}/255). Potential clipping in cortical bone.")
            recommendations.append("Verify cortical margins are not washed out.")
            exp_score = 65.0
        else:
            exp_score = 100.0
        scores.append(exp_score)

        # 5. Contrast Assessment (Standard Deviation of Intensity)
        contrast_std = float(np.std(gray))
        contrast_status = "good"
        if contrast_std < self.LOW_CONTRAST_UNUSABLE:
            contrast_status = "flat_unusable"
            issues.append(f"Severely low radiographic contrast (std: {contrast_std:.1f}). Bone and soft tissue boundaries lack dynamic range.")
            recommendations.append("Apply standard radiographic CLAHE or window leveling to separate bone density gradations.")
            contrast_score = 20.0
        elif contrast_std < self.LOW_CONTRAST_WARNING:
            contrast_status = "low_contrast"
            issues.append(f"Low contrast (std: {contrast_std:.1f}).")
            recommendations.append("Contrast is somewhat low; model may still detect macroscopic features.")
            contrast_score = 65.0
        else:
            contrast_score = 100.0
        scores.append(contrast_score)

        # 6. Field-of-View & Cropping Check (Border black/white uniformity)
        h, w = gray.shape
        border_size = max(2, min(h, w) // 25)
        top = gray[:border_size, :]
        bottom = gray[-border_size:, :]
        left = gray[:, :border_size]
        right = gray[:, -border_size:]
        borders = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
        border_zero_ratio = float(np.mean(borders < 5))
        if border_zero_ratio > 0.85:
            issues.append("Heavy black collimation borders detected along edges.")
            recommendations.append("Letterbox preprocessing will handle border areas without distorting knee anatomy.")

        # Compute composite quality score (0 - 100)
        final_score = float(np.mean(scores))
        final_score = max(0.0, min(100.0, round(final_score, 1)))

        # Determine overall status and inference eligibility
        is_severely_unusable = (
            width < 64 or height < 64
            or mean_brightness < 5.0 or mean_brightness > 250.0
            or aspect_ratio < 0.20 or aspect_ratio > 5.0
        )

        if is_severely_unusable:
            overall_status = "unusable"
            can_proceed = False
            is_acceptable = False
        elif dim_score <= 30.0 or blur_score <= 30.0 or exp_score <= 30.0 or contrast_score <= 30.0:
            overall_status = "review_recommended"
            can_proceed = True
            is_acceptable = False
        elif len(issues) > 0 or final_score < 75.0:
            overall_status = "review_recommended"
            can_proceed = True
            is_acceptable = True
        else:
            overall_status = "acceptable"
            can_proceed = True
            is_acceptable = True
            recommendations.append("Image meets all radiographic quality benchmarks for deep learning feature extraction.")

        return {
            "overall_status": overall_status,
            "quality_score": final_score,
            "is_acceptable": is_acceptable,
            "can_proceed_to_inference": can_proceed,
            "filename": filename,
            "format": img_format,
            "width": width,
            "height": height,
            "aspect_ratio": round(aspect_ratio, 2),
            "laplacian_variance": round(laplacian_var, 1),
            "mean_brightness": round(mean_brightness, 1),
            "contrast_std": round(contrast_std, 1),
            "blur_status": blur_status,
            "exposure_status": exposure_status,
            "contrast_status": contrast_status,
            "issues": issues,
            "recommendations": recommendations,
            "disclaimer": "Heuristic image quality screening only; does not replace formal radiographic positioning and exposure QA certification.",
        }

image_quality_service = ImageQualityAssessmentService()
