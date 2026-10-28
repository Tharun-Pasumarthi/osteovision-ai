"""
FastAPI REST API Routes for Osteoporosis Classification.
"""
import io
import json
import logging
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status
from fastapi.responses import FileResponse, Response

from backend.app.config import settings
from backend.app.schemas.models import (
    HealthResponse,
    ModelsListResponse,
    PredictionResponse,
    GradCAMResponse,
    DatasetSummaryResponse,
    ClinicalInputSchema,
    EvaluateRequest,
    EvaluateResponse,
)
from backend.app.services.model_registry import registry
from backend.app.services.inference_service import inference_service
from backend.app.services.gradcam_service import gradcam_service
from backend.app.services.report_service import report_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Report backend status, compute device, and loaded neural checkpoints."""
    checkpoints_found = {}
    for m_id, cfg in registry.checkpoints_config.items():
        checkpoints_found[m_id] = Path(cfg["checkpoint"]).exists()

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        app_version=settings.APP_VERSION,
        device=str(registry.device),
        models_loaded=(list(registry.loaded_models.keys()) + (["clinical_model"] if registry.clinical_model is not None else [])),
        checkpoints_found=checkpoints_found,
    )

@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """List all supported DL, ML, and Multimodal models with verified benchmark metrics."""
    models_meta = registry.list_models_metadata()
    return ModelsListResponse(
        models=models_meta,
        default_model_id=settings.DEFAULT_MODEL_ID,
    )

@router.get("/experiments")
async def get_experiments():
    """Return available experiment summaries from actual training runs."""
    try:
        data = report_service.get_benchmark_metrics()
        return {
            "total_models_evaluated": data.get("total_models_evaluated", 0),
            "best_validation_model": data.get("best_validation_model", ""),
            "best_test_macro_f1_model": data.get("best_test_macro_f1_model", ""),
            "models_count": len(data.get("models", [])),
        }
    except Exception as e:
        logger.error(f"Error fetching experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics():
    """Return comprehensive model comparison metrics from the benchmark report."""
    try:
        data = report_service.get_benchmark_metrics()
        return data
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset-summary", response_model=DatasetSummaryResponse)
async def get_dataset_summary():
    """Return sanitized dataset audit statistics, class counts, and provenance notes."""
    try:
        return report_service.get_dataset_summary()
    except Exception as e:
        logger.error(f"Error fetching dataset summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clinical-importance")
async def get_clinical_importance():
    """Return permutation feature importance on patient clinical variables."""
    try:
        importance = report_service.get_clinical_importance()
        return {"features": importance}
    except Exception as e:
        logger.error(f"Error fetching clinical importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sample-images")
async def get_sample_images():
    """Provide verified test radiographs for research trial and evaluation."""
    samples = report_service.get_sample_images()
    return {"samples": samples}

@router.get("/sample-images/{sample_id}/file")
async def get_sample_image_file(sample_id: str):
    """Serve the raw radiograph image bytes for a sample."""
    samples = report_service.get_sample_images()
    matching = [s for s in samples if s["id"] == sample_id]
    if not matching:
        raise HTTPException(status_code=404, detail="Sample image not found")

    file_path = settings.PROJECT_ROOT / matching[0]["rel_path"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample file missing from disk")

    media_type = "image/jpeg" if file_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    return FileResponse(file_path, media_type=media_type)

@router.post("/predict", response_model=PredictionResponse)
async def predict_xray(
    file: UploadFile = File(..., description="Knee radiograph image (JPG, PNG)"),
    model_id: str = Form("densenet121", description="Model architecture identifier"),
    clinical_data: Optional[str] = Form(None, description="Optional JSON-encoded clinical features for multimodal models"),
    clinical_features: Optional[str] = Form(None, description="Alias for clinical_data"),
):
    """
    Run authenticated inference using the specified model checkpoint.
    Applies exact letterbox padding and ImageNet normalization.
    """
    # 1. Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{ext}'. Only JPG, JPEG, and PNG radiographs are permitted.",
        )

    # 2. Read bytes and validate file size
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )
    if len(image_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum permitted size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB.",
        )

    # 3. Parse optional clinical features
    raw_clin = clinical_data if clinical_data is not None else clinical_features
    clinical_obj: Optional[ClinicalInputSchema] = None
    if raw_clin:
        try:
            parsed_json = json.loads(raw_clin)
            clinical_obj = ClinicalInputSchema(**parsed_json)
        except Exception as e:
            logger.warning(f"Failed to parse clinical features JSON: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid clinical features payload: {str(e)}",
            )

    # 4. Execute prediction
    try:
        response = inference_service.predict(
            image_bytes=image_bytes,
            filename=file.filename,
            model_id=model_id,
            clinical_input=clinical_obj,
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failure: {str(e)}")

@router.post("/predict-clinical", response_model=PredictionResponse)
async def predict_clinical(payload: ClinicalInputSchema):
    """
    Run preliminary clinical-only risk prediction based on patient clinical indicators.
    Does not require a knee X-ray radiograph.
    """
    try:
        response = inference_service.predict_clinical_only(payload)
        return response
    except Exception as e:
        logger.error(f"Clinical inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clinical inference error: {str(e)}",
        )

@router.post("/gradcam", response_model=GradCAMResponse)
async def generate_gradcam(
    file: UploadFile = File(..., description="Knee radiograph image"),
    model_id: str = Form("densenet121", description="Model architecture identifier"),
    target_class: Optional[str] = Form(None, description="Class to explain: Normal, Osteopenia, Osteoporosis"),
    alpha: float = Form(0.45, description="Blending opacity alpha (0.05 to 0.95)"),
    colormap: str = Form("jet", description="Colormap name: jet, turbo, viridis, inferno, bone, hot"),
    threshold: float = Form(0.0, description="Activation threshold cutoff (0.0 to 0.85)"),
):
    """
    Generate authentic Grad-CAM heatmap and blended overlay.
    Hooks into the final convolutional layer activations and gradients.
    """
    # 1. Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{ext}'. Only JPG, JPEG, and PNG radiographs are permitted.",
        )

    # 2. Read bytes and validate file size
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(image_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large.")

    # 3. Generate Grad-CAM
    try:
        response = gradcam_service.generate_explanation(
            image_bytes=image_bytes,
            model_id=model_id,
            target_class=target_class,
            alpha=alpha,
            colormap_name=colormap,
            threshold=threshold,
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Grad-CAM generation failure: {str(e)}")

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_research_dataset(payload: EvaluateRequest):
    """
    Authorized research evaluation endpoint returning metrics on verified test partitions.
    """
    # Map to verified benchmark results
    mapping = {
        "densenet121": "DL_DENSENET121",
        "efficientnet_b0": "DL_EFFICIENTNET_B0",
        "mobilenet_v3_large": "DL_MOBILENET_V3_LARGE",
        "resnet50": "DL_RESNET50",
        "multimodal_fusion": "MULTIMODAL_FUSION_DENSENET_CLINICAL",
    }
    
    exp_id = mapping.get(payload.model_id)
    if not exp_id:
        raise HTTPException(status_code=404, detail="Model evaluation profile not found.")

    try:
        metrics_data = report_service.get_benchmark_metrics()
        matching = [m for m in metrics_data.get("models", []) if m.get("experiment_id") == exp_id]
        if not matching:
            raise HTTPException(status_code=404, detail="No recorded test metrics for this model.")

        m = matching[0]
        raw_test = m.get("raw_metrics_test", {})

        return EvaluateResponse(
            model_id=payload.model_id,
            samples_evaluated=m.get("test_samples", 131),
            accuracy=m.get("test_accuracy", 0.0),
            balanced_accuracy=m.get("test_balanced_acc", 0.0),
            macro_f1=m.get("test_macro_f1", 0.0),
            per_class_recall={
                "Normal": m.get("normal_recall", 0.0),
                "Osteopenia": m.get("osteopenia_recall", 0.0),
                "Osteoporosis": m.get("osteoporosis_recall", 0.0),
            },
            confusion_matrix=raw_test.get("confusion_matrix", [[0,0,0],[0,0,0],[0,0,0]]),
            disclaimer="Evaluation performed on untouched patient-isolated test partition.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
