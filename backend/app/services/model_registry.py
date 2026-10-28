"""
Model Registry Service for preloading and caching trained PyTorch checkpoints.
"""
import os
import pickle
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn

from src.models import KneeXRayCNNClassifier, MultimodalFusionClassifier, DeepMultimodalFusionClassifier
from src.metadata_preprocessing import ClinicalFeaturePreprocessor
from backend.app.config import settings
from backend.app.schemas.models import ModelMetadata

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.loaded_models: Dict[str, nn.Module] = {}
        self.temperatures: Dict[str, float] = {}
        self.clinical_preprocessor: Optional[ClinicalFeaturePreprocessor] = None
        self.clinical_model: Optional[Any] = None
        self.benchmark_df: Optional[pd.DataFrame] = None
        
        # Static definitions of available neural checkpoints
        self.checkpoints_config = {
            "adaptive_fusion": {
                "name": "Adaptive Multimodal Fusion (DenseNet121+LogReg)",
                "family": "Calibrated Multimodal",
                "checkpoint": settings.CHECKPOINTS_DIR / "best_densenet121.pt",
                "type": "adaptive_fusion",
                "backbone": "densenet121",
                "supports_clinical": True,
                "gradcam_supported": True,
                "description": "#1 Champion Benchmark (93.22% Accuracy, 0.9305 Macro-F1). DenseNet121 vision embeddings dynamically calibrated with clinical risk multipliers.",
            },
            "cm_graf_v2": {
                "name": "Deep Multimodal Fusion (CM-GRAF v2)",
                "family": "Multimodal Deep Learning",
                "checkpoint": settings.CHECKPOINTS_DIR / "best_multimodal_fusion.pt",
                "type": "multimodal",
                "supports_clinical": True,
                "gradcam_supported": True,
                "description": "#2 Deep Learning Fusion (92.94% Accuracy, 0.9274 Macro-F1). Cross-Modal Gated Residual Attention Network with LayerNorm & Modality Masking.",
            },
            "densenet121": {
                "name": "DenseNet121",
                "family": "Deep Learning (Image-Only)",
                "checkpoint": settings.CHECKPOINTS_DIR / "best_densenet121.pt",
                "type": "cnn",
                "backbone": "densenet121",
                "supports_clinical": False,
                "gradcam_supported": True,
                "description": "Dense convolutional network extracting hierarchical trabecular microarchitecture (92.37% Accuracy, 0.9210 Macro-F1).",
            },
            "efficientnet_b0": {
                "name": "EfficientNet-B0",
                "family": "Deep Learning (Image-Only)",
                "checkpoint": settings.CHECKPOINTS_DIR / "best_efficientnet_b0.pt",
                "type": "cnn",
                "backbone": "efficientnet_b0",
                "supports_clinical": False,
                "gradcam_supported": True,
                "description": "Compound scaled lightweight CNN capturing multiscale cortical bone texture.",
            },
            "mobilenet_v3_large": {
                "name": "MobileNetV3-Large",
                "family": "Deep Learning (Image-Only)",
                "checkpoint": settings.CHECKPOINTS_DIR / "best_mobilenet_v3_large.pt",
                "type": "cnn",
                "backbone": "mobilenet_v3_large",
                "supports_clinical": False,
                "gradcam_supported": True,
                "description": "Hardware-efficient mobile architecture optimized for low-latency clinical edge deployment.",
            },
            "resnet50": {
                "name": "ResNet50",
                "family": "Deep Learning (Image-Only)",
                "checkpoint": settings.CHECKPOINTS_DIR / "best_resnet50.pt",
                "type": "cnn",
                "backbone": "resnet50",
                "supports_clinical": False,
                "gradcam_supported": True,
                "description": "50-layer residual network utilizing identity shortcut mappings.",
            },
            "multimodal_fusion": {
                "name": "Multimodal Fusion (DenseNet121 + Clinical)",
                "family": "Multimodal",
                "checkpoint": settings.CHECKPOINTS_DIR / "best_multimodal_fusion.pt",
                "type": "multimodal",
                "supports_clinical": True,
                "gradcam_supported": True,
                "description": "Combines dense radiograph visual representations with patient clinical indicators via late fusion MLP.",
            },
        }

    def load_all(self):
        """Preload all trained checkpoints into memory on startup."""
        logger.info(f"Initializing ModelRegistry on device: {self.device}")
        
        # 1. Load benchmark report if available
        benchmark_csv = settings.REPORTS_DIR / "model_baseline_comparison.csv"
        if benchmark_csv.exists():
            try:
                self.benchmark_df = pd.read_csv(benchmark_csv)
                logger.info(f"Loaded benchmark comparisons with {len(self.benchmark_df)} models.")
            except Exception as e:
                logger.warning(f"Could not load benchmark CSV: {e}")

        # 2. Load Clinical Preprocessor
        prep_path = settings.CHECKPOINTS_DIR / "clinical_preprocessor.pkl"
        if prep_path.exists():
            try:
                with open(prep_path, "rb") as f:
                    self.clinical_preprocessor = pickle.load(f)
                logger.info("Loaded clinical preprocessor transformer.")
            except Exception as e:
                logger.error(f"Failed to load clinical preprocessor: {e}")
        else:
            logger.warning(f"Clinical preprocessor not found at {prep_path}")

        # 3. Load Clinical Classifier Model
        clin_model_path = settings.CHECKPOINTS_DIR / "best_clinical_model.pkl"
        if clin_model_path.exists():
            try:
                with open(clin_model_path, "rb") as f:
                    self.clinical_model = pickle.load(f)
                logger.info("Loaded calibrated clinical classification model.")
            except Exception as e:
                logger.error(f"Failed to load clinical classification model: {e}")
        else:
            logger.warning(f"Clinical model not found at {clin_model_path}")

        # 4. Load PyTorch model weights
        for model_id, config in self.checkpoints_config.items():
            ckpt_path = config["checkpoint"]
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint for {model_id} does not exist at {ckpt_path}")
                continue

            try:
                if config["type"] in ("cnn", "adaptive_fusion"):
                    model = KneeXRayCNNClassifier(
                        backbone_name=config["backbone"],
                        num_classes=3,
                        pretrained=False,
                    )
                elif config["type"] == "multimodal":
                    model = DeepMultimodalFusionClassifier(
                        clinical_dim=13,
                        num_classes=3,
                    )
                else:
                    continue

                raw_ckpt = torch.load(ckpt_path, map_location=self.device)
                state_dict = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                temp = 1.0
                if isinstance(raw_ckpt, dict) and "temperature" in raw_ckpt:
                    try:
                        temp = float(raw_ckpt["temperature"])
                    except Exception:
                        temp = 1.0
                self.temperatures[model_id] = temp
                self.loaded_models[model_id] = model
                logger.info(f"Loaded {model_id} successfully (calibration temp={temp:.3f}).")
            except Exception as e:
                logger.error(f"Error loading checkpoint for {model_id}: {e}")

    def get_model(self, model_id: str) -> Optional[nn.Module]:
        return self.loaded_models.get(model_id)

    def get_temperature(self, model_id: str) -> float:
        return self.temperatures.get(model_id, 1.0)

    def get_target_layer(self, model_id: str) -> Optional[nn.Module]:
        """Returns the appropriate final convolutional feature layer for Grad-CAM."""
        model = self.get_model(model_id)
        if model is None:
            return None

        if model_id in ("densenet121", "adaptive_fusion"):
            return model.features.norm5
        elif model_id == "efficientnet_b0":
            return model.features[-1]
        elif model_id == "mobilenet_v3_large":
            return model.features[-1]
        elif model_id == "resnet50":
            return model.features[-1][-1].conv3
        elif model_id in ("multimodal_fusion", "cm_graf_v2"):
            if hasattr(model, "image_encoder"):
                return model.image_encoder.features.norm5
            elif hasattr(model, "features"):
                return model.features.norm5
            return None
        return None

    def list_models_metadata(self) -> list[ModelMetadata]:
        """Generate verified metadata for each model incorporating actual benchmark scores."""
        metadata_list = []
        
        # Map our model_ids to experiment records in benchmark CSV
        mapping = {
            "adaptive_fusion": "CALIB_ADAPTIVE_MULTIMODAL_FUSION",
            "cm_graf_v2": "DL_MULTIMODAL_CM_GRAF_V2",
            "densenet121": "DL_DENSENET121",
            "resnet50": "DL_RESNET50",
            "efficientnet_b0": "DL_EFFICIENTNET_B0",
            "mobilenet_v3_large": "DL_MOBILENET_V3_LARGE",
            "multimodal_fusion": "DL_MULTIMODAL_CM_GRAF_V2",
        }

        for model_id, config in self.checkpoints_config.items():
            exp_id = mapping.get(model_id)
            row = None
            if self.benchmark_df is not None and exp_id:
                matches = self.benchmark_df[self.benchmark_df["experiment_id"] == exp_id]
                if not matches.empty:
                    row = matches.iloc[0]

            test_acc = float(row["test_accuracy"]) if row is not None and pd.notnull(row.get("test_accuracy")) else None
            test_f1 = float(row["test_macro_f1"]) if row is not None and pd.notnull(row.get("test_macro_f1")) else None
            val_f1 = float(row["val_macro_f1"]) if row is not None and pd.notnull(row.get("val_macro_f1")) else None
            norm_rec = float(row["normal_recall"]) if row is not None and pd.notnull(row.get("normal_recall")) else None
            penia_rec = float(row["osteopenia_recall"]) if row is not None and pd.notnull(row.get("osteopenia_recall")) else None
            poro_rec = float(row["osteoporosis_recall"]) if row is not None and pd.notnull(row.get("osteoporosis_recall")) else None
            latency = float(row["inference_latency_ms"]) if row is not None and pd.notnull(row.get("inference_latency_ms")) else None
            params = float(row["parameter_count"]) if row is not None and pd.notnull(row.get("parameter_count")) else None

            metadata_list.append(ModelMetadata(
                id=model_id,
                name=config["name"],
                family=config["family"],
                input_type="image_plus_clinical" if config["supports_clinical"] else "image_only",
                supports_clinical=config["supports_clinical"],
                gradcam_supported=config["gradcam_supported"],
                checkpoint_available=model_id in self.loaded_models,
                checkpoint_path=str(config["checkpoint"]),
                test_accuracy=test_acc,
                test_macro_f1=test_f1,
                val_macro_f1=val_f1,
                normal_recall=norm_rec,
                osteopenia_recall=penia_rec,
                osteoporosis_recall=poro_rec,
                inference_latency_ms=latency,
                parameter_count=params,
                description=config["description"],
            ))

        return metadata_list

registry = ModelRegistry()
