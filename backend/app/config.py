"""
Configuration settings for the Osteoporosis Classification API.
"""
import os
from pathlib import Path
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseModel):
    APP_NAME: str = "Multimodal Knee X-Ray Osteoporosis Classification API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    CHECKPOINTS_DIR: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    REPORTS_DIR: Path = PROJECT_ROOT / "outputs" / "reports"
    FIGURES_DIR: Path = PROJECT_ROOT / "outputs" / "figures"
    GRADCAM_DIR: Path = PROJECT_ROOT / "outputs" / "gradcam_examples"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    FRONTEND_DIST_DIR: Path = PROJECT_ROOT / "frontend" / "dist"
    
    # Model configuration
    DEVICE: str = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"
    DEFAULT_MODEL_ID: str = "adaptive_fusion"
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]

    def get_allowed_origins(self) -> list[str]:
        env_cors = os.environ.get("CORS_ORIGINS")
        if env_cors:
            return [origin.strip() for origin in env_cors.split(",") if origin.strip()]
        return self.CORS_ORIGINS
    
    # Upload limits
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}

settings = Settings()
