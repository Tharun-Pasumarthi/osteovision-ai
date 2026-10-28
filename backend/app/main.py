"""
FastAPI Main Application Entry Point for Knee X-Ray Osteoporosis Classification.
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
import logging

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.app.config import settings
from backend.app.services.model_registry import registry
from backend.app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("osteoporosis_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager preloading models and resources on startup."""
    logger.info("Initializing Osteoporosis Deep Learning API server...")
    try:
        registry.load_all()
        logger.info(f"Model initialization complete. {len(registry.loaded_models)} models loaded in memory.")
    except Exception as e:
        logger.error(f"Failed to initialize models during startup: {e}", exc_info=True)
    yield
    logger.info("Shutting down Osteoporosis Deep Learning API server...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multimodal Deep Learning System for Osteoporosis, Osteopenia, and Normal classification from knee radiographs.",
    lifespan=lifespan,
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs figures directory if available
if settings.FIGURES_DIR.exists():
    app.mount("/outputs/figures", StaticFiles(directory=str(settings.FIGURES_DIR)), name="figures")

# Mount frontend production build assets if available
if (settings.FRONTEND_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(settings.FRONTEND_DIST_DIR / "assets")), name="frontend_assets")

# Include API routes
app.include_router(router)

@app.get("/")
async def root(request: Request):
    accept = request.headers.get("accept", "")
    index_file = settings.FRONTEND_DIST_DIR / "index.html"
    if "text/html" in accept and index_file.exists():
        return FileResponse(index_file)
    return {
        "project": "Multimodal Deep Learning for Knee X-Ray-Based Osteoporosis Classification",
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs",
        "health": "/api/health",
        "models": "/api/models",
    }

@app.get("/{full_path:path}")
async def catch_all_spa(full_path: str, request: Request):
    """Fallback handler for Single Page Application client-side routing."""
    if full_path.startswith("api") or full_path.startswith("outputs") or full_path.startswith("docs") or full_path == "openapi.json":
        return JSONResponse(status_code=404, content={"detail": f"Path '/{full_path}' not found."})
    index_file = settings.FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
