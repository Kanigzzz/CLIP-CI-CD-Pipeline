import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from src.api.routers import caption, search
from src.utils.logger import setup_logging

ROOT = Path(__file__).resolve().parent.parent.parent

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing models and vector database...")
    from src.model.image_captioner import ImageCaptioner
    from src.inference.clip_search import CLIPSearcher

    app.state.searcher = CLIPSearcher()
    app.state.captioner = ImageCaptioner()
    logger.info(
        "Models initialized successfully. Application is ready to receive requests.")
    yield
    logger.info("Shutting down aplication...")
    del app.state.searcher
    del app.state.captioner

app = FastAPI(title="CLIP | BLIP API", version="1.0.0", lifespan=lifespan)
app.mount("/animals", StaticFiles(directory=(ROOT /
          "data" / "animals"), check_dir=False), name="animals")

app.include_router(caption.router, prefix="/api/v1", tags=['caption'])
app.include_router(search.router, prefix="/api/v1", tags=['search'])


@app.get("/live", tags=['monitoring'])
def liveness_check():
    return {"status": "alive"}


@app.get("/ready", tags=["monitoring"])
def readiness_check(request: Request):
    if not hasattr(request.app.state, "searcher") or request.app.state.searcher.index is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    model_revision = os.getenv("MODEL_REVISION", "unknown")

    return {
        "status": "ready",
        "model_revision": model_revision,
        "index_size": request.app.state.searcher.index.ntotal
    }
