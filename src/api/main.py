from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.api.routers import caption, search
from src.model.image_captioner import ImageCaptioner
from src.inference.clip_search import CLIPSearcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.searcher = CLIPSearcher()
    app.state.captioner = ImageCaptioner()
    yield
    del app.state.searcher
    del app.state.captioner

app = FastAPI(title="CLIP | BLIP API", version="1.0.0", lifespan=lifespan)
app.mount("/animals", StaticFiles(directory="data/animals"), name="animals")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(caption.router, prefix="/api/v1", tags=['caption'])
app.include_router(search.router, prefix="/api/v1", tags=['search'])


@app.get("/health", tags=['monitoring'])
def health_check():
    return {"status": "healthy", "service": "clip-api", "version": "1.0.0"}
