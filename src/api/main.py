from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.api.routers import caption, search
from src.inference.caption_generate import _captioner
from src.inference.clip_search import searcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = _captioner
    _ = searcher
    yield

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
