from pydantic import BaseModel, Field


class CaptionResponse(BaseModel):
    caption: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=100)


class SearchResult(BaseModel):
    path: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
