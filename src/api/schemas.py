from pydantic import BaseModel

class CaptionResponse(BaseModel):
    caption: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    path: str
    score: float

class SearchResponse(BaseModel):
    results: list[SearchResult]