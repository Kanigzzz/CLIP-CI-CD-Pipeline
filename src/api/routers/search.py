from fastapi import APIRouter
from src.api.schemas import SearchRequest, SearchResponse, SearchResult
from src.inference.clip_search import searcher

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    raw_results = searcher.search(request.query, request.top_k)
    results = [SearchResult(path=r['path'], score=r['score']) for r in raw_results]
    return SearchResponse(results=results)