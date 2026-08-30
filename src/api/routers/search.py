from fastapi import APIRouter, Depends, Request
from src.api.schemas import SearchRequest, SearchResponse, SearchResult
from src.inference.clip_search import CLIPSearcher

router = APIRouter()


def get_searcher(request: Request) -> CLIPSearcher:
    return request.app.state.searcher


@router.post("/search", response_model=SearchResponse)
def search_images(body: SearchRequest, searcher: CLIPSearcher = Depends(get_searcher)):

    raw_results = searcher.search(body.query, body.top_k)
    results = [SearchResult(path=r['path'], score=r['score'])
               for r in raw_results]
    return SearchResponse(results=results)
