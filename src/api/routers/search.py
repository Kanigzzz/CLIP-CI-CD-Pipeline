import logging
from fastapi import APIRouter, Depends, Request, HTTPException
from src.api.schemas import SearchRequest, SearchResponse, SearchResult
from src.inference.clip_search import CLIPSearcher

logger = logging.getLogger(__name__)
router = APIRouter()


def get_searcher(request: Request) -> CLIPSearcher:
    return request.app.state.searcher


@router.post("/search", response_model=SearchResponse)
def search_images(body: SearchRequest, searcher: CLIPSearcher = Depends(get_searcher)):
    if not body.query or not body.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty or contain only whitespace."
        )
    try:
        raw_results = searcher.search(body.query, body.top_k)
        results = [SearchResult(path=r['path'], score=r['score'])
                   for r in raw_results]
        return SearchResponse(results=results)
    except Exception as e:
        logger.exception(
            f"Unexpected error during vector search of query: {body.query} : {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal error occurred while searching for images."
        )
