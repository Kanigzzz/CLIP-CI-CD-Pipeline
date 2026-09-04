import time
import logging
from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, Request, HTTPException
from src.api.schemas import SearchRequest, SearchResponse, SearchResult
if TYPE_CHECKING:
    from src.inference.clip_search import CLIPSearcher

logger = logging.getLogger(__name__)
router = APIRouter()


def get_searcher(request: Request) -> "CLIPSearcher":
    return request.app.state.searcher


@router.post("/search", response_model=SearchResponse)
def search_images(body: SearchRequest, searcher: "CLIPSearcher" = Depends(get_searcher)):
    if not body.query or not body.query.strip():
        logger.warning("Empty search query received",
                       extra={"top_k": body.top_k})
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty or contain only whitespace."
        )
    start_time = time.perf_counter()
    try:
        raw_results = searcher.search(body.query, body.top_k)
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        results = [SearchResult(path=r['path'], score=r['score'])
                   for r in raw_results]

        logger.info(
            "Vector search completed successfully",
            extra={
                "query": body.query,
                "top_k": body.top_k,
                "latency": latency
            }
        )
        return SearchResponse(results=results)
    except Exception as e:
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception("Vector search inference failure",
                         extra={"query": body.query, "top_k": body.top_k, "latency": latency})
        raise HTTPException(
            status_code=500,
            detail="Internal error occurred while searching for images."
        )
