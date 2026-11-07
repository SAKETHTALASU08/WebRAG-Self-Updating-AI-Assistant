import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.schemas import (
    ChatRequest,
    ChunkInspection,
    Citation,
    CrawlStats,
    CrawlStatusResponse,
    QueryRequest,
    QueryResponse,
    SiteCreateRequest,
    SiteSummary,
    SourceItem,
)
from app.rag.pipeline import RagPipeline
from app.storage.site_manager import SiteManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared singletons
site_manager = SiteManager()
rag_pipeline = RagPipeline(site_manager=site_manager)

# In-memory tracking for active crawls: site_id -> crawl state
crawl_states: Dict[str, dict] = {}


def _execute_crawl(site_id: str, max_depth: int, max_pages: int) -> None:
    """Worker function executed in background to crawl and incrementally index a site."""
    meta = site_manager.read_metadata(site_id)
    if not meta:
        logger.error(f"Cannot crawl unknown site {site_id}")
        return

    root_url = meta["root_url"]
    crawl_states[site_id] = {
        "is_crawling": True,
        "status": "crawling",
        "current_url": root_url,
        "pages_crawled": 0,
        "pages_total": max_pages,
        "pages_unchanged": 0,
        "pages_updated": 0,
        "chunks_added": 0,
        "chunks_unchanged": 0,
        "chunks_removed": 0,
        "time_saved_seconds": 0.0,
        "error": None,
        "started_at": time.time(),
    }

    def progress_callback(url: str, current: int, total: int):
        if site_id in crawl_states:
            crawl_states[site_id]["current_url"] = url
            crawl_states[site_id]["pages_crawled"] = current
            crawl_states[site_id]["pages_total"] = total

    try:
        report = rag_pipeline.crawl_and_index_site(
            site_id=site_id,
            max_depth=max_depth,
            max_pages=max_pages,
            on_progress=progress_callback,
        )
        if site_id in crawl_states:
            crawl_states[site_id]["chunks_added"] = report.chunks_added
            crawl_states[site_id]["chunks_unchanged"] = report.chunks_unchanged
            crawl_states[site_id]["chunks_removed"] = report.chunks_removed
            crawl_states[site_id]["pages_updated"] = report.pages_updated
            crawl_states[site_id]["pages_unchanged"] = report.pages_unchanged
            crawl_states[site_id]["pages_crawled"] = report.pages_crawled + report.pages_skipped_304
            crawl_states[site_id]["time_saved_seconds"] = report.time_saved_seconds
            crawl_states[site_id]["is_crawling"] = False
            crawl_states[site_id]["status"] = "completed"

    except Exception as e:
        logger.error(f"Error during crawl of site {site_id}: {e}", exc_info=True)
        if site_id in crawl_states:
            crawl_states[site_id]["is_crawling"] = False
            crawl_states[site_id]["status"] = "failed"
            crawl_states[site_id]["error"] = str(e)


# ---------------- API Endpoints ---------------- #

@router.get(
    "/health",
    summary="System Health & LLM Status",
    description="Check system health, Ollama LLM service connectivity, and sentence-transformers device.",
    tags=["System"],
)
def get_health():
    """System health and local LLM connectivity check."""
    ollama_alive = rag_pipeline.ollama.check_health()
    available_models = rag_pipeline.ollama.list_models() if ollama_alive else []

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ollama": {
            "connected": ollama_alive,
            "host": settings.OLLAMA_HOST,
            "target_model": settings.OLLAMA_MODEL,
            "available_models": available_models,
        },
        "embedding": {
            "model": settings.EMBEDDING_MODEL_NAME,
            "dimension": settings.EMBEDDING_DIM,
            "device": getattr(rag_pipeline.embedder, "_device", "cpu"),
        },
    }


@router.get(
    "/sites",
    response_model=List[SiteSummary],
    summary="List All Knowledge Bases",
    description="Retrieve a list of all configured website knowledge bases and their indexing stats.",
    tags=["Sites"],
)
def list_sites():
    """List all configured website knowledge bases."""
    return site_manager.list_sites()


@router.post(
    "/sites",
    response_model=SiteSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Create Site & Start Background Crawl",
    description="Register root URL, create isolated knowledge base folder with SQLite and FAISS, and trigger crawl+index in the background.",
    tags=["Sites"],
)
def create_site(request: SiteCreateRequest, background_tasks: BackgroundTasks):
    """Register a new website root URL, create isolated knowledge base, and trigger initial crawl."""
    summary = site_manager.create_or_get_site(
        root_url=request.root_url,
        name=request.name,
        max_depth=request.max_depth or settings.MAX_CRAWL_DEPTH,
        max_pages=request.max_pages or settings.MAX_CRAWL_PAGES,
    )
    # Automatically kick off initial crawl in background
    background_tasks.add_task(
        _execute_crawl,
        summary.id,
        request.max_depth or settings.MAX_CRAWL_DEPTH,
        request.max_pages or settings.MAX_CRAWL_PAGES,
    )
    return summary


@router.get(
    "/sites/{site_id}",
    response_model=SiteSummary,
    summary="Get Site Details",
    description="Retrieve metadata, crawl status, and statistics for a single website knowledge base.",
    tags=["Sites"],
)
def get_site(site_id: str):
    """Retrieve details and statistics for a specific website knowledge base."""
    site = site_manager.get_site(site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site knowledge base not found")
    return site


@router.delete(
    "/sites/{site_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Site Knowledge Base",
    description="Permanently deletes the site directory, SQLite database, and FAISS index.",
    tags=["Sites"],
)
def delete_site(site_id: str):
    """Delete an isolated knowledge base and all its crawled data/vectors."""
    success = site_manager.delete_site(site_id)
    if not success:
        raise HTTPException(status_code=404, detail="Site not found or could not be deleted")
    crawl_states.pop(site_id, None)
    return {"message": f"Site {site_id} deleted successfully", "site_id": site_id}


@router.post(
    "/sites/{site_id}/refresh",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Incremental Re-Crawl & Update",
    description="Trigger an incremental re-crawl using conditional requests (ETag/Last-Modified) and SHA-256 chunk-level change detection.",
    tags=["Crawl & Refresh"],
)
@router.post("/sites/{site_id}/crawl", status_code=status.HTTP_202_ACCEPTED, include_in_schema=False)
def refresh_site(site_id: str, background_tasks: BackgroundTasks):
    """Trigger an incremental update for the website knowledge base."""
    site = site_manager.get_site(site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site knowledge base not found")

    state = crawl_states.get(site_id)
    if state and state.get("is_crawling"):
        return {
            "message": "Crawl already in progress",
            "site_id": site_id,
            "status": "crawling",
        }

    meta = site_manager.read_metadata(site_id) or {}
    max_depth = meta.get("max_depth", settings.MAX_CRAWL_DEPTH)
    max_pages = meta.get("max_pages", settings.MAX_CRAWL_PAGES)

    background_tasks.add_task(_execute_crawl, site_id, max_depth, max_pages)
    return {
        "message": "Incremental refresh initiated",
        "site_id": site_id,
        "status": "accepted",
    }


@router.get(
    "/sites/{site_id}/status",
    response_model=CrawlStatusResponse,
    summary="Get Site Crawl & Indexing Status",
    description="Check the real-time background crawl status, current URL, counts, or the last completed run stats.",
    tags=["Crawl & Refresh"],
)
@router.get("/sites/{site_id}/crawl/status", response_model=CrawlStatusResponse, include_in_schema=False)
def get_site_status(site_id: str):
    """Check the real-time progress or last result of a crawl for this site."""
    site = site_manager.get_site(site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site knowledge base not found")

    state = crawl_states.get(site_id)
    db = site_manager.get_database(site_id)
    db_stats = db.get_stats()
    last_run = db_stats.get("last_run")

    if state and state.get("is_crawling"):
        return CrawlStatusResponse(
            site_id=site_id,
            is_crawling=True,
            status="crawling",
            current_url=state.get("current_url"),
            pages_crawled=state.get("pages_crawled", 0),
            pages_total=state.get("pages_total", 0),
            stats=CrawlStats(
                pages_crawled=state.get("pages_crawled", 0),
                pages_unchanged=state.get("pages_unchanged", 0),
                pages_updated=state.get("pages_updated", 0),
                chunks_added=state.get("chunks_added", 0),
                chunks_unchanged=state.get("chunks_unchanged", 0),
                chunks_removed=state.get("chunks_removed", 0),
                pages_skipped_304=state.get("pages_skipped_304", 0),
                time_saved_seconds=state.get("time_saved_seconds", 0.0),
                status="crawling",
            ),
        )

    # Return last recorded crawl run
    if last_run:
        return CrawlStatusResponse(
            site_id=site_id,
            is_crawling=False,
            status=last_run.get("status", "completed"),
            pages_crawled=last_run.get("pages_crawled", 0),
            pages_total=last_run.get("pages_discovered", 0),
            stats=CrawlStats(
                pages_discovered=last_run.get("pages_discovered", 0),
                pages_crawled=last_run.get("pages_crawled", 0),
                pages_skipped_304=last_run.get("pages_skipped_304", 0),
                pages_unchanged=last_run.get("pages_unchanged", 0),
                pages_updated=last_run.get("pages_updated", 0),
                chunks_added=last_run.get("chunks_added", 0),
                chunks_unchanged=last_run.get("chunks_unchanged", 0),
                chunks_removed=last_run.get("chunks_removed", 0),
                duration_seconds=last_run.get("duration_seconds", 0.0),
                time_saved_seconds=last_run.get("time_saved_seconds", 0.0),
                status=last_run.get("status", "completed"),
                error=last_run.get("error"),
            ),
            error=last_run.get("error"),
        )

    return CrawlStatusResponse(
        site_id=site_id,
        is_crawling=False,
        status="idle",
    )


@router.post(
    "/sites/{site_id}/chat",
    summary="Chat With Website (Streaming Answer + Sources)",
    description="Stream answers strictly grounded in website context blocks, citing [1], [2], with source URLs and snippets.",
    tags=["Chat & Query"],
)
def chat_with_site(site_id: str, req: ChatRequest):
    """
    Stream answer tokens and sources metadata for a user question.
    Emits text/event-stream:
    - event: sources (JSON array of {url, title, snippet, page_no})
    - event: token ({"token": string})
    - event: done ({})
    """
    site = site_manager.get_site(site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site knowledge base not found")

    top_k = req.top_k or settings.SIMILARITY_TOP_K
    threshold = req.similarity_threshold if req.similarity_threshold is not None else settings.SIMILARITY_THRESHOLD

    try:
        sources, token_stream = rag_pipeline.query_stream(
            site_id=site_id,
            query=req.query,
            top_k=top_k,
            similarity_threshold=threshold,
            model=req.model,
            rerank=req.rerank,
        )

        def event_generator():
            # First event emits sources list with exact required schema {url, title, snippet, page_no}
            sources_payload = [
                {
                    "url": s.url,
                    "title": s.title,
                    "snippet": s.snippet,
                    "page_no": s.page_no,
                }
                for s in sources
            ]
            yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"
            yield f"event: citations\ndata: {json.dumps(sources_payload)}\n\n"

            for token in token_stream:
                payload = json.dumps({"token": token})
                yield f"event: token\ndata: {payload}\n\n"

            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Chat streaming error for site {site_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/sites/{site_id}/query",
    response_model=QueryResponse,
    summary="Query Knowledge Base (JSON)",
    description="Non-streaming RAG query returning {answer, sources:[{url, title, snippet, page_no}]}.",
    tags=["Chat & Query"],
)
def query_knowledge_base(site_id: str, req: QueryRequest):
    """Ask a question against the site's isolated vector knowledge base."""
    site = site_manager.get_site(site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site knowledge base not found")

    top_k = req.top_k or settings.SIMILARITY_TOP_K
    threshold = req.similarity_threshold if req.similarity_threshold is not None else settings.SIMILARITY_THRESHOLD

    try:
        response = rag_pipeline.query(
            site_id=site_id,
            query=req.query,
            top_k=top_k,
            similarity_threshold=threshold,
            model=req.model,
            rerank=req.rerank,
        )
        return response
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error querying site {site_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/sites/{site_id}/query/stream",
    summary="Streaming Query Alias",
    include_in_schema=False,
)
def query_knowledge_base_stream(site_id: str, req: QueryRequest):
    """Backward compatibility alias for /sites/{site_id}/chat."""
    chat_req = ChatRequest(**req.model_dump())
    return chat_with_site(site_id, chat_req)


@router.get(
    "/sites/{site_id}/chunks",
    response_model=List[ChunkInspection],
    summary="Inspect Site Chunks",
    description="Inspect stored chunks and their SHA-256 hashes for transparency.",
    tags=["Inspection"],
)
def get_site_chunks(site_id: str, limit: int = Query(50, ge=1, le=200)):
    """Inspect stored chunks and their SHA-256 hashes for transparency."""
    site = site_manager.get_site(site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site knowledge base not found")

    db = site_manager.get_database(site_id)
    all_chunks = db.get_all_chunks()[:limit]

    return [
        ChunkInspection(
            id=c["id"],
            url=c["url"],
            title=c["title"],
            chunk_index=c["chunk_index"],
            page_number=c.get("page_number"),
            chunk_hash=c["chunk_hash"],
            text=c["text"],
            char_count=len(c["text"]),
            created_at=c["created_at"],
        )
        for c in all_chunks
    ]
