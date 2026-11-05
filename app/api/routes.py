import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.config import settings
from app.crawler.crawler import WebCrawler
from app.models.schemas import (
    ChunkInspection,
    Citation,
    CrawlStats,
    CrawlStatusResponse,
    QueryRequest,
    QueryResponse,
    SiteCreateRequest,
    SiteSummary,
)
from app.rag.pipeline import RagPipeline
from app.storage.site_manager import SiteManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

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
    db = site_manager.get_database(site_id)
    run_id = db.record_crawl_start()

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
        "error": None,
        "started_at": time.time(),
    }

    meta["status"] = "crawling"
    site_manager.write_metadata(site_id, meta)

    crawler = WebCrawler(
        root_url=root_url,
        max_depth=max_depth,
        max_pages=max_pages,
        rate_limit_delay=settings.CRAWL_DELAY_SECONDS,
    )

    pages_crawled_count = 0
    pages_updated_count = 0
    pages_unchanged_count = 0
    total_chunks_added = 0
    total_chunks_unchanged = 0
    total_chunks_removed = 0

    def progress_callback(url: str, current: int, total: int):
        if site_id in crawl_states:
            crawl_states[site_id]["current_url"] = url
            crawl_states[site_id]["pages_crawled"] = current
            crawl_states[site_id]["pages_total"] = total

    try:
        for doc in crawler.crawl_documents(on_page_crawled=progress_callback):
            pages_crawled_count += 1
            added, unchanged, removed = rag_pipeline.process_document_incremental(site_id, doc)
            
            total_chunks_added += added
            total_chunks_unchanged += unchanged
            total_chunks_removed += removed

            if added > 0 or removed > 0:
                pages_updated_count += 1
            else:
                pages_unchanged_count += 1

            if site_id in crawl_states:
                crawl_states[site_id]["chunks_added"] = total_chunks_added
                crawl_states[site_id]["chunks_unchanged"] = total_chunks_unchanged
                crawl_states[site_id]["chunks_removed"] = total_chunks_removed
                crawl_states[site_id]["pages_updated"] = pages_updated_count
                crawl_states[site_id]["pages_unchanged"] = pages_unchanged_count

        duration = time.time() - crawl_states[site_id]["started_at"]
        db.record_crawl_finish(
            run_id=run_id,
            pages_discovered=len(crawler.visited_urls),
            pages_crawled=pages_crawled_count,
            pages_unchanged=pages_unchanged_count,
            pages_updated=pages_updated_count,
            chunks_added=total_chunks_added,
            chunks_unchanged=total_chunks_unchanged,
            chunks_removed=total_chunks_removed,
            status="completed",
        )

        meta["status"] = "ready"
        meta["last_crawled_at"] = datetime.now(timezone.utc).isoformat()
        site_manager.write_metadata(site_id, meta)

        crawl_states[site_id]["is_crawling"] = False
        crawl_states[site_id]["status"] = "completed"
        logger.info(f"Crawl completed for site {site_id} in {duration:.1f}s")

    except Exception as e:
        logger.error(f"Error during crawl of site {site_id}: {e}", exc_info=True)
        db.record_crawl_finish(
            run_id=run_id,
            pages_discovered=0,
            pages_crawled=pages_crawled_count,
            pages_unchanged=pages_unchanged_count,
            pages_updated=pages_updated_count,
            chunks_added=total_chunks_added,
            chunks_unchanged=total_chunks_unchanged,
            chunks_removed=total_chunks_removed,
            status="failed",
            error=str(e),
        )
        meta["status"] = "failed"
        site_manager.write_metadata(site_id, meta)
        if site_id in crawl_states:
            crawl_states[site_id]["is_crawling"] = False
            crawl_states[site_id]["status"] = "failed"
            crawl_states[site_id]["error"] = str(e)


# ---------------- API Endpoints ---------------- #

@router.get("/health")
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
            "device": rag_pipeline.embedder._device,
        },
    }


@router.get("/sites", response_model=List[SiteSummary])
def list_sites():
    """List all configured website knowledge bases."""
    return site_manager.list_sites()


@router.post("/sites", response_model=SiteSummary, status_code=status.HTTP_201_CREATED)
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


@router.get("/sites/{site_id}", response_model=SiteSummary)
def get_site(site_id: str):
    """Retrieve details and statistics for a specific website knowledge base."""
    site = site_manager.get_site(site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site knowledge base not found")
    return site


@router.delete("/sites/{site_id}", status_code=status.HTTP_200_OK)
def delete_site(site_id: str):
    """Delete an isolated knowledge base and all its crawled data/vectors."""
    success = site_manager.delete_site(site_id)
    if not success:
        raise HTTPException(status_code=404, detail="Site not found or could not be deleted")
    crawl_states.pop(site_id, None)
    return {"message": f"Site {site_id} deleted successfully"}


@router.post("/sites/{site_id}/crawl", status_code=status.HTTP_202_ACCEPTED)
def trigger_crawl(site_id: str, background_tasks: BackgroundTasks):
    """Trigger an incremental crawl/update for the site (SHA-256 chunk-level change detection)."""
    site = site_manager.get_site(site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site knowledge base not found")

    state = crawl_states.get(site_id)
    if state and state.get("is_crawling"):
        return {"message": "Crawl already in progress", "site_id": site_id}

    meta = site_manager.read_metadata(site_id) or {}
    max_depth = meta.get("max_depth", settings.MAX_CRAWL_DEPTH)
    max_pages = meta.get("max_pages", settings.MAX_CRAWL_PAGES)

    background_tasks.add_task(_execute_crawl, site_id, max_depth, max_pages)
    return {"message": "Incremental crawl initiated", "site_id": site_id}


@router.get("/sites/{site_id}/crawl/status", response_model=CrawlStatusResponse)
def get_crawl_status(site_id: str):
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
                pages_unchanged=last_run.get("pages_unchanged", 0),
                pages_updated=last_run.get("pages_updated", 0),
                chunks_added=last_run.get("chunks_added", 0),
                chunks_unchanged=last_run.get("chunks_unchanged", 0),
                chunks_removed=last_run.get("chunks_removed", 0),
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


@router.post("/sites/{site_id}/query", response_model=QueryResponse)
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
        )
        return response
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error querying site {site_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sites/{site_id}/query/stream")
def query_knowledge_base_stream(site_id: str, req: QueryRequest):
    """Server-Sent Events (SSE) streaming response with citations and live tokens."""
    site = site_manager.get_site(site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site knowledge base not found")

    top_k = req.top_k or settings.SIMILARITY_TOP_K
    threshold = req.similarity_threshold if req.similarity_threshold is not None else settings.SIMILARITY_THRESHOLD

    try:
        citations, token_stream = rag_pipeline.query_stream(
            site_id=site_id,
            query=req.query,
            top_k=top_k,
            similarity_threshold=threshold,
            model=req.model,
        )

        def event_generator():
            # First event emits citations metadata
            citations_data = [c.model_dump() for c in citations]
            yield f"event: citations\ndata: {json.dumps(citations_data)}\n\n"

            # Stream LLM tokens
            for token in token_stream:
                payload = json.dumps({"token": token})
                yield f"event: token\ndata: {payload}\n\n"

            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Stream error for site {site_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sites/{site_id}/chunks", response_model=List[ChunkInspection])
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
