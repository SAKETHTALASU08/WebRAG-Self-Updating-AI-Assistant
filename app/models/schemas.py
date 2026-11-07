from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class SiteCreateRequest(BaseModel):
    root_url: str = Field(..., description="Root URL to crawl and build the knowledge base from")
    name: Optional[str] = Field(None, description="Optional friendly name for this knowledge base")
    max_depth: Optional[int] = Field(3, ge=1, le=10, description="Max crawling depth from root")
    max_pages: Optional[int] = Field(50, ge=1, le=500, description="Max pages to crawl")


class SiteSummary(BaseModel):
    id: str
    root_url: str
    name: str
    status: str
    created_at: str
    last_crawled_at: Optional[str] = None
    total_pages: int = 0
    total_chunks: int = 0


class CrawlStats(BaseModel):
    pages_discovered: int = 0
    pages_crawled: int = 0
    pages_skipped_304: int = 0
    pages_unchanged: int = 0
    pages_updated: int = 0
    chunks_added: int = 0
    chunks_unchanged: int = 0
    chunks_removed: int = 0
    duration_seconds: float = 0.0
    time_saved_seconds: float = 0.0
    status: str = "completed"
    error: Optional[str] = None


class CrawlSummaryReport(BaseModel):
    site_id: str
    root_url: str
    pages_discovered: int = 0
    pages_crawled: int = 0
    pages_skipped_304: int = 0
    pages_unchanged: int = 0
    pages_updated: int = 0
    chunks_added: int = 0
    chunks_unchanged: int = 0
    chunks_removed: int = 0
    total_active_chunks: int = 0
    total_active_pages: int = 0
    duration_seconds: float = 0.0
    time_saved_seconds: float = 0.0
    status: str = "completed"
    error: Optional[str] = None


class CrawlStatusResponse(BaseModel):
    site_id: str
    is_crawling: bool
    status: str
    current_url: Optional[str] = None
    pages_crawled: int = 0
    pages_total: int = 0
    stats: Optional[CrawlStats] = None
    error: Optional[str] = None


class Citation(BaseModel):
    url: str
    title: str
    chunk_index: int
    page_number: Optional[int] = None
    similarity_score: float
    snippet: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    model: Optional[str] = Field(None, description="Optional override for Ollama model")


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    latency_ms: float
    chunks_retrieved: int


class ChunkInspection(BaseModel):
    id: str
    url: str
    title: str
    chunk_index: int
    page_number: Optional[int] = None
    chunk_hash: str
    text: str
    char_count: int
    created_at: str
