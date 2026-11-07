from datetime import datetime, timezone
import logging
import time
from typing import AsyncGenerator, Callable, Dict, Generator, List, Optional, Tuple

import requests
from app.config import settings
from app.crawler.crawler import CrawlItem, WebCrawler
from app.crawler.extractor import ContentExtractor, ExtractedDocument
from app.models.schemas import Citation, CrawlStats, CrawlSummaryReport, QueryResponse
from app.rag.chunker import DocumentChunk, IncrementalChunker
from app.rag.embedder import LocalEmbedder
from app.rag.ollama_client import OllamaClient
from app.storage.site_manager import SiteManager

logger = logging.getLogger(__name__)


class RagPipeline:
    """Orchestrates incremental indexing (SHA-256 at chunk level) and RAG retrieval with citations."""

    def __init__(
        self,
        site_manager: SiteManager,
        embedder: Optional[LocalEmbedder] = None,
        ollama: Optional[OllamaClient] = None,
        chunker: Optional[IncrementalChunker] = None,
    ):
        self.site_manager = site_manager
        self.embedder = embedder or LocalEmbedder.get_instance()
        self.ollama = ollama or OllamaClient()
        self.chunker = chunker or IncrementalChunker()

    def process_document_incremental(
        self,
        site_id: str,
        doc: ExtractedDocument,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
    ) -> Tuple[int, int, int]:
        """
        Incrementally index a crawled document using SHA-256 chunk hashing.
        Diffs new chunk hashes against stored ones:
        - Classifies chunks into unchanged, new, or removed
        - Embeds ONLY new chunks
        - Deletes removed chunks from SQLite and FAISS
        - Skips unchanged chunks
        Returns: (chunks_added, chunks_unchanged, chunks_removed)
        """
        db = self.site_manager.get_database(site_id)
        vector_store = self.site_manager.get_vector_store(site_id)

        # 1. Chunk document
        new_chunks = self.chunker.chunk_document(doc)
        if not new_chunks:
            # If page is now empty, delete any previously stored chunks for this URL
            existing_chunks = db.get_existing_chunk_hashes_for_url(doc.url)
            if existing_chunks:
                stale_ids = [row["id"] for row in existing_chunks.values()]
                removed_faiss_ids = db.delete_chunks_by_ids(stale_ids)
                vector_store.remove(removed_faiss_ids)
                logger.info(f"Page became empty: removed {len(removed_faiss_ids)} chunks for {doc.url}")
                return (0, 0, len(removed_faiss_ids))
            return (0, 0, 0)

        # 2. Get existing chunks for this URL
        existing_chunks_by_hash = db.get_existing_chunk_hashes_for_url(doc.url)
        existing_hashes = set(existing_chunks_by_hash.keys())
        current_hashes = {c.chunk_hash for c in new_chunks}

        chunks_unchanged = 0
        chunks_to_embed: List[DocumentChunk] = []

        # Find unchanged vs modified/new chunks
        for chunk in new_chunks:
            if chunk.chunk_hash in existing_hashes:
                chunks_unchanged += 1
            else:
                chunks_to_embed.append(chunk)

        # 3. Find deleted/stale chunks for this URL that no longer appear
        stale_chunk_ids: List[str] = []
        for h, row in existing_chunks_by_hash.items():
            if h not in current_hashes:
                stale_chunk_ids.append(row["id"])

        chunks_removed = 0
        if stale_chunk_ids:
            removed_faiss_ids = db.delete_chunks_by_ids(stale_chunk_ids)
            vector_store.remove(removed_faiss_ids)
            chunks_removed = len(removed_faiss_ids)
            logger.info(f"Removed {chunks_removed} obsolete chunks for {doc.url}")

        # 4. Upsert parent page record first so foreign key constraints are satisfied
        import hashlib
        page_hash = hashlib.sha256(doc.raw_text.encode("utf-8")).hexdigest()
        db.upsert_page(
            url=doc.url,
            etag=etag,
            last_modified=last_modified,
            title=doc.title,
            content_type=doc.content_type,
            page_hash=page_hash,
            chunk_count=len(new_chunks),
        )

        # 5. Embed and store new/modified chunks
        chunks_added = 0
        if chunks_to_embed:
            if hasattr(self.embedder, "embed_chunks"):
                vectors = self.embedder.embed_chunks(chunks_to_embed, batch_size=settings.EMBEDDING_BATCH_SIZE)
            else:
                texts = [c.text for c in chunks_to_embed]
                vectors = self.embedder.encode(texts, batch_size=settings.EMBEDDING_BATCH_SIZE)

            current_max_faiss_id = db.get_max_faiss_id()
            assigned_faiss_ids: List[int] = []

            for i, chunk in enumerate(chunks_to_embed):
                faiss_id = current_max_faiss_id + 1 + i
                assigned_faiss_ids.append(faiss_id)

                db.insert_chunk(
                    chunk_id=chunk.chunk_id,
                    site_id=site_id,
                    url=chunk.url,
                    page_no=chunk.page_no,
                    heading=chunk.heading,
                    text=chunk.text,
                    hash=chunk.chunk_hash,
                    embedding_id=faiss_id,
                    status="active",
                )

            vector_store.add(assigned_faiss_ids, vectors)
            chunks_added = len(chunks_to_embed)
            logger.info(
                f"Embedded and saved {chunks_added} new chunks for {doc.url} "
                f"({chunks_unchanged} skipped as unchanged)"
            )

        logger.info(
            f"Chunk diff summary for {doc.url}: {chunks_added} new/embedded, "
            f"{chunks_unchanged} unchanged/skipped, {chunks_removed} removed"
        )
        return (chunks_added, chunks_unchanged, chunks_removed)

    def crawl_and_index_site(
        self,
        site_id: str,
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
        delay_seconds: Optional[float] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        session: Optional[requests.Session] = None,
    ) -> CrawlSummaryReport:
        """
        Execute full incremental re-crawl and re-indexing:
        - Sends conditional requests (ETag / Last-Modified) using stored SQLite page headers
        - Skips unchanged pages (HTTP 304 Not Modified)
        - Diffs chunk hashes for crawled pages (200 OK)
        - Embeds only new chunks, deletes removed ones from FAISS & SQLite, skips unchanged
        - Logs counts and estimated time saved
        - Persists crawl run stats to SQLite
        - Returns structured CrawlSummaryReport
        """
        meta = self.site_manager.read_metadata(site_id)
        if not meta:
            raise ValueError(f"Site '{site_id}' not found.")
        root_url = meta["root_url"]
        db = self.site_manager.get_database(site_id)
        vector_store = self.site_manager.get_vector_store(site_id)

        # 1. Retrieve stored conditional headers (ETag / Last-Modified) from SQLite
        known_headers = db.get_all_pages_headers()

        # 2. Configure crawler
        rate_delay = delay_seconds if delay_seconds is not None else settings.CRAWL_DELAY_SECONDS
        crawler = WebCrawler(
            root_url=root_url,
            max_depth=max_depth or meta.get("max_depth", settings.MAX_CRAWL_DEPTH),
            max_pages=max_pages or meta.get("max_pages", settings.MAX_CRAWL_PAGES),
            rate_limit_delay=rate_delay,
            known_headers=known_headers,
            session=session,
        )

        start_time = time.time()
        run_id = db.record_crawl_start()
        meta["status"] = "crawling"
        self.site_manager.write_metadata(site_id, meta)

        pages_crawled_count = 0
        pages_skipped_304 = 0
        pages_updated_count = 0
        pages_unchanged_count = 0
        total_chunks_added = 0
        total_chunks_unchanged = 0
        total_chunks_removed = 0
        total_embed_time = 0.0

        try:
            items = crawler.crawl_items(on_page_crawled=on_progress)

            for item in items:
                # HTTP 304: page unchanged via conditional headers
                if not item.is_modified or item.status_code == 304:
                    pages_skipped_304 += 1
                    pages_unchanged_count += 1
                    existing_chunks = db.get_existing_chunk_hashes_for_url(item.url)
                    total_chunks_unchanged += len(existing_chunks)
                    db.touch_page(item.url)
                    logger.info(
                        f"Skipped unchanged page (304 Not Modified): {item.url} "
                        f"({len(existing_chunks)} chunks preserved)"
                    )
                    continue

                pages_crawled_count += 1
                try:
                    if item.content_type == "application/pdf":
                        doc = ContentExtractor.extract_pdf(item.url, item.raw_bytes)
                    else:
                        doc = ContentExtractor.extract_html(item.url, item.raw_bytes)
                    if not doc or not doc.raw_text.strip():
                        continue

                    t_embed_0 = time.time()
                    added, unchanged, removed = self.process_document_incremental(
                        site_id=site_id,
                        doc=doc,
                        etag=item.etag,
                        last_modified=item.last_modified,
                    )
                    if added > 0:
                        total_embed_time += (time.time() - t_embed_0)

                    total_chunks_added += added
                    total_chunks_unchanged += unchanged
                    total_chunks_removed += removed

                    if added > 0 or removed > 0:
                        pages_updated_count += 1
                    else:
                        pages_unchanged_count += 1

                except Exception as e:
                    logger.error(f"Error processing document for {item.url}: {e}", exc_info=True)

            # Compute estimated time saved
            # 1. Embedding time saved: avoided embedding unchanged chunks
            if total_chunks_added > 0 and total_embed_time > 0:
                avg_embed_per_chunk = total_embed_time / total_chunks_added
            else:
                avg_embed_per_chunk = 0.025  # standard local embedding latency per chunk on CPU
            embed_time_saved = total_chunks_unchanged * avg_embed_per_chunk

            # 2. Network transfer & DOM parsing time saved for 304 pages
            network_time_saved = pages_skipped_304 * 0.15
            time_saved_seconds = round(embed_time_saved + network_time_saved, 3)
            duration_seconds = round(time.time() - start_time, 3)

            # Log counts and time saved
            logger.info(
                f"=== Re-Crawl Summary Report for site '{site_id}' ===\n"
                f"  Root URL: {root_url}\n"
                f"  Duration: {duration_seconds}s | Estimated Time Saved: {time_saved_seconds}s\n"
                f"  Pages Discovered: {len(crawler.visited_urls)}\n"
                f"  Pages Crawled (200): {pages_crawled_count} | Skipped (304 Not Modified): {pages_skipped_304}\n"
                f"  Pages Updated: {pages_updated_count} | Pages Unchanged: {pages_unchanged_count}\n"
                f"  Chunks Added (New): {total_chunks_added} (embedded)\n"
                f"  Chunks Unchanged: {total_chunks_unchanged} (skipped embedding)\n"
                f"  Chunks Removed: {total_chunks_removed} (deleted from SQLite & FAISS)\n"
                f"  Total Active Chunks in FAISS: {vector_store.total_vectors}"
            )

            # Record crawl finish in SQLite
            db.record_crawl_finish(
                run_id=run_id,
                pages_discovered=len(crawler.visited_urls),
                pages_crawled=pages_crawled_count + pages_skipped_304,
                pages_unchanged=pages_unchanged_count,
                pages_updated=pages_updated_count,
                chunks_added=total_chunks_added,
                chunks_unchanged=total_chunks_unchanged,
                chunks_removed=total_chunks_removed,
                pages_skipped_304=pages_skipped_304,
                duration_seconds=duration_seconds,
                time_saved_seconds=time_saved_seconds,
                status="completed",
            )

            meta["status"] = "idle"
            meta["last_crawled_at"] = datetime.now(timezone.utc).isoformat()
            self.site_manager.write_metadata(site_id, meta)

            stats = db.get_stats()
            return CrawlSummaryReport(
                site_id=site_id,
                root_url=root_url,
                pages_discovered=len(crawler.visited_urls),
                pages_crawled=pages_crawled_count,
                pages_skipped_304=pages_skipped_304,
                pages_unchanged=pages_unchanged_count,
                pages_updated=pages_updated_count,
                chunks_added=total_chunks_added,
                chunks_unchanged=total_chunks_unchanged,
                chunks_removed=total_chunks_removed,
                total_active_chunks=stats["total_chunks"],
                total_active_pages=stats["total_pages"],
                duration_seconds=duration_seconds,
                time_saved_seconds=time_saved_seconds,
                status="completed",
            )

        except Exception as e:
            duration_seconds = round(time.time() - start_time, 3)
            logger.error(f"Error during crawl and index for {site_id}: {e}", exc_info=True)
            db.record_crawl_finish(
                run_id=run_id,
                pages_discovered=len(crawler.visited_urls),
                pages_crawled=pages_crawled_count,
                pages_unchanged=pages_unchanged_count,
                pages_updated=pages_updated_count,
                chunks_added=total_chunks_added,
                chunks_unchanged=total_chunks_unchanged,
                chunks_removed=total_chunks_removed,
                pages_skipped_304=pages_skipped_304,
                duration_seconds=duration_seconds,
                time_saved_seconds=0.0,
                status="failed",
                error=str(e),
            )
            meta["status"] = "error"
            self.site_manager.write_metadata(site_id, meta)
            raise

    def retrieve_context(
        self,
        site_id: str,
        query: str,
        top_k: int = settings.SIMILARITY_TOP_K,
        similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
    ) -> Tuple[List[Citation], str]:
        """Retrieve top chunks for query from the site's isolated index and format prompt context."""
        db = self.site_manager.get_database(site_id)
        vector_store = self.site_manager.get_vector_store(site_id)

        if vector_store.total_vectors == 0:
            return [], ""

        # Encode user query
        query_vector = self.embedder.encode(query)[0]
        results = vector_store.search(query_vector, top_k=top_k)

        citations: List[Citation] = []
        context_blocks: List[str] = []

        for rank, (faiss_id, score) in enumerate(results, start=1):
            if score < similarity_threshold:
                continue

            chunk_row = db.get_chunk_by_faiss_id(faiss_id)
            if not chunk_row:
                continue

            page_info = f", Page {chunk_row['page_number']}" if chunk_row.get("page_number") else ""
            heading_info = f" > {chunk_row['heading']}" if chunk_row.get("heading") else ""

            citation = Citation(
                url=chunk_row["url"],
                title=chunk_row["title"],
                chunk_index=chunk_row["chunk_index"],
                page_number=chunk_row.get("page_number"),
                similarity_score=round(float(score), 4),
                snippet=chunk_row["text"][:240] + ("..." if len(chunk_row["text"]) > 240 else ""),
            )
            citations.append(citation)

            block = (
                f"[Source {rank} | URL: {chunk_row['url']}{page_info}{heading_info}]\n"
                f"{chunk_row['text']}"
            )
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)
        return citations, formatted_context

    def _build_rag_prompt(self, query: str, context: str) -> Tuple[str, str]:
        """Construct system prompt and user prompt strictly requiring citations."""
        system_prompt = (
            "You are WebRAG, an accurate AI assistant that answers questions using ONLY the provided website context.\n"
            "Rules you must strictly follow:\n"
            "1. Ground every statement strictly in the provided context.\n"
            "2. If the context does not contain enough information to answer, state clearly: "
            "'I cannot find the answer to this question in the indexed website content.'\n"
            "3. You MUST cite the source URLs for facts. Cite them in markdown format like [Source Title](URL) "
            "or [URL] directly where relevant.\n"
            "4. Do NOT make up URLs or assume facts not present in the excerpts."
        )

        user_prompt = (
            f"Indexed Website Context:\n"
            f"---------------------\n"
            f"{context}\n"
            f"---------------------\n\n"
            f"User Question: {query}\n\n"
            f"Provide a comprehensive, well-structured answer with exact citations to the source URLs:"
        )

        return system_prompt, user_prompt

    def query(
        self,
        site_id: str,
        query: str,
        top_k: int = settings.SIMILARITY_TOP_K,
        similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
        model: Optional[str] = None,
    ) -> QueryResponse:
        """Execute end-to-end RAG query with Ollama generation."""
        start_time = time.time()
        citations, context = self.retrieve_context(
            site_id=site_id,
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        if not context:
            latency_ms = (time.time() - start_time) * 1000
            return QueryResponse(
                answer="No relevant content found in this knowledge base for your query. Try rephrasing or check if the website has been crawled.",
                citations=[],
                latency_ms=round(latency_ms, 2),
                chunks_retrieved=0,
            )

        system_prompt, user_prompt = self._build_rag_prompt(query, context)
        answer = self.ollama.generate(prompt=user_prompt, system=system_prompt, model=model)
        latency_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            answer=answer.strip(),
            citations=citations,
            latency_ms=round(latency_ms, 2),
            chunks_retrieved=len(citations),
        )

    def query_stream(
        self,
        site_id: str,
        query: str,
        top_k: int = settings.SIMILARITY_TOP_K,
        similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
        model: Optional[str] = None,
    ) -> Tuple[List[Citation], Generator[str, None, None]]:
        """Streaming version returning citations and a token generator."""
        citations, context = self.retrieve_context(
            site_id=site_id,
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        if not context:
            def empty_gen():
                yield "No relevant content found in this knowledge base for your query. Try rephrasing or check if the website has been crawled."
            return [], empty_gen()

        system_prompt, user_prompt = self._build_rag_prompt(query, context)
        token_stream = self.ollama.generate_stream(prompt=user_prompt, system=system_prompt, model=model)
        return citations, token_stream
