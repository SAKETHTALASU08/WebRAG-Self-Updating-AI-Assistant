import logging
import time
from typing import AsyncGenerator, Dict, Generator, List, Optional, Tuple

from app.config import settings
from app.crawler.extractor import ExtractedDocument
from app.models.schemas import Citation, CrawlStats, QueryResponse
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
    ) -> Tuple[int, int, int]:
        """
        Incrementally index a crawled document using SHA-256 chunk hashing.
        Returns: (chunks_added, chunks_unchanged, chunks_removed)
        """
        db = self.site_manager.get_database(site_id)
        vector_store = self.site_manager.get_vector_store(site_id)

        # 1. Chunk document
        new_chunks = self.chunker.chunk_document(doc)
        if not new_chunks:
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
            vector_store.remove_vectors(removed_faiss_ids)
            chunks_removed = len(removed_faiss_ids)
            logger.info(f"Removed {chunks_removed} obsolete chunks for {doc.url}")

        # 4. Upsert parent page record first so foreign key constraints are satisfied
        import hashlib
        page_hash = hashlib.sha256(doc.raw_text.encode("utf-8")).hexdigest()
        db.upsert_page(
            url=doc.url,
            title=doc.title,
            content_type=doc.content_type,
            page_hash=page_hash,
            chunk_count=len(new_chunks),
        )

        # 5. Embed and store new/modified chunks
        chunks_added = 0
        if chunks_to_embed:
            texts = [c.text for c in chunks_to_embed]
            vectors = self.embedder.encode(texts)

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

            vector_store.add_vectors(assigned_faiss_ids, vectors)
            chunks_added = len(chunks_to_embed)
            logger.info(f"Embedded and saved {chunks_added} new chunks for {doc.url}")

        return (chunks_added, chunks_unchanged, chunks_removed)

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
