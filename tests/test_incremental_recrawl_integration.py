import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock
import numpy as np
import pytest
import requests

from app.crawler.crawler import CrawlItem, WebCrawler
from app.crawler.extractor import ContentExtractor, ExtractedDocument
from app.models.schemas import CrawlSummaryReport
from app.rag.pipeline import RagPipeline
from app.storage.database import SiteDatabase
from app.storage.site_manager import SiteManager
from app.storage.vector_store import FaissVectorStore


class MockHttpResponse:
    """Mock HTTP response supporting status_code, content, and headers."""

    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = headers or {}


class SpyEmbedder:
    """Deterministic local embedder that tracks every batch and text passed to it."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._device = "cpu"
        self.embedded_batches: List[List[str]] = []
        self.total_chunks_embedded = 0

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        self.embedded_batches.append(list(texts))
        self.total_chunks_embedded += len(texts)

        vectors = []
        for t in texts:
            import hashlib
            seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.dim).astype(np.float32)
            v /= np.linalg.norm(v)
            vectors.append(v)
        return np.vstack(vectors) if vectors else np.empty((0, self.dim), dtype=np.float32)

    def embed_chunks(self, chunks, batch_size=32):
        texts = [c.text if hasattr(c, "text") else c["text"] for c in chunks]
        return self.encode(texts, batch_size=batch_size)

    def reset_tracker(self):
        self.embedded_batches.clear()
        self.total_chunks_embedded = 0


@pytest.fixture
def isolated_site_env():
    temp_dir = Path(tempfile.mkdtemp(prefix="webrag_recrawl_test_"))
    site_manager = SiteManager(base_data_dir=temp_dir)
    spy_embedder = SpyEmbedder(dim=384)
    pipeline = RagPipeline(site_manager=site_manager, embedder=spy_embedder)

    yield site_manager, pipeline, spy_embedder, temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_incremental_recrawl_edits_one_paragraph_asserts_only_affected_reembedded(isolated_site_env):
    """
    Integration test:
    1. Initial crawl of mock website with multiple distinct sections.
    2. Assert all chunks are embedded.
    3. Re-crawl without changes -> assert conditional requests (ETag/Last-Modified) trigger 304,
       0 chunks re-embedded, time saved > 0.
    4. Edit exactly ONE paragraph on the mock site -> re-crawl.
    5. Assert:
       - Exactly 1 new chunk is embedded (only the affected section).
       - Exactly 1 obsolete chunk is removed from FAISS & SQLite.
       - Unchanged sections are completely skipped from embedding and keep their original IDs.
       - Summary report accurately logs counts and time saved.
    """
    site_manager, pipeline, embedder, temp_dir = isolated_site_env
    root_url = "https://mock-knowledge-base.org"
    docs_url = "https://mock-knowledge-base.org/handbook"

    site = site_manager.create_or_get_site(root_url=root_url, name="Mock Handbook")
    site_id = site.id

    # ---------------- Phase 1: Initial Mock Website Setup ---------------- #
    html_root_v1 = f"""
    <!DOCTYPE html>
    <html>
      <head><title>Mock Knowledge Base</title></head>
      <body>
        <h1>Welcome</h1>
        <p>This is the portal.</p>
        <a href="{docs_url}">Engineering Handbook</a>
      </body>
    </html>
    """

    # Section 1, 2, and 3 on the Handbook page
    sec1_text = "Overview: WebRAG generates self-updating local AI assistants from websites."
    sec2_text = "Architecture: FAISS IndexIDMap and SQLite provide isolated per-site vector storage."
    sec3_text = "Security: All computations run strictly locally on CPU or Apple Silicon without external APIs."

    html_docs_v1 = f"""
    <!DOCTYPE html>
    <html>
      <head><title>Engineering Handbook</title></head>
      <body>
        <main>
          <h2>Overview Section</h2>
          <p>{sec1_text}</p>
          <h2>Architecture Section</h2>
          <p>{sec2_text}</p>
          <h2>Security Section</h2>
          <p>{sec3_text}</p>
        </main>
      </body>
    </html>
    """

    current_site_state = {
        "root": {
            "html": html_root_v1,
            "etag": '"root-v1-etag"',
            "last_modified": "Mon, 10 Nov 2025 10:00:00 GMT",
        },
        "docs": {
            "html": html_docs_v1,
            "etag": '"docs-v1-etag"',
            "last_modified": "Mon, 10 Nov 2025 10:00:00 GMT",
        },
    }

    last_received_headers: Dict[str, Dict[str, str]] = {}

    def mock_get_handler(url, headers=None, **kwargs):
        last_received_headers[url] = dict(headers or {})

        if url.rstrip("/").endswith("robots.txt"):
            return MockHttpResponse(status_code=404)

        if url == root_url or url == f"{root_url}/":
            item = current_site_state["root"]
            if headers and headers.get("If-None-Match") == item["etag"]:
                return MockHttpResponse(status_code=304, headers={"ETag": item["etag"]})
            return MockHttpResponse(
                status_code=200,
                text=item["html"],
                headers={"Content-Type": "text/html", "ETag": item["etag"], "Last-Modified": item["last_modified"]},
            )

        if url == docs_url:
            item = current_site_state["docs"]
            if headers and headers.get("If-None-Match") == item["etag"]:
                return MockHttpResponse(status_code=304, headers={"ETag": item["etag"]})
            return MockHttpResponse(
                status_code=200,
                text=item["html"],
                headers={"Content-Type": "text/html", "ETag": item["etag"], "Last-Modified": item["last_modified"]},
            )

        return MockHttpResponse(status_code=404)

    mock_session = MagicMock(spec=requests.Session)
    mock_session.headers = {}
    mock_session.get.side_effect = mock_get_handler

    # 1. Execute initial crawl & indexing
    report_initial: CrawlSummaryReport = pipeline.crawl_and_index_site(
        site_id=site_id,
        max_depth=2,
        max_pages=10,
        delay_seconds=0.0,
        session=mock_session,
    )

    db: SiteDatabase = site_manager.get_database(site_id)
    vec_store: FaissVectorStore = site_manager.get_vector_store(site_id)

    assert report_initial.pages_crawled >= 2
    assert report_initial.pages_skipped_304 == 0
    assert report_initial.chunks_added > 0
    assert report_initial.chunks_unchanged == 0
    assert report_initial.chunks_removed == 0
    initial_total_chunks = report_initial.chunks_added
    assert vec_store.total_vectors == initial_total_chunks

    # Record initial chunk IDs and FAISS mappings
    initial_mappings = db.get_all_faiss_mappings()
    assert len(initial_mappings) == initial_total_chunks

    # Verify handbook chunks
    existing_docs_chunks = db.get_existing_chunk_hashes_for_url(docs_url)
    assert len(existing_docs_chunks) == 3

    # Find the initial chunk IDs for each section
    chunk_sec1 = next(c for c in existing_docs_chunks.values() if "Overview" in c["text"])
    chunk_sec2 = next(c for c in existing_docs_chunks.values() if "Architecture" in c["text"])
    chunk_sec3 = next(c for c in existing_docs_chunks.values() if "Security" in c["text"])

    initial_sec1_id = chunk_sec1["chunk_id"]
    initial_sec1_faiss = chunk_sec1["embedding_id"]

    initial_sec2_id = chunk_sec2["chunk_id"]
    initial_sec2_faiss = chunk_sec2["embedding_id"]

    initial_sec3_id = chunk_sec3["chunk_id"]
    initial_sec3_faiss = chunk_sec3["embedding_id"]

    # Reset embedder tracker to measure re-crawl behavior
    embedder.reset_tracker()

    # ---------------- Phase 2: Unchanged Re-Crawl (HTTP 304 via ETag) ---------------- #
    report_unchanged: CrawlSummaryReport = pipeline.crawl_and_index_site(
        site_id=site_id,
        max_depth=2,
        max_pages=10,
        delay_seconds=0.0,
        session=mock_session,
    )

    # Verify conditional headers were sent
    assert last_received_headers[root_url].get("If-None-Match") == '"root-v1-etag"'
    assert last_received_headers[docs_url].get("If-None-Match") == '"docs-v1-etag"'

    # Assert 304 skipping and 0 re-embeddings
    assert report_unchanged.pages_skipped_304 >= 2
    assert report_unchanged.chunks_added == 0
    assert report_unchanged.chunks_unchanged == initial_total_chunks
    assert report_unchanged.chunks_removed == 0
    assert report_unchanged.time_saved_seconds > 0.0
    assert embedder.total_chunks_embedded == 0  # 0 calls to sentence-transformers!

    # ---------------- Phase 3: Edit ONE Paragraph on Handbook Page ---------------- #
    # Section 1 (Overview) and Section 3 (Security) remain completely identical.
    # ONLY Section 2 (Architecture) is modified!
    edited_sec2_text = "Architecture: FAISS IndexIDMap and SQLite provide isolated per-site storage with zero external dependencies."
    html_docs_v2 = f"""
    <!DOCTYPE html>
    <html>
      <head><title>Engineering Handbook</title></head>
      <body>
        <main>
          <h2>Overview Section</h2>
          <p>{sec1_text}</p>
          <h2>Architecture Section</h2>
          <p>{edited_sec2_text}</p>
          <h2>Security Section</h2>
          <p>{sec3_text}</p>
        </main>
      </body>
    </html>
    """

    # Update mock server state: root remains 304, docs page has updated ETag and new HTML
    current_site_state["docs"]["html"] = html_docs_v2
    current_site_state["docs"]["etag"] = '"docs-v2-etag"'
    current_site_state["docs"]["last_modified"] = "Tue, 11 Nov 2025 14:00:00 GMT"

    embedder.reset_tracker()

    # Execute re-crawl after editing one paragraph
    report_edited: CrawlSummaryReport = pipeline.crawl_and_index_site(
        site_id=site_id,
        max_depth=2,
        max_pages=10,
        delay_seconds=0.0,
        session=mock_session,
    )

    # ---------------- Verification & Assertions ---------------- #
    # 1. Root page was skipped via 304 Not Modified
    assert report_edited.pages_skipped_304 >= 1

    # 2. Assert EXACTLY ONE chunk was added/embedded
    assert report_edited.chunks_added == 1
    # 3. Assert EXACTLY ONE chunk was removed (the old version of Section 2)
    assert report_edited.chunks_removed == 1
    # 4. Assert the other chunks were unchanged and skipped
    assert report_edited.chunks_unchanged >= 2
    # 5. Assert time saved was calculated and logged
    assert report_edited.time_saved_seconds > 0.0

    # 6. Verify embedder was invoked for ONLY the ONE affected paragraph!
    assert embedder.total_chunks_embedded == 1
    embedded_text = embedder.embedded_batches[0][0]
    assert "zero external dependencies" in embedded_text
    assert "Architecture Section" in embedded_text or "Architecture" in embedded_text

    # 7. Check SQLite records: Section 1 and Section 3 retained their exact chunk_ids and FAISS IDs
    updated_docs_chunks = db.get_existing_chunk_hashes_for_url(docs_url)
    assert len(updated_docs_chunks) == 3

    chunk_sec1_after = next(c for c in updated_docs_chunks.values() if "Overview" in c["text"])
    chunk_sec2_after = next(c for c in updated_docs_chunks.values() if "zero external dependencies" in c["text"])
    chunk_sec3_after = next(c for c in updated_docs_chunks.values() if "Security" in c["text"])

    # Unchanged chunks have identical content IDs and FAISS IDs!
    assert chunk_sec1_after["chunk_id"] == initial_sec1_id
    assert chunk_sec1_after["embedding_id"] == initial_sec1_faiss

    assert chunk_sec3_after["chunk_id"] == initial_sec3_id
    assert chunk_sec3_after["embedding_id"] == initial_sec3_faiss

    # The modified section has a brand new chunk_id and new FAISS ID
    assert chunk_sec2_after["chunk_id"] != initial_sec2_id
    assert chunk_sec2_after["embedding_id"] != initial_sec2_faiss

    # The old section 2 chunk ID is completely removed from SQLite
    assert db.get_chunk_by_chunk_id(initial_sec2_id) is None
    assert db.get_chunk_id_by_faiss_id(initial_sec2_faiss) is None

    # 8. Check FAISS vector store
    assert vec_store.total_vectors == initial_total_chunks
    all_faiss_ids = set(vec_store.get_all_ids())
    assert initial_sec1_faiss in all_faiss_ids
    assert initial_sec3_faiss in all_faiss_ids
    assert initial_sec2_faiss not in all_faiss_ids
    assert chunk_sec2_after["embedding_id"] in all_faiss_ids
