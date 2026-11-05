import numpy as np
import pytest
from unittest.mock import MagicMock
from app.crawler.extractor import ExtractedDocument, ExtractedSection
from app.rag.chunker import IncrementalChunker
from app.rag.embedder import LocalEmbedder
from app.rag.ollama_client import OllamaClient
from app.rag.pipeline import RagPipeline
from app.storage.site_manager import SiteManager


class DummyEmbedder:
    """Mock embedder that returns deterministic normalized vectors based on text hash."""
    def __init__(self, dim=384):
        self._device = "cpu"
        self.dim = dim

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for t in texts:
            import hashlib
            seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.dim).astype(np.float32)
            v /= np.linalg.norm(v)
            vectors.append(v)
        return np.vstack(vectors) if vectors else np.empty((0, self.dim), dtype=np.float32)


def test_incremental_indexing_workflow(mock_site_manager: SiteManager):
    site = mock_site_manager.create_or_get_site("https://acme.org/handbook")
    dummy_embedder = DummyEmbedder()
    pipeline = RagPipeline(site_manager=mock_site_manager, embedder=dummy_embedder)

    # 1. First Crawl Version (2 sections)
    doc_v1 = ExtractedDocument(
        url="https://acme.org/handbook",
        title="Acme Handbook",
        content_type="html",
        raw_text="Section 1\nAcme was founded in 2020.\n\nSection 2\nRemote work is our standard.",
        sections=[
            ExtractedSection(text="Acme was founded in 2020.", heading="Section 1"),
            ExtractedSection(text="Remote work is our standard policy.", heading="Section 2"),
        ],
        links=set(),
    )

    added, unchanged, removed = pipeline.process_document_incremental(site.id, doc_v1)
    assert added == 2
    assert unchanged == 0
    assert removed == 0

    db = mock_site_manager.get_database(site.id)
    assert db.get_stats()["total_chunks"] == 2

    # 2. Re-crawl unchanged page -> Everything should be recognized as UNCHANGED (0 re-embedded)
    added_re, unchanged_re, removed_re = pipeline.process_document_incremental(site.id, doc_v1)
    assert added_re == 0
    assert unchanged_re == 2
    assert removed_re == 0
    assert db.get_stats()["total_chunks"] == 2

    # 3. Updated Page (Section 1 unchanged, Section 2 modified, Section 3 added)
    doc_v2 = ExtractedDocument(
        url="https://acme.org/handbook",
        title="Acme Handbook",
        content_type="html",
        raw_text="Section 1\nAcme was founded in 2020.\n\nSection 2\nHybrid work is our new standard.\n\nSection 3\nHealthcare coverage details.",
        sections=[
            ExtractedSection(text="Acme was founded in 2020.", heading="Section 1"),  # Unchanged
            ExtractedSection(text="Hybrid work is our new standard policy.", heading="Section 2"),  # Modified
            ExtractedSection(text="Healthcare coverage details are available on the portal.", heading="Section 3"),  # New
        ],
        links=set(),
    )

    added_up, unchanged_up, removed_up = pipeline.process_document_incremental(site.id, doc_v2)
    # Section 1 unchanged (1), Section 2 modified (old removed = 1, new added = 1), Section 3 added (1) -> added=2, unchanged=1, removed=1
    assert added_up == 2
    assert unchanged_up == 1
    assert removed_up == 1

    assert db.get_stats()["total_chunks"] == 3


def test_rag_retrieval_and_citations(mock_site_manager: SiteManager):
    site = mock_site_manager.create_or_get_site("https://acme.org/faq")
    dummy_embedder = DummyEmbedder()
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.generate.return_value = "Acme was founded in 2020 as documented in [https://acme.org/faq]."

    pipeline = RagPipeline(
        site_manager=mock_site_manager,
        embedder=dummy_embedder,
        ollama=mock_ollama,
    )

    # Index document
    doc = ExtractedDocument(
        url="https://acme.org/faq",
        title="Acme FAQ",
        content_type="html",
        raw_text="Frequently Asked Questions. Acme was founded in 2020 by Jane Doe.",
        sections=[
            ExtractedSection(text="Acme was founded in 2020 by Jane Doe in San Francisco.", heading="Founding"),
        ],
        links=set(),
    )
    pipeline.process_document_incremental(site.id, doc)

    # Retrieve context
    citations, context = pipeline.retrieve_context(site.id, "When was Acme founded?", similarity_threshold=-1.0)
    assert len(citations) >= 1
    assert citations[0].url == "https://acme.org/faq"
    assert citations[0].title == "Acme FAQ"
    assert "https://acme.org/faq" in context

    # Test complete query
    response = pipeline.query(site.id, "When was Acme founded?", similarity_threshold=-1.0)
    assert len(response.citations) >= 1
    assert response.citations[0].url == "https://acme.org/faq"
    assert "https://acme.org/faq" in response.answer
    assert mock_ollama.generate.called
