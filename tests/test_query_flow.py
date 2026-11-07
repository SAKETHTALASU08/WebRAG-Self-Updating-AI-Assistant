import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from app.crawler.extractor import ExtractedDocument, ExtractedSection
from app.models.schemas import QueryResponse, SourceItem
from app.rag.ollama_client import OllamaClient
from app.rag.pipeline import RagPipeline
from app.rag.reranker import LocalReranker
from app.storage.site_manager import SiteManager


class MockDeterministicEmbedder:
    """Mock embedder returning deterministic vectors based on text hash."""
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._device = "cpu"

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

    def embed_chunks(self, chunks, batch_size=32):
        texts = [c.text if hasattr(c, "text") else c["text"] for c in chunks]
        return self.encode(texts, batch_size=batch_size)


@pytest.fixture
def query_pipeline_env():
    temp_dir = Path(tempfile.mkdtemp(prefix="webrag_query_flow_test_"))
    site_manager = SiteManager(base_data_dir=temp_dir)
    embedder = MockDeterministicEmbedder()
    mock_ollama = MagicMock(spec=OllamaClient)
    reranker = LocalReranker()

    pipeline = RagPipeline(
        site_manager=site_manager,
        embedder=embedder,
        ollama=mock_ollama,
        reranker=reranker,
    )

    yield site_manager, pipeline, mock_ollama, temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_query_flow_retrieval_and_numbered_blocks(query_pipeline_env):
    """Verify question embedding, FAISS retrieval, and numbered context blocks prompt construction."""
    site_manager, pipeline, mock_ollama, _ = query_pipeline_env
    site = site_manager.create_or_get_site("https://example.com/docs")

    # Index 2 documents with distinct sections and page numbers
    doc1 = ExtractedDocument(
        url="https://example.com/docs/auth",
        title="Authentication Guide",
        content_type="html",
        raw_text="Authentication is handled via Bearer tokens in HTTP authorization headers.",
        sections=[
            ExtractedSection(
                text="Authentication is handled via Bearer tokens in HTTP authorization headers.",
                heading="Bearer Authentication",
                page_no=2,
            )
        ],
        links=set(),
    )
    doc2 = ExtractedDocument(
        url="https://example.com/docs/limits",
        title="Rate Limiting",
        content_type="html",
        raw_text="API rate limits allow 100 requests per minute per IP.",
        sections=[
            ExtractedSection(
                text="API rate limits allow 100 requests per minute per IP address.",
                heading="Throttle Limits",
                page_no=5,
            )
        ],
        links=set(),
    )

    pipeline.process_document_incremental(site.id, doc1)
    pipeline.process_document_incremental(site.id, doc2)

    # Mock Ollama generation response citing [1]
    mock_ollama.generate.return_value = "API tokens must be passed in the Bearer header [1]."

    # Execute query
    response: QueryResponse = pipeline.query(
        site_id=site.id,
        query="How do I authenticate with the API?",
        top_k=2,
        similarity_threshold=-1.0,
    )

    assert mock_ollama.generate.called
    call_args = mock_ollama.generate.call_args[1]
    prompt = call_args["prompt"]
    system = call_args["system"]

    # Verify numbered context blocks: [1] and [2]
    assert "[1] Title:" in prompt
    assert "URL: https://example.com/docs" in prompt
    assert "[2] Title:" in prompt

    # Verify strict system prompt rules
    assert "You are WebRAG" in system
    assert "I couldn't find this on the website" in system
    assert "[1], [2]" in system

    # Verify return structure: {answer, sources:[{url, title, snippet, page_no}]}
    assert response.answer == "API tokens must be passed in the Bearer header [1]."
    assert len(response.sources) >= 1

    first_source: SourceItem = response.sources[0]
    assert hasattr(first_source, "url")
    assert hasattr(first_source, "title")
    assert hasattr(first_source, "snippet")
    assert hasattr(first_source, "page_no")
    assert first_source.url in ["https://example.com/docs/auth", "https://example.com/docs/limits"]
    assert first_source.page_no in [2, 5]

    # Verify dict representation matches exact requirements
    res_dict = response.model_dump()
    assert "answer" in res_dict
    assert "sources" in res_dict
    assert isinstance(res_dict["sources"], list)
    for s in res_dict["sources"]:
        assert "url" in s
        assert "title" in s
        assert "snippet" in s
        assert "page_no" in s


def test_query_flow_unsupported_question_returns_fallback(query_pipeline_env):
    """Verify that when no relevant content exists, the system returns 'I couldn't find this on the website'."""
    site_manager, pipeline, mock_ollama, _ = query_pipeline_env
    site = site_manager.create_or_get_site("https://empty-kb.org")

    # Empty knowledge base query
    response = pipeline.query(site.id, "What is the capital of Mars?")

    assert response.answer == "I couldn't find this on the website"
    assert response.sources == []
    # Ollama should NOT be invoked if there is zero context
    assert not mock_ollama.generate.called


def test_query_flow_optional_reranker(query_pipeline_env):
    """Verify that optional reranking reorders chunks using lexical/BM25 keyword overlap."""
    site_manager, pipeline, _, _ = query_pipeline_env
    site = site_manager.create_or_get_site("https://example.com/kb")

    # Insert two documents
    doc_generic = ExtractedDocument(
        url="https://example.com/generic",
        title="Generic Guide",
        content_type="html",
        raw_text="This is a general guide about our cloud service and background concepts.",
        sections=[
            ExtractedSection(
                text="This is a general guide about our cloud service and background concepts.",
                heading="Overview",
                page_no=1,
            )
        ],
        links=set(),
    )
    doc_specific = ExtractedDocument(
        url="https://example.com/payments",
        title="Stripe Webhook Integration",
        content_type="html",
        raw_text="Configuring Stripe webhook signing secrets and handling invoice.payment_succeeded events.",
        sections=[
            ExtractedSection(
                text="Configuring Stripe webhook signing secrets and handling invoice.payment_succeeded events.",
                heading="Stripe Webhooks",
                page_no=3,
            )
        ],
        links=set(),
    )

    pipeline.process_document_incremental(site.id, doc_generic)
    pipeline.process_document_incremental(site.id, doc_specific)

    # Query with exact keywords "Stripe webhook invoice"
    sources_with_rerank, _ = pipeline.retrieve_context(
        site.id,
        "Stripe webhook invoice",
        top_k=2,
        similarity_threshold=-1.0,
        rerank=True,
    )

    # Top result should be the Stripe payments page due to BM25 lexical boost
    assert len(sources_with_rerank) == 2
    assert sources_with_rerank[0].url == "https://example.com/payments"
    assert "Stripe" in sources_with_rerank[0].title


def test_query_stream_flow(query_pipeline_env):
    """Verify streaming response yields sources metadata and tokens incrementally."""
    site_manager, pipeline, mock_ollama, _ = query_pipeline_env
    site = site_manager.create_or_get_site("https://streaming-test.org")

    doc = ExtractedDocument(
        url="https://streaming-test.org/intro",
        title="Introduction",
        content_type="html",
        raw_text="WebRAG operates completely locally without remote API calls.",
        sections=[
            ExtractedSection(
                text="WebRAG operates completely locally without remote API calls.",
                heading="Local First",
                page_no=1,
            )
        ],
        links=set(),
    )
    pipeline.process_document_incremental(site.id, doc)

    # Setup streaming mock tokens
    def mock_stream(**kwargs):
        tokens = ["WebRAG ", "is ", "100% ", "local ", "[1]."]
        for t in tokens:
            yield t

    mock_ollama.generate_stream.side_effect = mock_stream

    sources, token_gen = pipeline.query_stream(site.id, "Does WebRAG use external APIs?", similarity_threshold=-1.0)

    assert len(sources) == 1
    assert sources[0].url == "https://streaming-test.org/intro"
    assert sources[0].page_no == 1

    received_tokens = list(token_gen)
    assert received_tokens == ["WebRAG ", "is ", "100% ", "local ", "[1]."]
    assembled_answer = "".join(received_tokens)
    assert assembled_answer == "WebRAG is 100% local [1]."
