import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from app.rag.chunker import DocumentChunk
from app.rag.embedder import LocalEmbedder
from app.storage.database import SiteDatabase
from app.storage.vector_store import FaissVectorStore


@pytest.fixture
def temp_dir():
    dirpath = Path(tempfile.mkdtemp(prefix="webrag_test_vec_"))
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)


def test_local_embedder_batch_encoding():
    """Verify local sentence-transformers embedder encodes texts and chunks in configurable batches."""
    embedder = LocalEmbedder.get_instance()

    texts = [
        "Antigravity is an agentic AI coding assistant.",
        "FAISS provides fast similarity search and clustering of dense vectors.",
        "SQLite is a C-language library that implements a small, fast, self-contained SQL engine.",
        "Sentence-transformers generates dense semantic vector representations.",
        "Incremental chunking avoids re-embedding unchanged documents.",
    ]

    # Batch size 2
    embeddings = embedder.encode(texts, batch_size=2)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (5, 384)
    assert embeddings.dtype == np.float32

    # Verify L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    # Test embed_chunks with DocumentChunk objects
    chunks = [
        DocumentChunk(
            url="https://example.com/test",
            text=text,
            heading="Test Heading",
            page_no=1,
            chunk_id=f"chk_{i}",
            chunk_hash=f"hash_{i}",
        )
        for i, text in enumerate(texts)
    ]

    chunk_embeddings = embedder.embed_chunks(chunks, batch_size=2)
    assert chunk_embeddings.shape == (5, 384)
    # The embeddings for the exact same text should match
    np.testing.assert_allclose(chunk_embeddings, embeddings, atol=1e-5)

    # Empty inputs should return empty (0, 384) array
    empty_res = embedder.embed_chunks([])
    assert empty_res.shape == (0, 384)


def test_faiss_vector_store_add_remove_search(temp_dir: Path):
    """Verify IndexIDMap add, remove, and search methods."""
    index_file = temp_dir / "test_site" / "faiss.index"
    store = FaissVectorStore(index_file=index_file, dimension=4)

    assert store.total_vectors == 0

    # Create 3 orthogonal 4-D vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    vectors = np.vstack([v1, v2, v3])
    ids = [100, 200, 300]

    store.add(ids=ids, vectors=vectors)
    assert store.total_vectors == 3
    assert set(store.get_all_ids()) == {100, 200, 300}

    # Search for v1
    results = store.search(v1, top_k=3)
    assert len(results) == 3
    assert results[0][0] == 100
    assert pytest.approx(results[0][1], rel=1e-3) == 1.0
    # Orthogonal vectors should have ~0.0 dot product
    assert pytest.approx(results[1][1], abs=1e-4) == 0.0

    # Remove vector ID 200
    removed_count = store.remove([200])
    assert removed_count == 1
    assert store.total_vectors == 2
    assert set(store.get_all_ids()) == {100, 300}

    # Searching v2 should no longer return ID 200
    v2_results = store.search(v2, top_k=2)
    result_ids = [r[0] for r in v2_results]
    assert 200 not in result_ids


def test_faiss_reload_persistence(temp_dir: Path):
    """Verify that FAISS IndexIDMap vectors and custom IDs persist cleanly to disk and reload."""
    index_file = temp_dir / "persisted_site" / "faiss.index"
    dim = 8
    store = FaissVectorStore(index_file=index_file, dimension=dim)

    rng = np.random.RandomState(42)
    raw_vecs = rng.randn(4, dim).astype(np.float32)
    faiss_ids = [10, 25, 42, 99]

    store.add(ids=faiss_ids, vectors=raw_vecs)
    store.save()
    assert index_file.exists()
    assert store.total_vectors == 4

    # Remove ID 25, leaving [10, 42, 99]
    store.remove([25])
    assert store.total_vectors == 3

    # Load from a completely new instance pointing to the same index file
    reloaded_store = FaissVectorStore(index_file=index_file, dimension=dim)
    assert reloaded_store.total_vectors == 3
    assert set(reloaded_store.get_all_ids()) == {10, 42, 99}

    # Query with raw_vecs[2] (which was assigned ID 42)
    query_vec = raw_vecs[2]
    results = reloaded_store.search(query_vec, top_k=2)
    assert results[0][0] == 42
    assert pytest.approx(results[0][1], rel=1e-3) == 1.0

    # Ensure removed vector 25 is not found
    all_retrieved_ids = [r[0] for r in reloaded_store.search(query_vec, top_k=5)]
    assert 25 not in all_retrieved_ids


def test_sqlite_faiss_id_mapping_and_persistence(temp_dir: Path):
    """Verify FAISS ID to chunk_id mapping is kept consistently in SQLite across updates and deletions."""
    db_path = temp_dir / "site_kb" / "site.db"
    index_path = temp_dir / "site_kb" / "faiss.index"

    db = SiteDatabase(db_path=db_path, site_id="site_kb")
    vec_store = FaissVectorStore(index_file=index_path, dimension=4)

    # Upsert parent page
    page_url = "https://example.com/docs/api"
    db.upsert_page(url=page_url, title="API Reference", content_type="html")

    # Insert 3 chunks with content-based SHA-256 chunk IDs and integer FAISS IDs
    chunks_data = [
        ("sha_chunk_001", 1, "Overview of API"),
        ("sha_chunk_002", 2, "Authentication headers"),
        ("sha_chunk_003", 3, "Rate limiting policies"),
    ]

    for chunk_id, faiss_id, heading in chunks_data:
        db.insert_chunk(
            chunk_id=chunk_id,
            site_id="site_kb",
            url=page_url,
            heading=heading,
            text=f"Text for {heading}",
            hash=f"hash_{chunk_id}",
            embedding_id=faiss_id,
        )

    # Add corresponding vectors to FAISS
    dim4_vecs = np.eye(4, dtype=np.float32)[:3]
    vec_store.add(ids=[1, 2, 3], vectors=dim4_vecs)

    # 1. Verify mappings in SQLite
    assert db.get_chunk_id_by_faiss_id(1) == "sha_chunk_001"
    assert db.get_chunk_id_by_faiss_id(2) == "sha_chunk_002"
    assert db.get_chunk_id_by_faiss_id(3) == "sha_chunk_003"
    assert db.get_chunk_id_by_faiss_id(999) is None

    assert db.get_faiss_id_by_chunk_id("sha_chunk_001") == 1
    assert db.get_faiss_id_by_chunk_id("sha_chunk_002") == 2
    assert db.get_faiss_id_by_chunk_id("sha_chunk_003") == 3
    assert db.get_faiss_id_by_chunk_id("non_existent") is None

    all_mappings = db.get_all_faiss_mappings()
    assert all_mappings == {
        1: "sha_chunk_001",
        2: "sha_chunk_002",
        3: "sha_chunk_003",
    }

    # 2. Delete chunk 2 by chunk_id and synchronize FAISS
    deleted_faiss_ids = db.delete_chunks_by_ids(["sha_chunk_002"])
    assert deleted_faiss_ids == [2]
    vec_store.remove(deleted_faiss_ids)

    assert db.get_chunk_id_by_faiss_id(2) is None
    assert db.get_faiss_id_by_chunk_id("sha_chunk_002") is None
    assert vec_store.total_vectors == 2
    assert set(vec_store.get_all_ids()) == {1, 3}

    # 3. Reload both SQLite DB and FAISS index from disk
    reloaded_db = SiteDatabase(db_path=db_path, site_id="site_kb")
    reloaded_vec_store = FaissVectorStore(index_file=index_path, dimension=4)

    assert reloaded_db.get_stats()["total_chunks"] == 2
    assert reloaded_vec_store.total_vectors == 2
    assert reloaded_db.get_all_faiss_mappings() == {
        1: "sha_chunk_001",
        3: "sha_chunk_003",
    }
    assert set(reloaded_vec_store.get_all_ids()) == {1, 3}
