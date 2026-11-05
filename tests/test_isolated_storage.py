import numpy as np
from app.storage.site_manager import SiteManager


def test_site_manager_isolation(mock_site_manager: SiteManager):
    # 1. Create two separate sites
    site_a = mock_site_manager.create_or_get_site("https://site-alpha.com/docs", name="Alpha Docs")
    site_b = mock_site_manager.create_or_get_site("https://site-beta.com/api", name="Beta API")

    assert site_a.id != site_b.id
    assert mock_site_manager.get_site_dir(site_a.id) != mock_site_manager.get_site_dir(site_b.id)

    # 2. Get databases
    db_a = mock_site_manager.get_database(site_a.id)
    db_b = mock_site_manager.get_database(site_b.id)

    # 3. Insert page and chunk into Site A only
    db_a.upsert_page("https://site-alpha.com/docs", "Alpha Docs", "html", "hash_alpha", 1)
    db_a.insert_chunk(
        chunk_id="chunk_a_1",
        url="https://site-alpha.com/docs",
        title="Alpha Docs",
        chunk_index=0,
        page_number=None,
        heading="Intro",
        chunk_hash="sha_alpha_1",
        text="Alpha knowledge base exclusively.",
        faiss_id=1,
    )

    # 4. Verify Site B is completely unaffected (isolated)
    assert db_a.get_stats()["total_chunks"] == 1
    assert db_b.get_stats()["total_chunks"] == 0
    assert db_b.get_page("https://site-alpha.com/docs") is None

    # 5. Verify isolated vector stores
    vec_a = mock_site_manager.get_vector_store(site_a.id)
    vec_b = mock_site_manager.get_vector_store(site_b.id)

    # Add random 384-d vector to vec_a
    rng = np.random.RandomState(42)
    fake_vector = rng.randn(1, 384).astype(np.float32)
    vec_a.add_vectors([1], fake_vector)

    assert vec_a.total_vectors == 1
    assert vec_b.total_vectors == 0

    # 6. Delete Site A, verify Site B remains intact
    deleted = mock_site_manager.delete_site(site_a.id)
    assert deleted is True
    assert mock_site_manager.get_site(site_a.id) is None
    assert mock_site_manager.get_site(site_b.id) is not None


def test_vector_store_search_and_removal(mock_site_manager: SiteManager):
    site = mock_site_manager.create_or_get_site("https://example.com")
    vec_store = mock_site_manager.get_vector_store(site.id)

    dim = 384
    # Create 3 distinct vectors
    v1 = np.zeros((1, dim), dtype=np.float32)
    v1[0, 0] = 1.0  # Points in dimension 0

    v2 = np.zeros((1, dim), dtype=np.float32)
    v2[0, 1] = 1.0  # Points in dimension 1

    v3 = np.zeros((1, dim), dtype=np.float32)
    v3[0, 0] = 0.95
    v3[0, 1] = 0.05  # Very close to v1

    vec_store.add_vectors([101, 102, 103], np.vstack([v1, v2, v3]))
    assert vec_store.total_vectors == 3

    # Query with v1, top match should be ID 101, followed by 103
    results = vec_store.search(v1, top_k=2)
    assert len(results) == 2
    top_id, top_score = results[0]
    assert top_id == 101
    assert top_score > 0.99  # Normalized dot product = 1.0

    second_id, second_score = results[1]
    assert second_id == 103

    # Remove vector 101
    removed = vec_store.remove_vectors([101])
    assert removed == 1
    assert vec_store.total_vectors == 2

    # Query again, top match should now be 103
    results_after = vec_store.search(v1, top_k=1)
    assert results_after[0][0] == 103
