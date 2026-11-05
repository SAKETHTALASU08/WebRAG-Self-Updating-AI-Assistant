import hashlib
import sqlite3
import tempfile
from pathlib import Path
from app.crawler.extractor import ExtractedDocument, ExtractedSection, StructuredDocument
from app.rag.chunker import DocumentChunk, HeadingAwareChunker, IncrementalChunker
from app.storage.database import SiteDatabase


def test_chunk_hash_reproducibility():
    """Verify SHA-256 calculation is deterministic and normalized."""
    text1 = "This is a test paragraph for WebRAG chunk hashing."
    text2 = "This is a test paragraph for WebRAG chunk hashing."

    hash1 = DocumentChunk.compute_hash(text1)
    hash2 = DocumentChunk.compute_hash(text2)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 is 64 hex characters

    # Whitespace normalization check
    text_with_spaces = "  This   is a test  paragraph for WebRAG chunk hashing.  \n"
    assert DocumentChunk.compute_hash(text_with_spaces) == hash1

    # Modification alters hash
    modified_text = "This is a modified paragraph for WebRAG chunk hashing."
    assert DocumentChunk.compute_hash(modified_text) != hash1


def test_unchanged_text_yields_identical_content_based_ids():
    """
    Prove that unchanged text yields identical chunk IDs, and inserting
    a paragraph in the middle of a document does NOT invalidate unrelated chunks.
    """
    chunker = HeadingAwareChunker(target_tokens=500, overlap_tokens=50)

    # Document Version 1: 3 sections
    sec1 = ExtractedSection(text="Section 1 discusses introduction and system overview.", heading="Introduction")
    sec2 = ExtractedSection(text="Section 2 discusses database persistence and indexing.", heading="Persistence")
    sec3 = ExtractedSection(text="Section 3 discusses retrieval performance and metrics.", heading="Performance")

    doc_v1 = StructuredDocument(
        url="https://example.com/guide",
        title="Guide V1",
        section_headings=["Introduction", "Persistence", "Performance"],
        text="...",
        sections=[sec1, sec2, sec3],
    )

    chunks_v1 = chunker.chunk_document(doc_v1)
    assert len(chunks_v1) == 3

    id_sec1_v1 = chunks_v1[0].chunk_id
    id_sec2_v1 = chunks_v1[1].chunk_id
    id_sec3_v1 = chunks_v1[2].chunk_id

    # 1. Re-chunking identical text yields 100% IDENTICAL chunk IDs
    chunks_v1_again = chunker.chunk_document(doc_v1)
    assert chunks_v1_again[0].chunk_id == id_sec1_v1
    assert chunks_v1_again[1].chunk_id == id_sec2_v1
    assert chunks_v1_again[2].chunk_id == id_sec3_v1

    # 2. Document Version 2: Insert a new paragraph between Section 1 and Section 2
    sec_new = ExtractedSection(
        text="Inserted section discussing newly added caching mechanism.",
        heading="Caching Layer",
    )
    doc_v2 = StructuredDocument(
        url="https://example.com/guide",
        title="Guide V2",
        section_headings=["Introduction", "Caching Layer", "Persistence", "Performance"],
        text="...",
        sections=[sec1, sec_new, sec2, sec3],
    )

    chunks_v2 = chunker.chunk_document(doc_v2)
    assert len(chunks_v2) == 4

    id_sec1_v2 = chunks_v2[0].chunk_id
    id_sec_new_v2 = chunks_v2[1].chunk_id
    id_sec2_v2 = chunks_v2[2].chunk_id
    id_sec3_v2 = chunks_v2[3].chunk_id

    # PROOF:
    # - Section 1 retained exact identical ID
    assert id_sec1_v2 == id_sec1_v1
    # - The new section received its own unique content ID
    assert id_sec_new_v2 not in (id_sec1_v1, id_sec2_v1, id_sec3_v1)
    # - Section 2 and Section 3 retained their EXACT IDENTICAL IDs despite being shifted in position!
    assert id_sec2_v2 == id_sec2_v1
    assert id_sec3_v2 == id_sec3_v1


def test_heading_aware_chunking_target_and_overlap():
    """Verify heading-aware chunking target ~500 tokens and ~50 token overlap."""
    target_tokens = 50
    overlap_tokens = 10
    chunker = HeadingAwareChunker(target_tokens=target_tokens, overlap_tokens=overlap_tokens)

    # Create a long section with 120 words
    words = [f"token{i}" for i in range(120)]
    long_section_text = " ".join(words)
    section = ExtractedSection(text=long_section_text, heading="Detailed Architecture")

    doc = StructuredDocument(
        url="https://example.com/arch",
        title="Architecture",
        section_headings=["Detailed Architecture"],
        text=long_section_text,
        sections=[section],
    )

    chunks = chunker.chunk_document(doc)
    # 120 words with window 50 and step (50 - 10 = 40):
    # Chunk 0: words 0..49
    # Chunk 1: words 40..89
    # Chunk 2: words 80..119
    assert len(chunks) == 3

    # Every chunk must preserve the heading
    for c in chunks:
        assert c.heading == "Detailed Architecture"
        assert len(c.chunk_id) == 64

    # Verify ~10 token overlap between chunk 0 and chunk 1
    words_c0 = set(chunks[0].text.split())
    words_c1 = set(chunks[1].text.split())
    overlap = words_c0.intersection(words_c1)
    assert len(overlap) == overlap_tokens


def test_chunker_preserves_page_number():
    """Verify chunker preserves page numbers for PDF citations."""
    chunker = HeadingAwareChunker(target_tokens=100, overlap_tokens=20)

    sec1 = ExtractedSection(text="Content from page 1", heading="Page 1", page_number=1)
    sec2 = ExtractedSection(text="Content from page 2", heading="Page 2", page_number=2)

    doc = ExtractedDocument(
        url="https://example.com/file.pdf",
        title="Sample PDF",
        content_type="pdf",
        raw_text="...",
        sections=[sec1, sec2],
        links=set(),
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].page_no == 1
    assert chunks[0].page_number == 1
    assert chunks[1].page_no == 2
    assert chunks[1].page_number == 2


def test_sqlite_schema_chunks_and_pages():
    """
    Verify SQLite schema matches:
    - chunks(chunk_id, site_id, url, page_no, heading, text, hash, embedding_id, status, updated_at)
    - pages(url, etag, last_crawled)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_site.db"
        db = SiteDatabase(db_path, site_id="test_site_123")

        # 1. Verify table columns via PRAGMA table_info
        with db._get_connection() as conn:
            cursor = conn.cursor()

            # Verify pages table columns
            cursor.execute("PRAGMA table_info(pages);")
            pages_columns = {row["name"]: row["type"] for row in cursor.fetchall()}
            assert "url" in pages_columns
            assert "etag" in pages_columns
            assert "last_crawled" in pages_columns

            # Verify chunks table columns
            cursor.execute("PRAGMA table_info(chunks);")
            chunks_columns = {row["name"]: row["type"] for row in cursor.fetchall()}
            expected_chunk_cols = [
                "chunk_id",
                "site_id",
                "url",
                "page_no",
                "heading",
                "text",
                "hash",
                "embedding_id",
                "status",
                "updated_at",
            ]
            for col in expected_chunk_cols:
                assert col in chunks_columns, f"Column '{col}' missing in chunks table"

        # 2. Test inserting into pages(url, etag, last_crawled)
        db.upsert_page(
            url="https://example.com/spec",
            etag='"etag-xyz-987"',
            last_crawled="2026-11-05T07:00:00Z",
            title="Specification",
        )
        page_rec = db.get_page("https://example.com/spec")
        assert page_rec is not None
        assert page_rec["url"] == "https://example.com/spec"
        assert page_rec["etag"] == '"etag-xyz-987"'
        assert page_rec["last_crawled"] == "2026-11-05T07:00:00Z"

        # 3. Test inserting into chunks(chunk_id, site_id, url, page_no, heading, text, hash, embedding_id, status, updated_at)
        chunk_text = "Heading-aware chunk text with SHA-256 content-based ID."
        chunk_hash = DocumentChunk.compute_hash(chunk_text)
        db.insert_chunk(
            chunk_id=chunk_hash,
            site_id="test_site_123",
            url="https://example.com/spec",
            page_no=1,
            heading="Section Header",
            text=chunk_text,
            hash=chunk_hash,
            embedding_id=42,
            status="active",
        )

        chunk_rec = db.get_chunk_by_embedding_id(42)
        assert chunk_rec is not None
        assert chunk_rec["chunk_id"] == chunk_hash
        assert chunk_rec["site_id"] == "test_site_123"
        assert chunk_rec["url"] == "https://example.com/spec"
        assert chunk_rec["page_no"] == 1
        assert chunk_rec["heading"] == "Section Header"
        assert chunk_rec["text"] == chunk_text
        assert chunk_rec["hash"] == chunk_hash
        assert chunk_rec["embedding_id"] == 42
        assert chunk_rec["status"] == "active"
        assert chunk_rec["updated_at"] is not None
