import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SiteDatabase:
    """Manages isolated SQLite database for a single website knowledge base."""

    def __init__(self, db_path: Path, site_id: Optional[str] = None):
        self.db_path = db_path
        self.site_id = site_id or db_path.parent.name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pages (
                    url TEXT PRIMARY KEY,
                    etag TEXT,
                    last_crawled TIMESTAMP NOT NULL,
                    title TEXT DEFAULT '',
                    content_type TEXT DEFAULT 'html',
                    page_hash TEXT DEFAULT '',
                    chunk_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    site_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    page_no INTEGER,
                    heading TEXT,
                    text TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    embedding_id INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    updated_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (url) REFERENCES pages (url) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_url ON chunks(url);
                CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash);
                CREATE INDEX IF NOT EXISTS idx_chunks_site ON chunks(site_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks(embedding_id);

                CREATE TABLE IF NOT EXISTS crawl_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TIMESTAMP NOT NULL,
                    finished_at TIMESTAMP,
                    pages_discovered INTEGER DEFAULT 0,
                    pages_crawled INTEGER DEFAULT 0,
                    pages_unchanged INTEGER DEFAULT 0,
                    pages_updated INTEGER DEFAULT 0,
                    chunks_added INTEGER DEFAULT 0,
                    chunks_unchanged INTEGER DEFAULT 0,
                    chunks_removed INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    error TEXT
                );
            """)

    def get_page(self, url: str) -> Optional[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM pages WHERE url = ?", (url,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            # Compatibility alias
            d["crawled_at"] = d.get("last_crawled")
            return d

    def get_existing_chunk_hashes_for_url(self, url: str) -> Dict[str, dict]:
        """Return dict of {hash: row_dict} currently stored for this URL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    chunk_id, chunk_id AS id,
                    site_id,
                    url,
                    page_no, page_no AS page_number,
                    heading,
                    text,
                    hash, hash AS chunk_hash,
                    embedding_id, embedding_id AS faiss_id,
                    status,
                    updated_at, updated_at AS created_at
                FROM chunks 
                WHERE url = ?
                """,
                (url,),
            )
            return {row["hash"]: dict(row) for row in cursor.fetchall()}

    def get_all_chunks(self) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    chunk_id, chunk_id AS id,
                    site_id,
                    url,
                    page_no, page_no AS page_number,
                    heading,
                    text,
                    hash, hash AS chunk_hash,
                    embedding_id, embedding_id AS faiss_id,
                    status,
                    updated_at, updated_at AS created_at,
                    0 AS chunk_index,
                    '' AS title
                FROM chunks 
                ORDER BY url, embedding_id ASC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_chunk_by_embedding_id(self, embedding_id: int) -> Optional[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    chunk_id, chunk_id AS id,
                    site_id,
                    url,
                    page_no, page_no AS page_number,
                    heading,
                    text,
                    hash, hash AS chunk_hash,
                    embedding_id, embedding_id AS faiss_id,
                    status,
                    updated_at, updated_at AS created_at,
                    0 AS chunk_index,
                    '' AS title
                FROM chunks 
                WHERE embedding_id = ?
                """,
                (embedding_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            # Find page title if available
            cursor.execute("SELECT title FROM pages WHERE url = ?", (d["url"],))
            p_row = cursor.fetchone()
            if p_row and p_row["title"]:
                d["title"] = p_row["title"]
            return d

    def get_chunk_by_faiss_id(self, faiss_id: int) -> Optional[dict]:
        """Backward compatibility alias for get_chunk_by_embedding_id."""
        return self.get_chunk_by_embedding_id(faiss_id)

    def get_chunk_by_chunk_id(self, chunk_id: str) -> Optional[dict]:
        """Retrieve full chunk record by content chunk_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    chunk_id, chunk_id AS id,
                    site_id,
                    url,
                    page_no, page_no AS page_number,
                    heading,
                    text,
                    hash, hash AS chunk_hash,
                    embedding_id, embedding_id AS faiss_id,
                    status,
                    updated_at, updated_at AS created_at
                FROM chunks 
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_chunk_id_by_faiss_id(self, faiss_id: int) -> Optional[str]:
        """Map FAISS integer embedding ID to SHA-256 chunk_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id FROM chunks WHERE embedding_id = ?", (faiss_id,))
            row = cursor.fetchone()
            return row["chunk_id"] if row else None

    def get_faiss_id_by_chunk_id(self, chunk_id: str) -> Optional[int]:
        """Map SHA-256 chunk_id to FAISS integer embedding ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding_id FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            return int(row["embedding_id"]) if row else None

    def get_all_faiss_mappings(self) -> Dict[int, str]:
        """Return full mapping of {faiss_id: chunk_id} stored in SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding_id, chunk_id FROM chunks ORDER BY embedding_id ASC")
            return {int(row["embedding_id"]): row["chunk_id"] for row in cursor.fetchall()}

    def upsert_page(
        self,
        url: str,
        etag: Optional[str] = None,
        last_crawled: Optional[str] = None,
        title: str = "",
        content_type: str = "html",
        page_hash: str = "",
        chunk_count: int = 0,
        **kwargs,
    ) -> None:
        """Upsert record in pages(url, etag, last_crawled) with metadata."""
        now = last_crawled or kwargs.get("crawled_at") or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO pages (url, etag, last_crawled, title, content_type, page_hash, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    etag = COALESCE(excluded.etag, pages.etag),
                    last_crawled = excluded.last_crawled,
                    title = CASE WHEN excluded.title != '' THEN excluded.title ELSE pages.title END,
                    content_type = excluded.content_type,
                    page_hash = excluded.page_hash,
                    chunk_count = excluded.chunk_count
                """,
                (url, etag, now, title, content_type, page_hash, chunk_count),
            )

    def insert_chunk(
        self,
        chunk_id: str,
        site_id: Optional[str] = None,
        url: str = "",
        page_no: Optional[int] = None,
        heading: Optional[str] = None,
        text: str = "",
        hash: Optional[str] = None,
        embedding_id: int = 0,
        status: str = "active",
        updated_at: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Insert or replace record in chunks(chunk_id, site_id, url, page_no, heading, text, hash, embedding_id, status, updated_at)."""
        now = updated_at or datetime.now(timezone.utc).isoformat()
        resolved_site_id = site_id or self.site_id
        resolved_hash = hash or kwargs.get("chunk_hash") or chunk_id
        resolved_page_no = page_no if page_no is not None else kwargs.get("page_number")
        resolved_embedding_id = embedding_id if embedding_id != 0 else kwargs.get("faiss_id", 0)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, site_id, url, page_no, heading, text, hash, embedding_id, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    resolved_site_id,
                    url,
                    resolved_page_no,
                    heading,
                    text,
                    resolved_hash,
                    resolved_embedding_id,
                    status,
                    now,
                ),
            )

    def delete_chunks_by_ids(self, chunk_ids: List[str]) -> List[int]:
        """Delete chunks by chunk_id and return their embedding_ids so vector store can remove them."""
        if not chunk_ids:
            return []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in chunk_ids)
            cursor.execute(f"SELECT embedding_id FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
            embedding_ids = [row["embedding_id"] for row in cursor.fetchall()]

            cursor.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
            return embedding_ids

    def delete_chunks_by_faiss_ids(self, faiss_ids: List[int]) -> List[str]:
        """Delete chunks by embedding_id (FAISS ID) and return their chunk_ids."""
        if not faiss_ids:
            return []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in faiss_ids)
            cursor.execute(f"SELECT chunk_id FROM chunks WHERE embedding_id IN ({placeholders})", faiss_ids)
            chunk_ids = [row["chunk_id"] for row in cursor.fetchall()]

            cursor.execute(f"DELETE FROM chunks WHERE embedding_id IN ({placeholders})", faiss_ids)
            return chunk_ids

    def get_max_embedding_id(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COALESCE(MAX(embedding_id), -1) AS max_id FROM chunks")
            row = cursor.fetchone()
            return int(row["max_id"])

    def get_max_faiss_id(self) -> int:
        """Backward compatibility alias for get_max_embedding_id."""
        return self.get_max_embedding_id()

    def record_crawl_start(self) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO crawl_runs (started_at, status) VALUES (?, 'running')",
                (now,),
            )
            return cursor.lastrowid

    def record_crawl_finish(
        self,
        run_id: int,
        pages_discovered: int,
        pages_crawled: int,
        pages_unchanged: int,
        pages_updated: int,
        chunks_added: int,
        chunks_unchanged: int,
        chunks_removed: int,
        status: str = "completed",
        error: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE crawl_runs SET
                    finished_at = ?,
                    pages_discovered = ?,
                    pages_crawled = ?,
                    pages_unchanged = ?,
                    pages_updated = ?,
                    chunks_added = ?,
                    chunks_unchanged = ?,
                    chunks_removed = ?,
                    status = ?,
                    error = ?
                WHERE id = ?
                """,
                (
                    now,
                    pages_discovered,
                    pages_crawled,
                    pages_unchanged,
                    pages_updated,
                    chunks_added,
                    chunks_unchanged,
                    chunks_removed,
                    status,
                    error,
                    run_id,
                ),
            )

    def get_stats(self) -> dict:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total_pages FROM pages")
            total_pages = cursor.fetchone()["total_pages"]

            cursor.execute("SELECT COUNT(*) AS total_chunks FROM chunks")
            total_chunks = cursor.fetchone()["total_chunks"]

            cursor.execute("SELECT * FROM crawl_runs ORDER BY id DESC LIMIT 1")
            last_run = cursor.fetchone()
            last_run_dict = dict(last_run) if last_run else None

            return {
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "last_run": last_run_dict,
            }
