import hashlib
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from app.config import settings
from app.models.schemas import SiteSummary
from app.storage.database import SiteDatabase
from app.storage.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


class SiteManager:
    """Manages isolated per-website storage folders, databases, and vector stores."""

    def __init__(self, base_data_dir: Path = settings.DATA_DIR):
        self.sites_dir = base_data_dir / "sites"
        self.sites_dir.mkdir(parents=True, exist_ok=True)
        # Cache for open DBs and vector stores
        self._db_cache: Dict[str, SiteDatabase] = {}
        self._vector_cache: Dict[str, FaissVectorStore] = {}

    @staticmethod
    def generate_site_id(root_url: str) -> str:
        """Generate a clean, deterministic directory slug and ID for a given root URL."""
        parsed = urlparse(root_url.strip())
        netloc = parsed.netloc.lower() or "site"
        # Sanitize netloc for folder name
        clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", netloc)
        url_hash = hashlib.sha256(root_url.strip().encode("utf-8")).hexdigest()[:8]
        return f"{clean_name}_{url_hash}"

    def get_site_dir(self, site_id: str) -> Path:
        return self.sites_dir / site_id

    def get_metadata_path(self, site_id: str) -> Path:
        return self.get_site_dir(site_id) / "metadata.json"

    def read_metadata(self, site_id: str) -> Optional[dict]:
        meta_file = self.get_metadata_path(site_id)
        if not meta_file.exists():
            return None
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata for {site_id}: {e}")
            return None

    def write_metadata(self, site_id: str, meta: dict) -> None:
        meta_file = self.get_metadata_path(site_id)
        tmp_file = meta_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        tmp_file.replace(meta_file)

    def create_or_get_site(
        self,
        root_url: str,
        name: Optional[str] = None,
        max_depth: int = 3,
        max_pages: int = 50,
    ) -> SiteSummary:
        site_id = self.generate_site_id(root_url)
        site_dir = self.get_site_dir(site_id)
        site_dir.mkdir(parents=True, exist_ok=True)

        meta = self.read_metadata(site_id)
        if not meta:
            parsed = urlparse(root_url)
            display_name = name or parsed.netloc or "Website"
            meta = {
                "id": site_id,
                "root_url": root_url.strip(),
                "name": display_name,
                "status": "idle",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_crawled_at": None,
                "max_depth": max_depth,
                "max_pages": max_pages,
            }
            self.write_metadata(site_id, meta)

        db = self.get_database(site_id)
        stats = db.get_stats()

        return SiteSummary(
            id=site_id,
            root_url=meta["root_url"],
            name=meta["name"],
            status=meta.get("status", "idle"),
            created_at=meta["created_at"],
            last_crawled_at=meta.get("last_crawled_at"),
            total_pages=stats["total_pages"],
            total_chunks=stats["total_chunks"],
        )

    def list_sites(self) -> List[SiteSummary]:
        summaries: List[SiteSummary] = []
        if not self.sites_dir.exists():
            return summaries

        for entry in self.sites_dir.iterdir():
            if entry.is_dir() and (entry / "metadata.json").exists():
                meta = self.read_metadata(entry.name)
                if meta:
                    db = self.get_database(entry.name)
                    stats = db.get_stats()
                    summaries.append(
                        SiteSummary(
                            id=entry.name,
                            root_url=meta["root_url"],
                            name=meta["name"],
                            status=meta.get("status", "idle"),
                            created_at=meta["created_at"],
                            last_crawled_at=meta.get("last_crawled_at"),
                            total_pages=stats["total_pages"],
                            total_chunks=stats["total_chunks"],
                        )
                    )
        return sorted(summaries, key=lambda s: s.created_at, reverse=True)

    def get_site(self, site_id: str) -> Optional[SiteSummary]:
        meta = self.read_metadata(site_id)
        if not meta:
            return None
        db = self.get_database(site_id)
        stats = db.get_stats()
        return SiteSummary(
            id=site_id,
            root_url=meta["root_url"],
            name=meta["name"],
            status=meta.get("status", "idle"),
            created_at=meta["created_at"],
            last_crawled_at=meta.get("last_crawled_at"),
            total_pages=stats["total_pages"],
            total_chunks=stats["total_chunks"],
        )

    def get_database(self, site_id: str) -> SiteDatabase:
        if site_id not in self._db_cache:
            db_path = self.get_site_dir(site_id) / "site.db"
            self._db_cache[site_id] = SiteDatabase(db_path, site_id=site_id)
        return self._db_cache[site_id]

    def get_vector_store(self, site_id: str) -> FaissVectorStore:
        if site_id not in self._vector_cache:
            index_path = self.get_site_dir(site_id) / "faiss.index"
            self._vector_cache[site_id] = FaissVectorStore(index_path, dimension=settings.EMBEDDING_DIM)
        return self._vector_cache[site_id]

    def delete_site(self, site_id: str) -> bool:
        site_dir = self.get_site_dir(site_id)
        if not site_dir.exists():
            return False

        # Reset FAISS index to release C-level references before removing files
        vec_store = self._vector_cache.pop(site_id, None)
        if vec_store is not None:
            import faiss
            base_index = faiss.IndexFlatIP(vec_store.dimension)
            vec_store.index = faiss.IndexIDMap(base_index)

        self._db_cache.pop(site_id, None)

        try:
            shutil.rmtree(site_dir)
            logger.info(f"Deleted site knowledge base: {site_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting site {site_id}: {e}")
            return False
