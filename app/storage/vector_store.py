import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """Manages an isolated FAISS vector index using IndexIDMap and cosine similarity (IndexFlatIP)."""

    def __init__(self, index_file: Path, dimension: int = 384):
        self.index_file = Path(index_file)
        self.dimension = dimension
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self, path: Optional[Path] = None) -> None:
        """Load index from disk or initialize an empty IndexIDMap."""
        if path is not None:
            self.index_file = Path(path)

        if self.index_file.exists() and self.index_file.stat().st_size > 0:
            try:
                loaded = faiss.read_index(str(self.index_file))
                logger.info(f"Loaded existing FAISS index with {loaded.ntotal} vectors from {self.index_file}")
                self.index = loaded
                return
            except Exception as e:
                logger.error(f"Error loading FAISS index from {self.index_file}: {e}, creating a new one.")

        # Create IndexFlatIP (Inner Product) wrapped in IndexIDMap for custom integer ID mapping
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base_index)
        logger.info(f"Initialized new FAISS IndexIDMap(FlatIP) with dimension {self.dimension}")

    def save(self, path: Optional[Path] = None) -> None:
        """Persist index to disk atomically."""
        target_path = Path(path) if path is not None else self.index_file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = target_path.with_suffix(".tmp")
        faiss.write_index(self.index, str(tmp_file))
        tmp_file.replace(target_path)
        logger.debug(f"Saved FAISS index ({self.index.ntotal} vectors) to {target_path}")

    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        """Add batch of normalized embedding vectors with explicit integer IDs."""
        if len(ids) == 0:
            return

        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[0] != len(ids):
            raise ValueError(f"Mismatch between number of IDs ({len(ids)}) and vectors ({vectors.shape[0]})")

        # L2 Normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        id_array = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(vectors, id_array)
        self.save()
        logger.info(f"Added {len(ids)} vectors to FAISS index. Total: {self.index.ntotal}")

    def add_vectors(self, ids: List[int], vectors: np.ndarray) -> None:
        """Backward-compatible alias for add()."""
        self.add(ids, vectors)

    def remove(self, ids: List[int]) -> int:
        """Remove vectors by IDs using IndexIDMap.remove_ids."""
        if not ids or self.index.ntotal == 0:
            return 0

        id_array = np.array(ids, dtype=np.int64)
        removed_count = self.index.remove_ids(id_array)
        if removed_count > 0:
            self.save()
        logger.info(f"Removed {removed_count} vectors from FAISS index. Total: {self.index.ntotal}")
        return int(removed_count)

    def remove_vectors(self, ids: List[int]) -> int:
        """Backward-compatible alias for remove()."""
        return self.remove(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for nearest vectors using cosine similarity.
        Returns list of (faiss_id, similarity_score).
        """
        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(query_vector, dtype=np.float32)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        faiss.normalize_L2(query_vector)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, top_k)

        results: List[Tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:  # -1 represents unfilled slots in faiss
                results.append((int(idx), float(score)))

        return results

    def get_all_ids(self) -> List[int]:
        """Return list of all vector IDs currently in the index."""
        n = self.index.ntotal
        if n == 0:
            return []
        return [int(faiss.rev_swig_ptr(self.index.id_map.data(), n)[i]) for i in range(n)]

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal
