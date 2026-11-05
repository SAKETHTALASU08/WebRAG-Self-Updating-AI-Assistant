import logging
import os
from pathlib import Path
from typing import List, Tuple
import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """Manages an isolated FAISS vector index using IndexIDMap2 and cosine similarity (IndexFlatIP)."""

    def __init__(self, index_file: Path, dimension: int = 384):
        self.index_file = index_file
        self.dimension = dimension
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.index = self._load_or_create()

    def _load_or_create(self) -> faiss.IndexIDMap2:
        if self.index_file.exists() and self.index_file.stat().st_size > 0:
            try:
                loaded = faiss.read_index(str(self.index_file))
                logger.info(f"Loaded existing FAISS index with {loaded.ntotal} vectors from {self.index_file}")
                return loaded
            except Exception as e:
                logger.error(f"Error loading FAISS index from {self.index_file}: {e}, creating a new one.")

        # Create IndexFlatIP (Inner Product) wrapped in IndexIDMap2 for custom integer ID mapping
        base_index = faiss.IndexFlatIP(self.dimension)
        id_index = faiss.IndexIDMap2(base_index)
        logger.info(f"Initialized new FAISS IndexIDMap2(FlatIP) with dimension {self.dimension}")
        return id_index

    def save(self) -> None:
        """Persist index to disk."""
        tmp_file = self.index_file.with_suffix(".tmp")
        faiss.write_index(self.index, str(tmp_file))
        tmp_file.replace(self.index_file)
        logger.debug(f"Saved FAISS index ({self.index.ntotal} vectors) to {self.index_file}")

    def add_vectors(self, ids: List[int], vectors: np.ndarray) -> None:
        """Add batch of normalized embedding vectors with explicit integer IDs."""
        if len(ids) == 0:
            return
        
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # L2 Normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        id_array = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(vectors, id_array)
        self.save()
        logger.info(f"Added {len(ids)} vectors to FAISS index. Total: {self.index.ntotal}")

    def remove_vectors(self, ids: List[int]) -> int:
        """Remove vectors by IDs, rebuilding the index to avoid FAISS internal corruption."""
        if not ids or self.index.ntotal == 0:
            return 0

        ids_to_remove = set(ids)

        # Extract all current vectors and their IDs
        n = self.index.ntotal
        all_ids = np.array(
            [faiss.rev_swig_ptr(self.index.id_map.data(), n)[i] for i in range(n)],
            dtype=np.int64,
        )
        # Reconstruct vectors from the sub-index
        all_vectors = np.empty((n, self.dimension), dtype=np.float32)
        for i in range(n):
            all_vectors[i] = self.index.reconstruct(int(all_ids[i]))

        # Filter out the vectors to remove
        keep_mask = np.array([int(aid) not in ids_to_remove for aid in all_ids])
        removed_count = int(np.sum(~keep_mask))

        if removed_count == 0:
            return 0

        kept_ids = all_ids[keep_mask]
        kept_vectors = all_vectors[keep_mask]

        # Rebuild index from scratch
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap2(base_index)

        if len(kept_ids) > 0:
            faiss.normalize_L2(kept_vectors)
            self.index.add_with_ids(kept_vectors, kept_ids)

        self.save()
        logger.info(f"Removed {removed_count} vectors (rebuild). Remaining: {self.index.ntotal}")
        return removed_count

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

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal
