import logging
from typing import List, Optional, Union
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Wrapper around sentence-transformers embedding model running 100% locally."""

    _instance: Optional["LocalEmbedder"] = None

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._device = self._detect_device()

    @classmethod
    def get_instance(cls, model_name: str = settings.EMBEDDING_MODEL_NAME) -> "LocalEmbedder":
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("Using Apple Silicon MPS acceleration for embeddings.")
                return "mps"
            elif torch.cuda.is_available():
                logger.info("Using CUDA acceleration for embeddings.")
                return "cuda"
        except Exception:
            pass
        logger.info("Using CPU for embeddings.")
        return "cpu"

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local SentenceTransformer model: {self.model_name} on {self._device}...")
            self._model = SentenceTransformer(self.model_name, device=self._device)
            logger.info("SentenceTransformer model successfully loaded.")
        return self._model

    def encode(self, texts: Union[str, List[str]], batch_size: int = settings.EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """Generate normalized float32 embeddings for text(s) in batches."""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = settings.EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """Embed a batch of text strings using the local model."""
        return self.encode(texts, batch_size=batch_size)

    def embed_chunks(self, chunks: List, batch_size: int = settings.EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """
        Embed a list of DocumentChunks (or objects/dicts containing chunk text) in batches.
        Returns float32 numpy array with shape (len(chunks), EMBEDDING_DIM).
        """
        if not chunks:
            return np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)

        texts: List[str] = []
        for c in chunks:
            if hasattr(c, "text"):
                texts.append(c.text)
            elif isinstance(c, dict) and "text" in c:
                texts.append(c["text"])
            else:
                texts.append(str(c))

        return self.encode(texts, batch_size=batch_size)
