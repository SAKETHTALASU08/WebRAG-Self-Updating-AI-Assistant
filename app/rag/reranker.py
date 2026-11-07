import logging
import math
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class LocalReranker:
    """
    Reranks retrieved candidate chunks using BM25 / token-overlap scoring
    combined with vector cosine similarity, with optional CrossEncoder support.
    """

    def __init__(self, cross_encoder_model: Optional[str] = None):
        self.cross_encoder_model_name = cross_encoder_model
        self._cross_encoder = None

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\w+", text) if len(w) > 1]

    def _compute_bm25_score(self, query_tokens: List[str], doc_tokens: List[str], avg_doc_len: float = 100.0) -> float:
        """Compute BM25-style lexical relevance score."""
        if not query_tokens or not doc_tokens:
            return 0.0

        k1 = 1.5
        b = 0.75
        doc_len = len(doc_tokens)
        score = 0.0

        # Term frequency in document
        tf_dict = {}
        for token in doc_tokens:
            tf_dict[token] = tf_dict.get(token, 0) + 1

        for q in query_tokens:
            tf = tf_dict.get(q, 0)
            if tf > 0:
                # Approximate IDF assuming query term is informative
                idf = math.log(1.0 + (doc_len / (tf + 0.5)))
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1.0)))
                score += idf * (numerator / max(denominator, 1e-6))

        return score

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 5,
        dense_weight: float = 0.6,
    ) -> List[dict]:
        """
        Rerank candidate chunk dictionaries.
        Each candidate dictionary should contain 'text', 'similarity_score', and metadata.
        Returns reordered list of candidates with updated 'rerank_score'.
        """
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k]

        query_tokens = self._tokenize(query)

        # Compute lexical scores
        doc_token_lists = [self._tokenize(c.get("text", "")) for c in candidates]
        avg_doc_len = sum(len(d) for d in doc_token_lists) / max(len(doc_token_lists), 1)

        raw_bm25_scores = [
            self._compute_bm25_score(query_tokens, doc_tokens, avg_doc_len)
            for doc_tokens in doc_token_lists
        ]

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(raw_bm25_scores) if raw_bm25_scores and max(raw_bm25_scores) > 0 else 1.0
        normalized_bm25 = [score / max_bm25 for score in raw_bm25_scores]

        # Combine with dense similarity scores
        reranked = []
        for i, cand in enumerate(candidates):
            dense_sim = float(cand.get("similarity_score", 0.0))
            lex_sim = normalized_bm25[i]

            # Bonus for heading match
            heading = str(cand.get("heading") or "").lower()
            if any(q in heading for q in query_tokens if len(q) > 2):
                lex_sim = min(1.0, lex_sim + 0.2)

            combined_score = (dense_weight * dense_sim) + ((1.0 - dense_weight) * lex_sim)
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = round(combined_score, 4)
            reranked.append(cand_copy)

        # Sort descending by combined rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.debug(f"Reranked {len(candidates)} candidates, returning top {top_k}")
        return reranked[:top_k]
