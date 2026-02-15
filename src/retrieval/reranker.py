"""Cross-encoder re-ranking for improved relevance."""

import logging
from typing import Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class ReRanker:
    """Re-rank search results using a cross-encoder model."""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-base",
        top_n: int = 5,
    ) -> None:
        self.top_n = top_n
        self.model = CrossEncoder(model)

    def rerank(self, query: str, search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-rank search results using cross-encoder scoring.

        Args:
            query: The user's search query.
            search_results: List of dicts from hybrid search, each with
                            at least a "content" key.

        Returns:
            Top-N results re-ordered by cross-encoder relevance score.
        """
        if not search_results:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, r["content"]) for r in search_results]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort descending
        scored_results: list[dict[str, Any]] = []
        for result, score in zip(search_results, scores, strict=True):
            scored_results.append({**result, "rerank_score": float(score)})

        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        reranked = scored_results[: self.top_n]
        logger.info("Re-ranked %d → %d results", len(search_results), len(reranked))
        return reranked
