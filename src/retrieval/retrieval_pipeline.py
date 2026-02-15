"""Full retrieval pipeline: Search → Filter → Re-rank → Context Build."""

import logging
import time

from src.config.settings import settings
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import ReRanker

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """End-to-end retrieval pipeline."""

    def __init__(self) -> None:
        self.search_engine = HybridSearchEngine()
        self.reranker = ReRanker(top_n=settings.TOP_K_RERANK)
        self.context_builder = ContextBuilder()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        categories: list[str] | None = None,
        date_filter_days: int | None = None,
    ) -> dict:
        """Full retrieval: hybrid search → re-rank → format context.

        Args:
            query: The user's natural-language question.
            top_k: Number of candidates to fetch from hybrid search.
            categories: Optional arXiv category filter (e.g. ["cs.AI"]).
            date_filter_days: Optional recency filter in days.

        Returns:
            Dict with keys "context", "sources", and "retrieval_time_ms".
        """
        start = time.time()
        top_k = top_k or settings.TOP_K_RETRIEVAL

        # Step 1: Hybrid search
        raw_results = self.search_engine.search(
            query=query,
            top_k=top_k,
            categories=categories,
            date_filter_days=date_filter_days,
        )

        # Step 2: Re-rank
        reranked_results = self.reranker.rerank(query, raw_results)

        # Step 3: Build context
        context, sources = self.context_builder.build(reranked_results)

        elapsed_ms = int((time.time() - start) * 1000)

        return {
            "context": context,
            "sources": sources,
            "retrieval_time_ms": elapsed_ms,
        }
