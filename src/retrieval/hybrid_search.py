"""Hybrid search combining BM25 and vector similarity."""

import logging
from datetime import datetime, timedelta
from typing import Any

from src.processing.embeddings import EmbeddingGenerator
from src.storage.opensearch_client import OpenSearchClient

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Combined keyword + semantic search with metadata filtering."""

    def __init__(self) -> None:
        self.os_client = OpenSearchClient()
        self.embedder = EmbeddingGenerator()

    def search(
        self,
        query: str,
        top_k: int = 10,
        categories: list[str] | None = None,
        date_filter_days: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute hybrid search with optional filters."""
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Build filters
        filters: dict[str, Any] = {}
        if categories:
            filters["categories"] = categories
        if date_filter_days:
            date_from = (datetime.now() - timedelta(days=date_filter_days)).isoformat()
            filters["date_from"] = date_from

        # Execute hybrid search
        results = self.os_client.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters if filters else None,
        )

        logger.info("Hybrid search returned %d results for: '%s'", len(results), query)
        return results
