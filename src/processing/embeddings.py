"""Embedding generation using Ollama (nomic-embed-text)."""

import logging

import httpx

from src.config.settings import settings
from src.models.paper import Chunk

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings via Ollama's embedding API."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.OLLAMA_EMBED_MODEL
        self.base_url = settings.ollama_base_url
        self.dimension = settings.OPENSEARCH_EMBEDDING_DIM

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = [self.embed_text(t) for t in batch]
            all_embeddings.extend(batch_embeddings)
            logger.info("Embedded batch %d", i // batch_size + 1)
        return all_embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """Generate embeddings for chunks, returning (chunk, embedding) pairs."""
        texts = [c.content for c in chunks]
        embeddings = self.embed_batch(texts)

        results: list[tuple[Chunk, list[float]]] = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            chunk.embedding_model = self.model
            results.append((chunk, emb))

        return results
