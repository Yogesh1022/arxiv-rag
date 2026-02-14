"""Backfill OpenSearch index from PostgreSQL data."""

import logging

from src.models.paper import Paper
from src.processing.chunker import SemanticChunker
from src.processing.embeddings import EmbeddingGenerator
from src.storage.opensearch_client import OpenSearchClient
from src.storage.postgres_client import PostgresClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backfill() -> None:
    """Pull parsed papers from PostgreSQL and index into OpenSearch."""
    pg = PostgresClient()
    os_client = OpenSearchClient()
    chunker = SemanticChunker()
    embedder = EmbeddingGenerator()

    # 1. Create index
    os_client.create_index()

    # 2. Fetch parsed papers
    papers_data = pg.get_parsed_papers(limit=500)
    logger.info("Found %d parsed papers", len(papers_data))

    # 3. Chunk + embed + index
    total_chunks = 0
    for paper_dict in papers_data:
        paper = Paper(**paper_dict)
        chunks = chunker.chunk(paper)
        embedded_chunks = embedder.embed_chunks(chunks)

        docs = []
        for chunk, embedding in embedded_chunks:
            docs.append(
                {
                    "chunk_id": str(chunk.id),
                    "paper_id": str(chunk.paper_id),
                    "arxiv_id": chunk.metadata.get("arxiv_id", ""),
                    "title": chunk.metadata.get("title", ""),
                    "content": chunk.content,
                    "embedding": embedding,
                    "section_title": chunk.section_title,
                    "chunk_type": chunk.chunk_type,
                    "chunk_index": chunk.chunk_index,
                    "categories": chunk.metadata.get("categories", []),
                    "published_date": chunk.metadata.get("published_date"),
                }
            )

        if docs:
            os_client.bulk_index(docs)
            total_chunks += len(docs)

    logger.info("Backfill complete: %d chunks indexed", total_chunks)


if __name__ == "__main__":
    backfill()
