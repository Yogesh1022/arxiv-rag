"""Step 5: Chunk papers from PostgreSQL and display results."""

import logging

from src.models.paper import Paper
from src.processing.chunker import SemanticChunker
from src.storage.postgres_client import PostgresClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    db = PostgresClient()
    chunker = SemanticChunker()

    papers_data = db.get_parsed_papers(limit=10)
    print(f"Papers to chunk: {len(papers_data)}")

    for pd in papers_data:
        paper = Paper(**pd)
        chunks = chunker.chunk(paper)
        print(f"\n--- {paper.arxiv_id}: {paper.title[:50]}... ---")
        print(f"  Total chunks: {len(chunks)}")
        for i, c in enumerate(chunks[:5]):
            print(
                f"  Chunk {i}: [{c.chunk_type}] "
                f'section="{c.section_title}" '
                f"tokens={c.token_count} chars={c.char_count}"
            )
            print(f"    Preview: {c.content[:100]}...")
        if len(chunks) > 5:
            print(f"  ... and {len(chunks) - 5} more chunks")


if __name__ == "__main__":
    main()
