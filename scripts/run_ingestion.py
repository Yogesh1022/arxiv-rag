"""Fetch, parse, and store papers — run the full ingestion pipeline."""

import logging

from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.pdf_parser import PDFParser
from src.models.paper import ParsingStatus
from src.storage.postgres_client import PostgresClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(max_papers: int = 3, days_back: int = 7) -> None:
    client = ArxivClient(max_results=max_papers)
    parser = PDFParser()
    db = PostgresClient()

    # Step 1: Fetch
    logger.info("Fetching papers from arXiv (last %d days)...", days_back)
    papers = client.fetch_recent_papers(days_back=days_back)
    logger.info("Fetched %d papers", len(papers))

    # Step 2: Download + Parse
    parsed_papers = []
    for paper in papers[:max_papers]:
        logger.info("Processing: %s", paper.title[:60])
        pdf_path = client.download_pdf(paper)
        if not pdf_path:
            logger.warning("  SKIP: download failed for %s", paper.arxiv_id)
            continue

        result = parser.parse(pdf_path)
        paper.raw_text = result["text"]
        paper.parsed_content = result
        paper.parsing_status = ParsingStatus.PARSED
        parsed_papers.append(paper)

        logger.info(
            "  Parsed: %s | %d chars | method=%s",
            paper.arxiv_id,
            len(result["text"]),
            result["metadata"]["method"],
        )

    # Step 3: Store in PostgreSQL
    if parsed_papers:
        count = db.upsert_papers(parsed_papers)
        logger.info("Upserted %d papers into PostgreSQL", count)
    else:
        logger.warning("No papers were parsed successfully")

    # Summary
    print("\n" + "=" * 50)
    print(f"Papers fetched:  {len(papers)}")
    print(f"Papers parsed:   {len(parsed_papers)}")
    print("=" * 50)
    for p in parsed_papers:
        print(f"  [{p.arxiv_id}] {p.title[:70]}")


if __name__ == "__main__":
    main()
