"""arXiv API client for fetching paper metadata and PDFs."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import arxiv
import httpx

from src.config.settings import settings
from src.models.paper import Author, Paper

logger = logging.getLogger(__name__)


class ArxivClient:
    """Client for interacting with the arXiv API."""

    def __init__(
        self,
        categories: list[str] | None = None,
        max_results: int | None = None,
    ):
        self.categories = categories or settings.arxiv_categories_list
        self.max_results = max_results or settings.ARXIV_MAX_RESULTS
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,  # Respect rate limits
            num_retries=3,
        )

    def fetch_recent_papers(self, days_back: int = 1) -> list[Paper]:
        """Fetch papers published in the last N days for configured categories.

        Uses category-only queries sorted by submission date and filters
        by date client-side, because the arXiv API's ``submittedDate``
        range query is unreliable (often returns HTTP 500).
        """
        papers: list[Paper] = []
        cutoff = datetime.now() - timedelta(days=days_back)
        seen_ids: set[str] = set()

        for category in self.categories:
            query = f"cat:{category}"
            logger.info("Fetching papers: %s  (cutoff: %s)", query, cutoff.date())

            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            for result in self.client.results(search):
                arxiv_id = result.entry_id.split("/")[-1]

                # Skip duplicates across categories
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                # Client-side date filter
                if result.published and result.published.replace(tzinfo=None) < cutoff:
                    continue

                paper = Paper(
                    arxiv_id=arxiv_id,
                    title=result.title,
                    abstract=result.summary,
                    categories=result.categories or [category],
                    primary_category=result.primary_category,
                    published_date=result.published,
                    updated_date=result.updated,
                    pdf_url=result.pdf_url,
                    doi=result.doi or None,
                    journal_ref=result.journal_ref or None,
                    authors=[Author(name=a.name) for a in result.authors],
                )
                papers.append(paper)

        logger.info("Total papers fetched: %d", len(papers))
        return papers

    def download_pdf(self, paper: Paper, output_dir: str = "./data/pdfs") -> Path | None:
        """Download a paper's PDF to disk."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / f"{paper.arxiv_id.replace('/', '_')}.pdf"

        if output_path.exists():
            logger.info("PDF already exists: %s", output_path)
            return output_path

        try:
            response = httpx.get(str(paper.pdf_url), follow_redirects=True, timeout=60)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            logger.info("Downloaded: %s", output_path)
            return output_path
        except Exception:
            logger.exception("Failed to download %s", paper.arxiv_id)
            return None
