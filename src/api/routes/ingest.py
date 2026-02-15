"""POST /ingest endpoint — Manually trigger ingestion."""

from fastapi import APIRouter, BackgroundTasks

router = APIRouter()


@router.post("/ingest")
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    categories: list[str] | None = None,
    days_back: int = 1,
) -> dict[str, str]:
    """Manually trigger paper ingestion pipeline."""

    async def run_ingestion() -> int:
        from src.ingestion.arxiv_client import ArxivClient

        client = ArxivClient(categories=categories)
        papers = client.fetch_recent_papers(days_back=days_back)
        return len(papers)

    background_tasks.add_task(run_ingestion)

    return {
        "status": "ingestion_started",
        "message": "Paper ingestion triggered in the background.",
    }
