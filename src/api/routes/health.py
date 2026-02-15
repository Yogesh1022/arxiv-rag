"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check."""
    return {"status": "healthy", "service": "arxiv-rag"}
