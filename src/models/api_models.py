"""API request/response schemas."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    date_filter_days: int | None = Field(default=None, ge=1, le=365)
    categories: list[str] | None = None
    model: str = "llama3"


class Source(BaseModel):
    paper_title: str
    arxiv_id: str
    arxiv_url: str
    section: str | None = None
    relevance_score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model_used: str
    retrieval_time_ms: int
    generation_time_ms: int
    total_time_ms: int
