"""Paper domain models."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


class ParsingStatus(str, StrEnum):
    PENDING = "pending"
    PARSED = "parsed"
    FAILED = "failed"


class Author(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    affiliation: str | None = None


class Paper(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    arxiv_id: str
    title: str
    abstract: str | None = None
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    published_date: datetime | None = None
    updated_date: datetime | None = None
    pdf_url: HttpUrl | None = None
    html_url: HttpUrl | None = None
    doi: str | None = None
    journal_ref: str | None = None
    authors: list[Author] = Field(default_factory=list)
    raw_text: str | None = None
    parsed_content: dict | None = None
    parsing_status: ParsingStatus = ParsingStatus.PENDING

    class Config:
        from_attributes = True


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    paper_id: UUID
    chunk_index: int
    content: str
    section_title: str | None = None
    chunk_type: str = "text"
    token_count: int | None = None
    char_count: int | None = None
    embedding_model: str | None = None
    metadata: dict = Field(default_factory=dict)
