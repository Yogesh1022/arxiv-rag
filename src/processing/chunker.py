"""Chunking strategies for academic papers."""

import logging
import re
from uuid import uuid4

from src.config.settings import settings
from src.models.paper import Chunk, Paper

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Chunk academic papers with awareness of document structure."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk(self, paper: Paper) -> list[Chunk]:
        """Split paper text into semantically meaningful chunks."""
        if not paper.raw_text:
            return []

        sections = self._split_into_sections(paper.raw_text)
        chunks: list[Chunk] = []
        chunk_idx = 0

        for section_title, section_text in sections:
            section_chunks = self._recursive_chunk(section_text)
            for text in section_chunks:
                chunk = Chunk(
                    id=uuid4(),
                    paper_id=paper.id,
                    chunk_index=chunk_idx,
                    content=text.strip(),
                    section_title=section_title,
                    chunk_type=self._detect_chunk_type(text),
                    token_count=len(text.split()),
                    char_count=len(text),
                    metadata={
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "categories": paper.categories,
                        "published_date": (
                            paper.published_date.isoformat() if paper.published_date else None
                        ),
                    },
                )
                chunks.append(chunk)
                chunk_idx += 1

        logger.info("Paper %s: %d chunks created", paper.arxiv_id, len(chunks))
        return chunks

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Split markdown text into (section_title, content) pairs."""
        pattern = r"^(#{1,3})\s+(.+)$"
        sections: list[tuple[str, str]] = []
        current_title = "Introduction"
        current_content: list[str] = []

        for line in text.split("\n"):
            match = re.match(pattern, line)
            if match:
                if current_content:
                    sections.append((current_title, "\n".join(current_content)))
                current_title = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections.append((current_title, "\n".join(current_content)))

        return sections

    def _recursive_chunk(self, text: str) -> list[str]:
        """Recursively split text respecting paragraph boundaries."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Try splitting on double newlines (paragraphs)
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Handle oversized paragraphs
                if len(para) > self.chunk_size:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                            current_chunk += " " + sent if current_chunk else sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _detect_chunk_type(self, text: str) -> str:
        """Detect the type of content in a chunk."""
        if "|" in text and "---" in text:
            return "table"
        if re.search(r"\$\$.+\$\$", text, re.DOTALL):
            return "equation"
        if text.strip().startswith(("Figure", "Fig.")):
            return "figure_caption"
        return "text"
