"""Build formatted context from retrieved chunks for LLM prompting."""

import logging

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Format retrieved chunks into structured prompt context."""

    def __init__(self, max_context_tokens: int = 3000) -> None:
        self.max_context_tokens = max_context_tokens

    def build(self, chunks: list[dict]) -> tuple[str, list[dict]]:
        """Build context string and source list from retrieved chunks.

        Args:
            chunks: Re-ranked search results, each with at least
                    "content", "arxiv_id", "title", and "section_title" keys.

        Returns:
            Tuple of (context_string, sources_list).
        """
        if not chunks:
            return "", []

        context_parts: list[str] = []
        sources: list[dict] = []
        total_tokens = 0

        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk["content"]
            chunk_tokens = len(chunk_text.split())

            if total_tokens + chunk_tokens > self.max_context_tokens:
                break

            arxiv_id = chunk.get("arxiv_id", "unknown")
            title = chunk.get("title", "Untitled")
            section = chunk.get("section_title", "")

            context_parts.append(
                f"[Source {i}] (Paper: {title} | arXiv: {arxiv_id} | "
                f"Section: {section})\n{chunk_text}"
            )

            sources.append(
                {
                    "paper_title": title,
                    "arxiv_id": arxiv_id,
                    "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "section": section,
                    "relevance_score": round(chunk.get("score", 0.0), 4),
                    "snippet": (chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text),
                }
            )

            total_tokens += chunk_tokens

        context_string = "\n\n---\n\n".join(context_parts)
        logger.info("Built context: %d sources, ~%d tokens", len(sources), total_tokens)
        return context_string, sources
