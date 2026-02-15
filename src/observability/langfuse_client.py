"""Langfuse integration for tracing RAG pipeline operations."""

import logging
from contextlib import contextmanager

from langfuse import Langfuse

from src.config.settings import settings

logger = logging.getLogger(__name__)

_langfuse_client: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    """Initialize and return a Langfuse client singleton.

    Returns None if Langfuse keys are not configured.
    """
    global _langfuse_client  # noqa: PLW0603

    if _langfuse_client is not None:
        return _langfuse_client

    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.warning("Langfuse keys not configured — tracing disabled.")
        return None

    try:
        _langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        logger.info("Langfuse client initialized (host=%s)", settings.LANGFUSE_HOST)
        return _langfuse_client
    except Exception:
        logger.exception("Failed to initialise Langfuse client")
        return None


class LangfuseTracer:
    """Lightweight wrapper to trace RAG pipeline spans in Langfuse.

    If Langfuse is not configured the tracer gracefully no-ops.

    Usage::

        tracer = LangfuseTracer()
        with tracer.trace("ask", input={"question": q}) as trace:
            with tracer.span(trace, "retrieval") as span:
                results = search(...)
                tracer.end_span(span, output=results)
            with tracer.generation(trace, model="llama3", input=prompt) as gen:
                answer = llm.generate(prompt)
                tracer.end_generation(gen, output=answer)
            tracer.end_trace(trace, output=answer)
    """

    def __init__(self) -> None:
        self.client = get_langfuse()
        self.enabled = self.client is not None

    # ── Trace (top-level) ──────────────────────────────────────────
    @contextmanager
    def trace(self, name: str, *, input: dict | None = None, metadata: dict | None = None):  # noqa: A002
        """Create a top-level Langfuse trace.

        Yields the trace object (or a no-op dict).
        """
        if not self.enabled:
            yield {}
            return

        t = self.client.trace(name=name, input=input, metadata=metadata or {})
        try:
            yield t
        except Exception:
            t.update(metadata={"error": True})
            raise
        finally:
            self.client.flush()

    def end_trace(self, trace, *, output=None):
        """Attach final output to a trace."""
        if not self.enabled or not hasattr(trace, "update"):
            return
        trace.update(output=output)

    # ── Span (sub-step) ────────────────────────────────────────────
    @contextmanager
    def span(self, trace, name: str, *, input: dict | None = None):  # noqa: A002
        """Create a child span within a trace."""
        if not self.enabled or not hasattr(trace, "span"):
            yield {}
            return

        s = trace.span(name=name, input=input)
        try:
            yield s
        except Exception:
            s.update(metadata={"error": True})
            raise

    def end_span(self, span, *, output=None):
        """Attach output to a span."""
        if not self.enabled or not hasattr(span, "update"):
            return
        span.update(output=output)

    # ── Generation (LLM call) ─────────────────────────────────────
    @contextmanager
    def generation(
        self,
        trace,
        *,
        name: str = "llm",
        model: str = "",
        input: str | dict | None = None,  # noqa: A002
        metadata: dict | None = None,
    ):
        """Create a generation observation (for LLM calls)."""
        if not self.enabled or not hasattr(trace, "generation"):
            yield {}
            return

        g = trace.generation(name=name, model=model, input=input, metadata=metadata or {})
        try:
            yield g
        except Exception:
            g.update(metadata={"error": True})
            raise

    def end_generation(self, gen, *, output=None, usage: dict | None = None):
        """Attach output and token usage to a generation."""
        if not self.enabled or not hasattr(gen, "update"):
            return
        gen.update(output=output, usage=usage)


def shutdown() -> None:
    """Flush and shut down the Langfuse client."""
    global _langfuse_client  # noqa: PLW0603
    if _langfuse_client is not None:
        _langfuse_client.flush()
        _langfuse_client.shutdown()
        _langfuse_client = None
        logger.info("Langfuse client shut down.")
