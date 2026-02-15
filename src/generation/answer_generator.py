"""Answer generator combining retrieval + LLM generation."""

import logging
import time

from src.generation.llm_client import OllamaClient
from src.generation.prompt_templates import SYSTEM_PROMPT, format_prompt
from src.observability.langfuse_client import LangfuseTracer
from src.retrieval.retrieval_pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate answers with citations from retrieved context."""

    def __init__(self, model: str | None = None) -> None:
        self.retrieval = RetrievalPipeline()
        self.llm = OllamaClient(model=model)
        self.tracer = LangfuseTracer()

    async def ask(
        self,
        question: str,
        top_k: int = 5,
        categories: list[str] | None = None,
        date_filter_days: int | None = None,
        model: str | None = None,
    ) -> dict:
        """Full RAG pipeline: retrieve → generate → respond.

        Args:
            question: The user's natural-language question.
            top_k: Number of candidate chunks to retrieve.
            categories: Optional arXiv category filter.
            date_filter_days: Optional recency filter in days.
            model: Override the default LLM model for this request.

        Returns:
            Dict with answer, sources, model_used, and timing info.
        """
        with self.tracer.trace("ask", input={"question": question, "top_k": top_k}) as trace:
            # Step 1: Retrieve context
            with self.tracer.span(trace, "retrieval", input={"query": question}) as ret_span:
                retrieval_result = self.retrieval.retrieve(
                    query=question,
                    top_k=top_k,
                    categories=categories,
                    date_filter_days=date_filter_days,
                )
                context = retrieval_result["context"]
                sources = retrieval_result["sources"]
                retrieval_time = retrieval_result["retrieval_time_ms"]
                self.tracer.end_span(
                    ret_span,
                    output={
                        "num_sources": len(sources),
                        "retrieval_time_ms": retrieval_time,
                    },
                )

            # Step 2: Generate answer
            gen_start = time.time()
            if model:
                self.llm.model = model

            prompt = format_prompt(context=context, question=question)

            with self.tracer.generation(
                trace,
                name="llm",
                model=self.llm.model,
                input={"prompt": prompt[:500], "system_prompt": SYSTEM_PROMPT[:200]},
            ) as gen_obs:
                answer = await self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
                gen_time = int((time.time() - gen_start) * 1000)
                self.tracer.end_generation(gen_obs, output=answer[:500])

            total_time = retrieval_time + gen_time
            result = {
                "answer": answer,
                "sources": sources,
                "model_used": self.llm.model,
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": gen_time,
                "total_time_ms": total_time,
            }
            self.tracer.end_trace(trace, output={"total_time_ms": total_time})

        return result
