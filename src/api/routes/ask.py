"""POST /ask endpoint — Ask a question about research papers."""

from fastapi import APIRouter, HTTPException

from src.generation.answer_generator import AnswerGenerator
from src.models.api_models import AskRequest, AskResponse, Source

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """Ask a question and get an AI-generated answer with citations."""
    try:
        generator = AnswerGenerator(model=request.model)
        result = await generator.ask(
            question=request.question,
            top_k=request.top_k,
            categories=request.categories,
            date_filter_days=request.date_filter_days,
            model=request.model,
        )

        sources = [Source(**s) for s in result["sources"]]

        return AskResponse(
            answer=result["answer"],
            sources=sources,
            model_used=result["model_used"],
            retrieval_time_ms=result["retrieval_time_ms"],
            generation_time_ms=result["generation_time_ms"],
            total_time_ms=result["total_time_ms"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
