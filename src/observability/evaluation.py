"""RAGAS evaluation for RAG quality assessment."""

import asyncio
import logging

logger = logging.getLogger(__name__)


# ── Evaluation test set ───────────────────────────────────────────
EVAL_TEST_SET: list[dict[str, str]] = [
    {
        "question": "What are the main approaches to Retrieval-Augmented Generation?",
        "ground_truth": (
            "RAG combines retrieval of relevant documents with LLM generation. "
            "Key approaches include dense retrieval with embedding similarity, "
            "sparse keyword retrieval (BM25), and hybrid methods that combine both."
        ),
    },
    {
        "question": "How do transformer architectures handle long sequences?",
        "ground_truth": (
            "Techniques include sparse attention patterns, sliding window attention, "
            "linear attention approximations, memory-augmented approaches, and "
            "hierarchical chunking strategies that process documents in segments."
        ),
    },
    {
        "question": "What is the state of the art in few-shot learning?",
        "ground_truth": (
            "Recent approaches include meta-learning (MAML, Prototypical Networks), "
            "prompt-based / in-context learning with large language models, and "
            "transfer learning with fine-tuning on limited labelled data."
        ),
    },
]


async def run_ragas_evaluation(
    generator,
    test_set: list[dict[str, str]] | None = None,
) -> dict:
    """Run RAGAS evaluation on a test set using the AnswerGenerator.

    Args:
        generator: An ``AnswerGenerator`` instance (must have an ``.ask()`` method).
        test_set: List of dicts with ``question`` and ``ground_truth`` keys.
            Defaults to ``EVAL_TEST_SET``.

    Returns:
        Dict with per-metric scores (faithfulness, answer_relevancy,
        context_precision) as returned by ``ragas.evaluate()``.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        logger.error("Missing ragas / datasets packages. Install with: " "uv add ragas datasets")
        raise ImportError("ragas and datasets must be installed for evaluation") from exc

    test_set = test_set or EVAL_TEST_SET

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for idx, item in enumerate(test_set):
        q = item["question"]
        logger.info("Evaluating %d/%d: %s", idx + 1, len(test_set), q)
        try:
            result = await generator.ask(question=q)
            questions.append(q)
            answers.append(result["answer"])
            contexts.append([s["snippet"] for s in result.get("sources", [])])
            ground_truths.append(item["ground_truth"])
        except Exception:
            logger.exception("Failed to generate answer for: %s", q)
            continue

    if not questions:
        logger.error("No successful evaluations — aborting.")
        return {}

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    logger.info("Running RAGAS metrics …")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    logger.info("RAGAS Results: %s", result)
    return dict(result)


def run_sync(generator, test_set=None) -> dict:
    """Convenience wrapper to run the async evaluation synchronously."""
    return asyncio.run(run_ragas_evaluation(generator, test_set))
