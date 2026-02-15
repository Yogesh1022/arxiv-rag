"""Run RAGAS quality evaluation against the live RAG pipeline.

Usage:
    uv run python scripts/run_ragas_eval.py
"""

import asyncio
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    from src.generation.answer_generator import AnswerGenerator
    from src.observability.evaluation import EVAL_TEST_SET, run_ragas_evaluation

    logger.info("Initialising AnswerGenerator …")
    generator = AnswerGenerator()

    logger.info("Running RAGAS evaluation with %d test questions …", len(EVAL_TEST_SET))
    results = await run_ragas_evaluation(generator)

    if not results:
        logger.error("Evaluation returned no results.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("RAGAS Evaluation Results")
    print("=" * 60)
    for metric, score in results.items():
        print(f"  {metric:25s}: {score:.4f}")
    print("=" * 60)

    # Persist to file
    out_path = "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    asyncio.run(main())
