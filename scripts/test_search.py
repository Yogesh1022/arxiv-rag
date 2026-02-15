"""Quick test for hybrid search."""

from src.retrieval.hybrid_search import HybridSearchEngine

engine = HybridSearchEngine()
results = engine.search("What is AttentionRetriever?", top_k=3)
print(f"Results: {len(results)}")
for i, hit in enumerate(results, 1):
    title = hit.get("title", "?")[:60]
    score = hit["score"]
    print(f"  [{i}] {title} score={score:.4f}")
