"""OpenSearch client for vector + keyword hybrid storage."""

import logging
from typing import Any

from opensearchpy import OpenSearch, helpers

from src.config.settings import settings

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """Client for OpenSearch operations (indexing, search)."""

    def __init__(self) -> None:
        self.client = OpenSearch(
            hosts=[
                {
                    "host": settings.OPENSEARCH_HOST,
                    "port": settings.OPENSEARCH_PORT,
                }
            ],
            use_ssl=False,
            verify_certs=False,
        )
        self.index_name = settings.OPENSEARCH_INDEX
        self.embedding_dim = settings.OPENSEARCH_EMBEDDING_DIM

    def create_index(self) -> None:
        """Create the hybrid index with kNN + BM25 mappings."""
        if self.client.indices.exists(index=self.index_name):
            logger.info("Index '%s' already exists", self.index_name)
            return

        index_body: dict[str, Any] = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                },
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "paper_id": {"type": "keyword"},
                    "arxiv_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                    },
                    "section_title": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "categories": {"type": "keyword"},
                    "published_date": {"type": "date"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16,
                            },
                        },
                    },
                }
            },
        }

        self.client.indices.create(index=self.index_name, body=index_body)
        logger.info("Created index: %s", self.index_name)

    def index_chunk(
        self,
        chunk_id: str,
        paper_id: str,
        arxiv_id: str,
        title: str,
        content: str,
        embedding: list[float],
        section_title: str | None = None,
        chunk_type: str = "text",
        chunk_index: int = 0,
        categories: list[str] | None = None,
        published_date: str | None = None,
    ) -> None:
        """Index a single chunk document."""
        doc = {
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "title": title,
            "content": content,
            "embedding": embedding,
            "section_title": section_title,
            "chunk_type": chunk_type,
            "chunk_index": chunk_index,
            "categories": categories or [],
            "published_date": published_date,
        }
        self.client.index(index=self.index_name, id=chunk_id, body=doc)

    def bulk_index(self, documents: list[dict]) -> int:
        """Bulk index documents into OpenSearch."""
        actions = [
            {
                "_index": self.index_name,
                "_id": doc["chunk_id"],
                "_source": doc,
            }
            for doc in documents
        ]
        success, errors = helpers.bulk(self.client, actions)
        if errors:
            logger.warning("Bulk index errors: %d", len(errors))
        logger.info("Bulk indexed: %d success", success)
        return success

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute hybrid search combining BM25 + kNN.

        Uses two separate queries (BM25 keyword + kNN vector) and merges
        results via reciprocal-rank fusion to avoid nmslib engine limitations
        with top-level ``knn`` syntax.
        """
        filter_clauses: list[dict[str, Any]] = []
        if filters:
            if "categories" in filters:
                filter_clauses.append({"terms": {"categories": filters["categories"]}})
            if "date_from" in filters:
                filter_clauses.append({"range": {"published_date": {"gte": filters["date_from"]}}})

        # ── Phase 1: BM25 keyword search ──────────────────────────
        bm25_body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"content": {"query": query_text}}}],
                    "filter": filter_clauses,
                }
            },
        }
        bm25_resp = self.client.search(index=self.index_name, body=bm25_body)
        bm25_hits = bm25_resp["hits"]["hits"]

        # ── Phase 2: kNN vector search ────────────────────────────
        knn_body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                    }
                }
            },
        }
        if filter_clauses:
            knn_body["query"] = {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": top_k,
                                }
                            }
                        }
                    ],
                    "filter": filter_clauses,
                }
            }
        knn_resp = self.client.search(index=self.index_name, body=knn_body)
        knn_hits = knn_resp["hits"]["hits"]

        # ── Reciprocal Rank Fusion (RRF) ──────────────────────────
        rrf_k = 60  # standard RRF constant
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, hit in enumerate(bm25_hits):
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            docs[doc_id] = hit

        for rank, hit in enumerate(knn_hits):
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            docs[doc_id] = hit

        # Sort by fused score, take top_k
        ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]

        results = []
        for doc_id in ranked_ids:
            hit = docs[doc_id]
            results.append(
                {
                    "chunk_id": doc_id,
                    "score": scores[doc_id],
                    **hit["_source"],
                }
            )

        logger.info("Hybrid search returned %d results for: '%s'", len(results), query_text)
        return results
