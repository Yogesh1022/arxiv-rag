# 🔬 ArXiv Paper Curator — RAG System

> **The Mother of AI — Phase 1**
>
> A production-grade, local-first Retrieval-Augmented Generation (RAG) system for ingesting, indexing, and querying academic research papers from arXiv.

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Ingestion | Apache Airflow | Daily paper fetching & processing |
| Parsing | Docling + EasyOCR | PDF → structured markdown |
| Metadata DB | PostgreSQL 16 | Papers, authors, chunks |
| Vector DB | OpenSearch 2.x | BM25 + kNN hybrid search |
| Embeddings | nomic-embed-text (Ollama) | Local embedding generation |
| RAG Engine | LlamaIndex | Retrieval → Re-ranking → Synthesis |
| LLM | Ollama (Llama 3) | Local inference |
| Backend | FastAPI | Async REST API |
| Frontend | Gradio | Chat interface |
| Observability | Langfuse | Tracing, evals, prompt versioning |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/arxiv-rag.git
cd arxiv-rag

# 2. Install dependencies
uv sync

# 3. Setup environment
cp .env.example .env

# 4. Start infrastructure
make up

# 5. Pull LLM models
make pull-models

# 6. Verify services
make verify

# 7. Start API + UI
make api   # terminal 1
make ui    # terminal 2
```

## Project Structure

```
arxiv-rag/
├── src/
│   ├── config/          # Settings & constants
│   ├── models/          # Pydantic domain models
│   ├── ingestion/       # arXiv client, PDF parser, OCR
│   ├── processing/      # Chunking, embeddings
│   ├── storage/         # PostgreSQL & OpenSearch clients
│   ├── retrieval/       # Hybrid search, re-ranking, context builder
│   ├── generation/      # LLM client, prompts, answer generator
│   ├── api/             # FastAPI application & routes
│   ├── ui/              # Gradio chat interface
│   └── observability/   # Langfuse tracing & RAGAS evals
├── dags/                # Airflow DAGs
├── docker/              # Docker Compose & service configs
├── tests/               # Unit, integration, e2e tests
├── scripts/             # Utility scripts
├── notebooks/           # Exploration notebooks
└── prompts/             # Version-controlled prompt templates
```

## Services

| Service | Port | URL |
|---------|------|-----|
| FastAPI | 8000 | http://localhost:8000/docs |
| Gradio UI | 7860 | http://localhost:7860 |
| PostgreSQL | 5432 | localhost:5432 |
| OpenSearch | 9200 | http://localhost:9200 |
| Ollama | 11434 | http://localhost:11434 |
| Airflow | 8080 | http://localhost:8080 |
| Langfuse | 3000 | http://localhost:3000 |

## Development

```bash
make lint       # Lint with ruff
make format     # Format with ruff
make test       # Run all tests
make test-unit  # Unit tests only
make eval       # RAGAS evaluation
```

## License

MIT
