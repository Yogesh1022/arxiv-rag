# ArXiv RAG — Complete End-to-End Run Guide

Run the entire pipeline from scratch: **Fetch → Parse → Store → Chunk → Embed → Index → Query**.

---

## Prerequisites

| Service        | Purpose                        | Default Port |
|----------------|--------------------------------|--------------|
| PostgreSQL 16  | Paper metadata & chunks store  | 5432         |
| OpenSearch 2.15| Vector + keyword hybrid search | 9200         |
| Ollama         | Embeddings + LLM generation    | 11434        |

> All three must be running before you start. Either use Docker Compose or local installs.

---

## Step 0: Start Infrastructure

### Option A — Docker Compose (recommended)

```powershell
cd e:\ARXVI_rag\arxiv-rag
docker compose -f docker/docker-compose.yml up -d postgres opensearch ollama
```

Wait for health checks:

```powershell
docker compose -f docker/docker-compose.yml ps
```

### Option B — Local services already running

Ensure PostgreSQL, OpenSearch, and Ollama are running on the ports in `.env`.

### Pull required Ollama models

```powershell
# Embedding model (768-dim vectors)
ollama pull nomic-embed-text

# LLM for answer generation
ollama pull llama3
```

Verify Ollama is ready:

```powershell
curl http://localhost:11434/api/tags
```

---

## Step 1: Initialize the Database Schema

Apply the schema (tables: `papers`, `authors`, `paper_authors`, `chunks`, `ingestion_runs`, `query_logs`):

```powershell
cd e:\ARXVI_rag\arxiv-rag
uv run python -m scripts.init_db
```

Verify tables exist:

```powershell
uv run python -m scripts.check_db
```

Expected output:

```
📋 Tables in 'arxiv_rag': ['authors', 'chunks', 'ingestion_runs', 'paper_authors', 'papers', 'query_logs']
📄 Papers count: 0
```

---

## Step 2: Fetch Papers from arXiv

This fetches recent papers for categories defined in `.env` (`cs.AI,cs.LG,cs.CL,cs.IR`).

```powershell
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)

from src.ingestion.arxiv_client import ArxivClient

client = ArxivClient(max_results=5)  # start small
papers = client.fetch_recent_papers(days_back=7)

print(f'\n=== Fetched {len(papers)} papers ===')
for p in papers[:5]:
    print(f'  [{p.arxiv_id}] {p.title[:80]}...')
    print(f'    Categories: {p.categories}')
    print(f'    Published:  {p.published_date}')
    print()
"
```

---

## Step 3: Download & Parse PDFs

Download PDFs and extract text using Docling:

```powershell
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)

from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.pdf_parser import PDFParser

client = ArxivClient(max_results=3)  # 3 papers for demo
papers = client.fetch_recent_papers(days_back=7)
parser = PDFParser()

for paper in papers[:3]:
    print(f'\n--- Processing: {paper.title[:60]}... ---')

    # Download PDF
    pdf_path = client.download_pdf(paper)
    if not pdf_path:
        print('  SKIP: download failed')
        continue

    # Parse PDF -> structured text
    result = parser.parse(pdf_path)
    paper.raw_text = result['text']
    paper.parsed_content = result
    paper.parsing_status = 'parsed'

    print(f'  PDF:      {pdf_path}')
    print(f'  Method:   {result[\"metadata\"][\"method\"]}')
    print(f'  Text len: {len(result[\"text\"]):,} chars')
    print(f'  Tables:   {len(result[\"tables\"])}')
    print(f'  Status:   {paper.parsing_status}')
"
```

> **Note:** First run downloads Docling ML models (~2 GB). Subsequent runs are faster.

---

## Step 4: Store Papers in PostgreSQL

Fetch, parse, and upsert into the `papers` table:

```powershell
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)

from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.pdf_parser import PDFParser
from src.storage.postgres_client import PostgresClient
from src.models.paper import ParsingStatus

client = ArxivClient(max_results=3)
papers = client.fetch_recent_papers(days_back=7)
parser = PDFParser()
db = PostgresClient()

parsed_papers = []
for paper in papers[:3]:
    pdf_path = client.download_pdf(paper)
    if not pdf_path:
        continue
    result = parser.parse(pdf_path)
    paper.raw_text = result['text']
    paper.parsed_content = result
    paper.parsing_status = ParsingStatus.PARSED
    parsed_papers.append(paper)

count = db.upsert_papers(parsed_papers)
print(f'\n=== Upserted {count} papers into PostgreSQL ===')
"
```

Verify in DB:

```powershell
uv run python -m scripts.check_db
```

Or query directly:

```powershell
uv run python -c "
from src.storage.postgres_client import PostgresClient
db = PostgresClient()
papers = db.get_parsed_papers(limit=10)
print(f'Parsed papers in DB: {len(papers)}')
for p in papers:
    print(f'  [{p[\"arxiv_id\"]}] {p[\"title\"][:60]}... ({len(p.get(\"raw_text\",\"\") or \"\"):,} chars)')
"
```

---

## Step 5: Chunk Papers (Semantic Chunking)

See how the chunker splits papers into sections and sized chunks:

```powershell
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)

from src.storage.postgres_client import PostgresClient
from src.processing.chunker import SemanticChunker
from src.models.paper import Paper

db = PostgresClient()
chunker = SemanticChunker()  # uses CHUNK_SIZE=512, CHUNK_OVERLAP=50

papers_data = db.get_parsed_papers(limit=5)
print(f'Papers to chunk: {len(papers_data)}')

for pd in papers_data:
    paper = Paper(**pd)
    chunks = chunker.chunk(paper)
    print(f'\n--- {paper.arxiv_id}: {paper.title[:50]}... ---')
    print(f'  Total chunks: {len(chunks)}')
    for i, c in enumerate(chunks[:5]):
        print(f'  Chunk {i}: [{c.chunk_type}] section=\"{c.section_title}\" '
              f'tokens={c.token_count} chars={c.char_count}')
        print(f'    Preview: {c.content[:100]}...')
    if len(chunks) > 5:
        print(f'  ... and {len(chunks)-5} more chunks')
"
```

---

## Step 6: Generate Embeddings (Ollama + nomic-embed-text)

Embed chunks into 768-dimensional vectors:

```powershell
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)

from src.storage.postgres_client import PostgresClient
from src.processing.chunker import SemanticChunker
from src.processing.embeddings import EmbeddingGenerator
from src.models.paper import Paper

db = PostgresClient()
chunker = SemanticChunker()
embedder = EmbeddingGenerator()  # model: nomic-embed-text, dim: 768

papers_data = db.get_parsed_papers(limit=1)  # 1 paper for demo
paper = Paper(**papers_data[0])
chunks = chunker.chunk(paper)

print(f'Embedding {len(chunks)} chunks from: {paper.title[:60]}...')
print(f'Model:     {embedder.model}')
print(f'Dimension: {embedder.dimension}')

embedded = embedder.embed_chunks(chunks)

for chunk, emb in embedded[:3]:
    print(f'\n  Chunk {chunk.chunk_index}: \"{chunk.content[:60]}...\"')
    print(f'    Embedding dim:  {len(emb)}')
    print(f'    First 5 values: {emb[:5]}')

print(f'\n=== Embedded {len(embedded)} chunks total ===')
"
```

---

## Step 7: Index into OpenSearch (Vector DB)

Create the hybrid index and bulk-insert chunk embeddings:

```powershell
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)

from src.storage.opensearch_client import OpenSearchClient

os_client = OpenSearchClient()
os_client.create_index()

# Verify index exists
info = os_client.client.indices.get(index='arxiv_papers')
mappings = info['arxiv_papers']['mappings']['properties']
print('=== OpenSearch Index: arxiv_papers ===')
print(f'Fields: {list(mappings.keys())}')
print(f'Embedding type: {mappings[\"embedding\"][\"type\"]}')
print(f'Embedding dim:  {mappings[\"embedding\"][\"dimension\"]}')
print(f'Engine:         {mappings[\"embedding\"][\"method\"][\"engine\"]}')
print(f'Space type:     {mappings[\"embedding\"][\"method\"][\"space_type\"]}')
"
```

Now run the full backfill (chunks + embeds + indexes all at once):

```powershell
uv run python -m scripts.backfill_opensearch
```

Verify document count:

```powershell
uv run python -c "
from src.storage.opensearch_client import OpenSearchClient
os_client = OpenSearchClient()
count = os_client.client.count(index='arxiv_papers')
print(f'Documents in OpenSearch: {count[\"count\"]}')
"
```

---

## Step 8: Search (Hybrid BM25 + kNN)

Run a hybrid search combining keyword matching and vector similarity:

```powershell
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)

from src.retrieval.hybrid_search import HybridSearchEngine

engine = HybridSearchEngine()
results = engine.search(
    query='What are the latest advances in retrieval augmented generation?',
    top_k=5,
)

print(f'\n=== Hybrid Search: {len(results)} results ===')
for i, r in enumerate(results, 1):
    print(f'\n  [{i}] Score: {r.get(\"score\", 0):.4f}')
    print(f'      Paper: {r.get(\"title\", \"N/A\")[:60]}')
    print(f'      arXiv: {r.get(\"arxiv_id\", \"N/A\")}')
    print(f'      Section: {r.get(\"section_title\", \"N/A\")}')
    print(f'      Content: {r.get(\"content\", \"\")[:120]}...')
"
```

---

## Step 9: Re-Rank with Cross-Encoder

Apply the BAAI/bge-reranker-base cross-encoder to refine relevance:

```powershell
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)

from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import ReRanker

engine = HybridSearchEngine()
reranker = ReRanker(top_n=3)

query = 'What are the latest advances in retrieval augmented generation?'
results = engine.search(query=query, top_k=10)

print(f'Search returned {len(results)} results')
reranked = reranker.rerank(query, results)

print(f'\n=== Re-ranked to top {len(reranked)} ===')
for i, r in enumerate(reranked, 1):
    print(f'\n  [{i}] Rerank score: {r[\"rerank_score\"]:.4f}')
    print(f'      Paper: {r.get(\"title\", \"N/A\")[:60]}')
    print(f'      Content: {r.get(\"content\", \"\")[:120]}...')
"
```

> **Note:** First run downloads the reranker model (~1.1 GB). Subsequent runs use cache.

---

## Step 10: Full RAG Query (Retrieve + Generate)

Run the complete pipeline — search, re-rank, build context, generate answer with Ollama:

```powershell
uv run python -c "
import asyncio
import logging
logging.basicConfig(level=logging.INFO)

from src.generation.answer_generator import AnswerGenerator

async def main():
    gen = AnswerGenerator()
    result = await gen.ask(
        question='What are the latest advances in retrieval augmented generation?',
        top_k=5,
    )

    print('\n' + '='*60)
    print('ANSWER:')
    print('='*60)
    print(result['answer'])
    print('\n' + '-'*60)
    print(f'Model:           {result[\"model_used\"]}')
    print(f'Retrieval time:  {result[\"retrieval_time_ms\"]}ms')
    print(f'Generation time: {result[\"generation_time_ms\"]}ms')
    print(f'Total time:      {result[\"total_time_ms\"]}ms')
    print(f'Sources:         {len(result[\"sources\"])}')
    for i, s in enumerate(result['sources'], 1):
        print(f'  [{i}] {s[\"paper_title\"][:50]}... (score: {s[\"relevance_score\"]:.4f})')
        print(f'      {s[\"arxiv_url\"]}')

asyncio.run(main())
"
```

---

## Step 11: Start the API Server

```powershell
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Test endpoints:

```powershell
# Health Check
curl http://localhost:8000/health

# Ask a question (POST)
curl -X POST http://localhost:8000/api/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What are recent advances in RAG?\", \"top_k\": 5}"
```

API Docs (Swagger UI): http://localhost:8000/docs

---

## All-in-One Script

Run **Steps 2–7** in a single shot (fetch → parse → store → chunk → embed → index):

```powershell
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('pipeline')

from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.pdf_parser import PDFParser
from src.storage.postgres_client import PostgresClient
from src.processing.chunker import SemanticChunker
from src.processing.embeddings import EmbeddingGenerator
from src.storage.opensearch_client import OpenSearchClient
from src.models.paper import ParsingStatus, Paper

# ── Config ──
MAX_PAPERS = 3
DAYS_BACK = 7

# ── Init ──
arxiv_client = ArxivClient(max_results=MAX_PAPERS)
parser = PDFParser()
db = PostgresClient()
chunker = SemanticChunker()
embedder = EmbeddingGenerator()
os_client = OpenSearchClient()

# ── Step 1: Fetch papers ──
logger.info('STEP 1: Fetching papers from arXiv...')
papers = arxiv_client.fetch_recent_papers(days_back=DAYS_BACK)
logger.info('Fetched %d papers', len(papers))

# ── Step 2: Download & Parse PDFs ──
logger.info('STEP 2: Downloading and parsing PDFs...')
parsed_papers = []
for paper in papers[:MAX_PAPERS]:
    pdf_path = arxiv_client.download_pdf(paper)
    if not pdf_path:
        continue
    result = parser.parse(pdf_path)
    paper.raw_text = result['text']
    paper.parsed_content = result
    paper.parsing_status = ParsingStatus.PARSED
    parsed_papers.append(paper)
    logger.info('  Parsed: %s (%d chars)', paper.arxiv_id, len(result['text']))

# ── Step 3: Store in PostgreSQL ──
logger.info('STEP 3: Storing %d papers in PostgreSQL...', len(parsed_papers))
db.upsert_papers(parsed_papers)

# ── Step 4: Create OpenSearch index ──
logger.info('STEP 4: Creating OpenSearch index...')
os_client.create_index()

# ── Step 5: Chunk + Embed + Index ──
logger.info('STEP 5: Chunking, embedding, and indexing...')
total_chunks = 0
for paper in parsed_papers:
    chunks = chunker.chunk(paper)
    if not chunks:
        continue
    embedded = embedder.embed_chunks(chunks)

    docs = []
    for chunk, emb in embedded:
        docs.append({
            'chunk_id': str(chunk.id),
            'paper_id': str(chunk.paper_id),
            'arxiv_id': chunk.metadata.get('arxiv_id', ''),
            'title': chunk.metadata.get('title', ''),
            'content': chunk.content,
            'embedding': emb,
            'section_title': chunk.section_title,
            'chunk_type': chunk.chunk_type,
            'chunk_index': chunk.chunk_index,
            'categories': chunk.metadata.get('categories', []),
            'published_date': chunk.metadata.get('published_date'),
        })

    os_client.bulk_index(docs)
    total_chunks += len(docs)
    logger.info('  Paper %s: %d chunks indexed', paper.arxiv_id, len(docs))

# ── Summary ──
db_papers = db.get_parsed_papers(limit=100)
os_count = os_client.client.count(index='arxiv_papers')
print()
print('='*60)
print('PIPELINE COMPLETE')
print('='*60)
print(f'  Papers in PostgreSQL: {len(db_papers)}')
print(f'  Chunks in OpenSearch: {os_count[\"count\"]}')
print(f'  Embedding model:      nomic-embed-text (768d)')
print(f'  Chunk size:           512 tokens')
print('='*60)
"
```

---

## Architecture Recap

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  FastAPI     │────▶│  Retrieval   │────▶│  OpenSearch   │
│  POST /ask   │     │  Pipeline    │     │  (Hybrid)    │
└─────────────┘     └──────┬───────┘     └──────────────┘
                           │
                    ┌──────▼───────┐
                    │  Re-Ranker   │  CrossEncoder
                    │  (top-5)     │  BAAI/bge-reranker
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Context     │  Format chunks
                    │  Builder     │  with citations
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐     ┌──────────────┐
                    │  Answer      │────▶│  Ollama      │
                    │  Generator   │     │  (llama3)    │
                    └──────────────┘     └──────────────┘
                           │
                    ┌──────▼───────┐
                    │  Response    │  answer + sources
                    │  + Sources   │  + timing metadata
                    └──────────────┘

Ingestion Pipeline (offline):
  arXiv API → PDF Download → Docling Parse → PostgreSQL
       → SemanticChunker → Ollama Embeddings → OpenSearch
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `connection refused` on PostgreSQL | Ensure PostgreSQL is running: `pg_isready -h localhost -p 5432` |
| `connection refused` on OpenSearch | Ensure OpenSearch is running: `curl http://localhost:9200` |
| `model not found` on Ollama | Pull models: `ollama pull nomic-embed-text && ollama pull llama3` |
| `Docling` slow on first run | Normal — downloads ~2 GB of ML models once |
| CrossEncoder slow first run | Normal — downloads ~1.1 GB reranker model once |
| `0 papers fetched` | Try `days_back=30` or use broader categories |
| DB permission errors | Run: `psql -U raguser -d arxiv_rag -c "GRANT ALL ON SCHEMA public TO arxiv"` |
