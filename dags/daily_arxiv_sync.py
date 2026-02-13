"""Airflow DAG: Daily arXiv paper sync and processing."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "arxiv-rag",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="daily_arxiv_sync",
    default_args=default_args,
    description="Daily sync of arXiv papers: fetch, parse, chunk, embed, index",
    schedule_interval="0 6 * * *",  # 6 AM UTC daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["arxiv", "ingestion"],
)


def fetch_papers(**context):
    """Step 1: Fetch metadata from arXiv API."""
    from src.ingestion.arxiv_client import ArxivClient

    client = ArxivClient()
    papers = client.fetch_recent_papers(days_back=1)
    paper_dicts = [p.model_dump(mode="json") for p in papers]
    context["ti"].xcom_push(key="papers", value=paper_dicts)
    return f"Fetched {len(papers)} papers"


def download_and_parse(**context):
    """Step 2: Download PDFs and parse with Docling."""
    from src.ingestion.arxiv_client import ArxivClient
    from src.ingestion.pdf_parser import PDFParser
    from src.models.paper import Paper

    paper_dicts = context["ti"].xcom_pull(task_ids="fetch_papers", key="papers")
    papers = [Paper(**p) for p in paper_dicts]

    client = ArxivClient()
    parser = PDFParser()
    parsed_papers = []

    for paper in papers:
        pdf_path = client.download_pdf(paper)
        if pdf_path:
            result = parser.parse(pdf_path)
            paper.raw_text = result["text"]
            paper.parsed_content = result
            paper.parsing_status = "parsed" if result["text"] else "failed"
        else:
            paper.parsing_status = "failed"
        parsed_papers.append(paper.model_dump(mode="json"))

    context["ti"].xcom_push(key="parsed_papers", value=parsed_papers)
    return f"Parsed {len(parsed_papers)} papers"


def store_metadata(**context):
    """Step 3: Save metadata to PostgreSQL."""
    from src.models.paper import Paper
    from src.storage.postgres_client import PostgresClient

    paper_dicts = context["ti"].xcom_pull(task_ids="download_and_parse", key="parsed_papers")
    papers = [Paper(**p) for p in paper_dicts]

    pg_client = PostgresClient()
    stored = pg_client.upsert_papers(papers)
    return f"Stored {stored} papers in PostgreSQL"


def chunk_and_embed(**context):
    """Step 4: Chunk text and generate embeddings."""
    from src.models.paper import Paper
    from src.processing.chunker import SemanticChunker
    from src.processing.embeddings import EmbeddingGenerator

    paper_dicts = context["ti"].xcom_pull(task_ids="download_and_parse", key="parsed_papers")
    papers = [Paper(**p) for p in paper_dicts if p.get("raw_text")]

    chunker = SemanticChunker()
    embedder = EmbeddingGenerator()

    all_chunks = []
    for paper in papers:
        chunks = chunker.chunk(paper)
        chunks_with_embeddings = embedder.embed_chunks(chunks)
        all_chunks.extend(chunks_with_embeddings)

    context["ti"].xcom_push(key="chunks_count", value=len(all_chunks))
    return f"Created {len(all_chunks)} chunks"


def index_to_opensearch(**context):
    """Step 5: Push chunks + embeddings to OpenSearch."""
    from src.storage.opensearch_client import OpenSearchClient

    os_client = OpenSearchClient()
    os_client.bulk_index_chunks()
    return "Indexed chunks to OpenSearch"


# ── Define DAG Tasks ──
t1 = PythonOperator(task_id="fetch_papers", python_callable=fetch_papers, dag=dag)
t2 = PythonOperator(task_id="download_and_parse", python_callable=download_and_parse, dag=dag)
t3 = PythonOperator(task_id="store_metadata", python_callable=store_metadata, dag=dag)
t4 = PythonOperator(task_id="chunk_and_embed", python_callable=chunk_and_embed, dag=dag)
t5 = PythonOperator(task_id="index_to_opensearch", python_callable=index_to_opensearch, dag=dag)

t1 >> t2 >> t3 >> t4 >> t5
