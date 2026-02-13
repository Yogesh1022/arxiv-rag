"""PostgreSQL client for metadata storage operations."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import psycopg2
from psycopg2.extras import Json, execute_values

from src.config.settings import settings
from src.models.paper import Paper

logger = logging.getLogger(__name__)


class PostgresClient:
    """Synchronous PostgreSQL client for Airflow tasks."""

    def __init__(self) -> None:
        self.conn_params = {
            "host": settings.POSTGRES_HOST,
            "port": settings.POSTGRES_PORT,
            "user": settings.POSTGRES_USER,
            "password": settings.POSTGRES_PASSWORD,
            "dbname": settings.POSTGRES_DB,
        }

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Yield a psycopg2 connection with auto-commit / rollback."""
        conn = psycopg2.connect(**self.conn_params)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert_papers(self, papers: list[Paper]) -> int:
        """Insert or update papers in the database."""
        query = """
            INSERT INTO papers (arxiv_id, title, abstract, categories,
                primary_category, published_date, updated_date, pdf_url,
                doi, journal_ref, raw_text, parsed_content, parsing_status)
            VALUES %s
            ON CONFLICT (arxiv_id) DO UPDATE SET
                title = EXCLUDED.title,
                abstract = EXCLUDED.abstract,
                raw_text = EXCLUDED.raw_text,
                parsed_content = EXCLUDED.parsed_content,
                parsing_status = EXCLUDED.parsing_status,
                updated_at = NOW()
        """
        values = [
            (
                p.arxiv_id,
                p.title,
                p.abstract,
                p.categories,
                p.primary_category,
                p.published_date,
                p.updated_date,
                str(p.pdf_url) if p.pdf_url else None,
                p.doi,
                p.journal_ref,
                p.raw_text,
                Json(p.parsed_content),
                p.parsing_status.value,
            )
            for p in papers
        ]

        with self.get_connection() as conn:
            cur = conn.cursor()
            execute_values(cur, query, values)
            count = cur.rowcount
            logger.info("Upserted %d papers", count)
            return count

    def get_parsed_papers(self, limit: int = 100) -> list[dict]:
        """Retrieve parsed papers for indexing."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """SELECT id, arxiv_id, title, abstract, raw_text, categories,
                          published_date, parsed_content
                   FROM papers WHERE parsing_status = 'parsed'
                   ORDER BY published_date DESC LIMIT %s""",
                (limit,),
            )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row, strict=False)) for row in cur.fetchall()]
