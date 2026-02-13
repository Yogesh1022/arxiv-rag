"""Verify all service connections are healthy."""

import sys

import httpx
import psycopg2
from opensearchpy import OpenSearch

from src.config.settings import settings


def check_postgres() -> bool:
    """Check PostgreSQL connectivity."""
    try:
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            dbname=settings.POSTGRES_DB,
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        conn.close()
        print("✅ PostgreSQL: Connected")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL: {e}")
        return False


def check_opensearch() -> bool:
    """Check OpenSearch connectivity."""
    try:
        client = OpenSearch(
            hosts=[{"host": settings.OPENSEARCH_HOST, "port": settings.OPENSEARCH_PORT}],
            use_ssl=False,
        )
        info = client.info()
        print(f"✅ OpenSearch: Connected (v{info['version']['number']})")
        return True
    except Exception as e:
        print(f"❌ OpenSearch: {e}")
        return False


def check_ollama() -> bool:
    """Check Ollama connectivity and list available models."""
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=10)
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"✅ Ollama: Connected — Models: {models or 'none pulled yet'}")
        return True
    except Exception as e:
        print(f"❌ Ollama: {e}")
        return False


def check_langfuse() -> bool:
    """Check Langfuse reachability."""
    try:
        resp = httpx.get(settings.LANGFUSE_HOST, timeout=10)
        print(f"✅ Langfuse: Reachable (HTTP {resp.status_code})")
        return True
    except Exception as e:
        print(f"❌ Langfuse: {e}")
        return False


if __name__ == "__main__":
    print("\n🔍 Verifying Service Connections...\n")
    results = [check_postgres(), check_opensearch(), check_ollama(), check_langfuse()]
    print(f"\n{'=' * 40}")
    passed = sum(results)
    print(f"Result: {passed}/{len(results)} services healthy")
    sys.exit(0 if all(results) else 1)
