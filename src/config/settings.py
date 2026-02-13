"""Application settings via environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── PostgreSQL ──
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "arxiv"
    POSTGRES_PASSWORD: str = "arxiv_secret"
    POSTGRES_DB: str = "arxiv_rag"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn_sync(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── OpenSearch ──
    OPENSEARCH_HOST: str = "localhost"
    OPENSEARCH_PORT: int = 9200
    OPENSEARCH_INDEX: str = "arxiv_papers"
    OPENSEARCH_EMBEDDING_DIM: int = 768

    @property
    def opensearch_url(self) -> str:
        return f"http://{self.OPENSEARCH_HOST}:{self.OPENSEARCH_PORT}"

    # ── Ollama ──
    OLLAMA_HOST: str = "localhost"
    OLLAMA_PORT: int = 11434
    OLLAMA_LLM_MODEL: str = "llama3"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"

    # ── Langfuse ──
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"

    # ── RAG Config ──
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5
    SIMILARITY_THRESHOLD: float = 0.3

    # ── arXiv ──
    ARXIV_CATEGORIES: str = "cs.AI,cs.LG,cs.CL,cs.IR"
    ARXIV_MAX_RESULTS: int = 50

    @property
    def arxiv_categories_list(self) -> list[str]:
        """Parse comma-separated categories into a list."""
        return [s.strip() for s in self.ARXIV_CATEGORIES.split(",") if s.strip()]

    # ── API ──
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


settings = Settings()
