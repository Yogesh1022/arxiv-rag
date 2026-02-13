-- ============================================
-- ArXiv RAG — PostgreSQL Schema
-- ============================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ──────────── Airflow DB ────────────
CREATE DATABASE airflow;

-- ──────────── Langfuse DB ────────────
CREATE DATABASE langfuse;

-- ──────────── Papers Table ────────────
CREATE TABLE IF NOT EXISTS papers (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    arxiv_id        VARCHAR(50) UNIQUE NOT NULL,
    title           TEXT NOT NULL,
    abstract        TEXT,
    categories      TEXT[] DEFAULT '{}',
    primary_category VARCHAR(20),
    published_date  TIMESTAMP WITH TIME ZONE,
    updated_date    TIMESTAMP WITH TIME ZONE,
    pdf_url         TEXT,
    html_url        TEXT,
    doi             VARCHAR(100),
    journal_ref     TEXT,
    raw_text        TEXT,
    parsed_content  JSONB,
    parsing_status  VARCHAR(20) DEFAULT 'pending',
    parsing_error   TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_papers_arxiv_id ON papers(arxiv_id);
CREATE INDEX idx_papers_published_date ON papers(published_date DESC);
CREATE INDEX idx_papers_categories ON papers USING GIN(categories);
CREATE INDEX idx_papers_parsing_status ON papers(parsing_status);
CREATE INDEX idx_papers_title_trgm ON papers USING GIN(title gin_trgm_ops);

-- ──────────── Authors Table ────────────
CREATE TABLE IF NOT EXISTS authors (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT NOT NULL,
    affiliation TEXT,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_authors_name ON authors(name);

-- ──────────── Paper-Author Junction ────────────
CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id    UUID REFERENCES papers(id) ON DELETE CASCADE,
    author_id   UUID REFERENCES authors(id) ON DELETE CASCADE,
    author_order INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (paper_id, author_id)
);

-- ──────────── Chunks Table ────────────
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id        UUID REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    section_title   TEXT,
    chunk_type      VARCHAR(30) DEFAULT 'text',
    token_count     INTEGER,
    char_count      INTEGER,
    embedding_model VARCHAR(100),
    embedding_dim   INTEGER,
    opensearch_id   VARCHAR(100),
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_chunks_paper_id ON chunks(paper_id);
CREATE INDEX idx_chunks_chunk_type ON chunks(chunk_type);

-- ──────────── Ingestion Runs ────────────
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_date        DATE NOT NULL,
    categories      TEXT[] DEFAULT '{}',
    papers_fetched  INTEGER DEFAULT 0,
    papers_parsed   INTEGER DEFAULT 0,
    papers_failed   INTEGER DEFAULT 0,
    chunks_created  INTEGER DEFAULT 0,
    status          VARCHAR(20) DEFAULT 'running',
    error_message   TEXT,
    started_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at    TIMESTAMP WITH TIME ZONE
);

-- ──────────── Query Logs ────────────
CREATE TABLE IF NOT EXISTS query_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text      TEXT NOT NULL,
    answer_text     TEXT,
    sources         JSONB DEFAULT '[]',
    retrieval_time_ms INTEGER,
    generation_time_ms INTEGER,
    total_time_ms   INTEGER,
    model_used      VARCHAR(100),
    num_chunks_retrieved INTEGER,
    user_feedback   INTEGER,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ──────────── Updated Trigger ────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_papers_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
