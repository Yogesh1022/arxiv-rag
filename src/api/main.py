"""FastAPI application — ArXiv RAG API."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import ask, health, ingest

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup/shutdown lifecycle."""
    logging.info("ArXiv RAG API starting up...")
    yield
    logging.info("ArXiv RAG API shutting down...")


app = FastAPI(
    title="ArXiv Paper Curator — RAG API",
    description="Ask questions about academic research papers from arXiv",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(ask.router, prefix="/api", tags=["Q&A"])
app.include_router(ingest.router, prefix="/api", tags=["Ingestion"])
