.PHONY: help up down logs test lint format verify backfill api ui pull-models eval

help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Docker ──
up: ## Start all Docker services
	docker compose -f docker/docker-compose.yml up -d

down: ## Stop all Docker services
	docker compose -f docker/docker-compose.yml down

logs: ## Tail Docker logs
	docker compose -f docker/docker-compose.yml logs -f

# ── Development ──
verify: ## Verify all service connections
	uv run python scripts/verify_connections.py

backfill: ## Backfill OpenSearch from PostgreSQL
	uv run python scripts/backfill_opensearch.py

api: ## Start FastAPI server
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

ui: ## Start Gradio frontend
	uv run python src/ui/gradio_app.py

# ── Quality ──
test: ## Run all tests
	uv run pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	uv run pytest tests/unit/ -v -m unit

test-int: ## Run integration tests
	uv run pytest tests/integration/ -v -m integration

lint: ## Lint code with ruff
	uv run ruff check src/ tests/

format: ## Format code with ruff
	uv run ruff format src/ tests/

# ── Models ──
pull-models: ## Pull required Ollama models
	docker exec arxiv_ollama ollama pull llama3
	docker exec arxiv_ollama ollama pull nomic-embed-text

# ── Evaluation ──
eval: ## Run RAGAS evaluation
	uv run python scripts/run_ragas_eval.py
