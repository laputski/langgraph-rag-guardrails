.PHONY: up up-langfuse down ollama-pull ingest dev test test-e2e test-unit lint help

help:
	@echo "Advanced RAG — available commands:"
	@echo ""
	@echo "  make up            Start required Docker services (Qdrant + Redis)"
	@echo "  make up-langfuse   Start all services including Langfuse observability"
	@echo "  make down          Stop all Docker services"
	@echo "  make ollama-pull   Pull the default Ollama model (llama3.2, ~2GB)"
	@echo "  make ingest        Load knowledge base into Qdrant"
	@echo "  make dev           Start FastAPI dev server (port 8000)"
	@echo "  make test          Run all tests"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-e2e      Run e2e tests only (requires running server + ingested data)"
	@echo "  make lint          Run ruff linter"

up:
	docker compose up -d --wait
	@echo ""
	@echo "✓ Services ready:"
	@echo "  Qdrant  → http://localhost:6333"
	@echo "  Redis   → localhost:6379"

up-langfuse:
	docker compose -f docker-compose.yml -f docker-compose.langfuse.yml up -d --wait
	@echo ""
	@echo "✓ Services ready:"
	@echo "  Qdrant   → http://localhost:6333"
	@echo "  Redis    → localhost:6379"
	@echo "  Langfuse → http://localhost:3000"

down:
	docker compose -f docker-compose.yml -f docker-compose.langfuse.yml down

ollama-pull:
	ollama pull llama3.2
	@echo "✓ llama3.2 ready. Alternative models: ollama pull mistral | gemma2:2b | phi3"

ingest:
	python -m ingest.run

dev:
	uvicorn app.main:app --reload --port 8000

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-e2e:
	pytest tests/e2e/ -v

lint:
	ruff check app/ tests/ ingest/
