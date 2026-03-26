# Advanced RAG — Reference Implementation

Reference implementation of **Advanced RAG** from scratch using LangGraph, hybrid retrieval, semantic caching, and production guardrails.

Knowledge base: **Acme Corp HR Policies** — 20 fictional policy documents, demonstrating the system end-to-end.

---

## What is Advanced RAG?

```
User Question
      │
      ▼
┌─────────────┐     blocked      ┌───────────────┐
│ AI Gateway  │─────────────────►│  Error (400)  │
│ Input Rails │  (injection/     └───────────────┘
└──────┬──────┘   toxic)
       │ pass
       ▼
┌─────────────────┐   cache hit   ┌───────────────────┐
│  Semantic Cache │──────────────►│ Streaming Response│
│  (Redis, TTL)   │               └───────────────────┘
└────────┬────────┘
         │ cache miss
         ▼
┌─────────────────┐
│  Embed Query    │  (sentence-transformers, local)
└────────┬────────┘
         ▼
┌─────────────────┐   ┌─────────────────┐
│ Dense Retrieval │   │ Sparse Retrieval │
│ (Qdrant HNSW)  │   │   (BM25)        │
└────────┬────────┘   └────────┬────────┘
         └─────────┬───────────┘
                   ▼
         ┌─────────────────┐
         │   RRF Fusion    │  1/(60+rank), deduplication
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │    Reranker     │  cross-encoder/ms-marco-MiniLM-L-6-v2
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │ Prompt Registry │  versioned YAML templates (v1, v2)
         └────────┬────────┘
                  ▼
         ┌─────────────────┐  ◄──────────────────────────┐
         │  LLM (OpenAI /  │                             │ retry
         │   Ollama)       │                             │ (max 2x)
         └────────┬────────┘                             │
                  ▼                                      │
         ┌─────────────────┐   fail (hallucination /    │
         │   Guardrails    │────toxicity)────────────────┘
         │   Output Rails  │
         └────────┬────────┘   pass
                  ▼
         ┌─────────────────┐
         │  Cache Store    │  save answer to Redis with TTL
         └────────┬────────┘
                  ▼
         ┌─────────────────────┐
         │  Streaming Response │  SSE token-by-token + done event
         └─────────────────────┘
```

### Components

| Component | What it does | Implementation |
|---|---|---|
| **Gateway + Input Rails** | Auth, rate limiting, injection/toxicity detection | `app/guardrails/input_guard.py` |
| **Semantic Cache** | Cosine-similarity lookup, skip LLM on hit (TTL-based) | `app/services/semantic_cache.py` |
| **Embedding Model** | Local dense embeddings (no external API) | `sentence-transformers/all-MiniLM-L6-v2` |
| **Hybrid Retrieval** | Dense (HNSW) + Sparse (BM25) in parallel | `app/graph/nodes/retrieve.py` |
| **RRF Fusion** | Merge ranked lists without score normalization | `app/graph/nodes/fuse_rerank.py` |
| **Reranker** | Cross-encoder scores each (query, chunk) pair jointly | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Prompt Registry** | Versioned YAML templates, A/B-testable | `prompts/templates.yaml` |
| **LLM** | OpenAI (default) or Ollama (sovereign/air-gapped) | `app/services/llm.py` |
| **Output Guardrails** | Hallucination detection (token overlap) + toxicity | `app/guardrails/output_guard.py` |
| **Streaming Response** | SSE token-by-token stream | `app/api/routes.py` |
| **Observability** | Langfuse self-hosted (optional Docker profile) | `docker-compose.langfuse.yml` |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- [Ollama](https://ollama.ai) (free, local — default) **or** an OpenAI API key (optional)

### 1. Clone and configure

```bash
git clone <repo>
cd langgraph-rag-guardrails
cp .env.example .env
# Default config uses Ollama (free). No API key needed.
# To use OpenAI instead: set LLM_PROVIDER=openai and OPENAI_API_KEY=sk-... in .env
```

### 2. Pull the Ollama model

```bash
make ollama-pull   # downloads llama3.2 (~2GB, one-time)
```

### 3. Start Docker services (Qdrant + Redis)

```bash
make up
```

### 4. Install Python dependencies

```bash
pip install -e ".[dev]"
```

> First run downloads ~160MB of model weights (embedding model + reranker).
> They are cached in `~/.cache/huggingface/` for subsequent runs.

### 5. Load the knowledge base

```bash
make ingest
# Expected: Indexed 140+ chunks from 20 documents
```

### 6. Start the server

```bash
make dev
# API docs: http://localhost:8000/docs
```

### 7. Ask a question

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many PTO days do I get per year?"}'
```

### 8. Run tests

```bash
make test           # all tests
make test-unit      # unit tests only (no Docker required)
make test-e2e       # e2e tests (requires running server)
```

---

## Configuration

All settings are read from `.env` (see `.env.example` for full list):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` (default, free) or `openai` |
| `OPENAI_API_KEY` | — | Required only when `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB |
| `REDIS_URL` | `redis://localhost:6379` | Redis for semantic cache |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity threshold for cache hits |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry TTL (1 hour) |
| `VECTOR_TOP_K` | `10` | Candidates retrieved by dense search |
| `BM25_TOP_K` | `10` | Candidates retrieved by BM25 |
| `RERANK_TOP_K` | `5` | Final chunks after cross-encoder reranking |
| `MAX_LLM_RETRIES` | `2` | Max guardrail retry attempts |
| `HALLUCINATION_THRESHOLD` | `0.25` | Min token overlap for hallucination detection |
| `PROMPT_TEMPLATE_VERSION` | `v2` | Prompt template version (`v1` or `v2`) |
| `ADMIN_TOKEN` | `change_me` | Token for `POST /ingest` |

---

## Sovereign AI / Local Mode (Ollama)

To run fully offline without any external API calls:

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2

# Set in .env:
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

make dev
```

The embedding model and reranker already run locally (no API key needed).
The only external dependency in default mode is OpenAI.

---

## Observability (Langfuse)

To enable distributed tracing of every pipeline step:

```bash
# Start all services including Langfuse
make up-langfuse

# In .env:
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

Open http://localhost:3000 to view traces. Create an account on first visit.

Langfuse uses its own Postgres database — this is **not** connected to the RAG pipeline.
Without Langfuse, the app works normally; traces are simply not recorded.

---

## API Reference

### `POST /query`

Returns a **Server-Sent Events** stream.

```json
{ "query": "How many PTO days do I get?", "user_id": "emp_123", "prompt_version": "v2" }
```

Events:
```
data: {"type": "token", "content": "You "}
data: {"type": "token", "content": "receive "}
...
data: {"type": "done", "sources": ["PTO Policy"], "from_cache": false, "request_id": "uuid", "retry_count": 0}
data: {"type": "error", "error_code": "max_retries_exceeded", "detail": "..."}
```

Errors before stream (HTTP 400):
```json
{"detail": {"error": "injection_detected", "detail": "Input appears to be a prompt injection attempt", "request_id": "uuid"}}
{"detail": {"error": "content_filtered", "detail": "Input contains prohibited content", "request_id": "uuid"}}
```

### `POST /ingest`

Requires admin token. Loads all knowledge base documents.

```json
{ "admin_token": "your_admin_token", "force_rebuild": false }
```

### `GET /health`

```json
{
  "status": "ok",
  "services": {
    "qdrant": "ok",
    "redis": "ok",
    "embedding_model": "loaded",
    "bm25_index": "loaded (147 chunks)"
  }
}
```

---

## Running Tests

### Unit tests (no Docker, no API key required)

```bash
make test-unit
```

Tests cover:
- `test_rrf_fusion.py` — RRF formula correctness, deduplication
- `test_semantic_cache.py` — cosine similarity threshold logic, cache hit/miss
- `test_prompt_registry.py` — template rendering, version fallback
- `test_guardrails.py` — injection patterns, toxicity, hallucination detection

### E2E tests (requires Docker + ingested data + API key)

```bash
make up && make ingest
make test-e2e
```

Happy path tests:
- Health check
- PTO query → grounded answer, correct sources
- Cache hit on repeated query
- 401k match, parental leave, remote work queries

Guardrails tests:
- Prompt injection (multiple patterns) → 400
- Toxic input → 400
- Off-topic question → graceful "I don't have information" response
- Hallucination retry → succeeds on second attempt
- Max retries exceeded → error in SSE stream

---

## Project Structure

```
├── app/
│   ├── config.py              # All settings via pydantic-settings
│   ├── main.py                # FastAPI app + lifespan startup
│   ├── api/                   # Routes, schemas, rate limiter
│   ├── graph/
│   │   ├── state.py           # RAGState TypedDict (shared pipeline state)
│   │   ├── builder.py         # LangGraph StateGraph assembly
│   │   └── nodes/             # One file per pipeline step
│   ├── services/              # Embedding, Qdrant, BM25, Redis, LLM, Reranker
│   └── guardrails/            # Input and output validation
├── ingest/                    # Knowledge base loader + chunker
├── knowledge_base/            # 20 Acme Corp HR policy documents (Markdown)
├── prompts/templates.yaml     # Versioned prompt templates
├── tests/
│   ├── unit/                  # Fast tests, no external services needed
│   └── e2e/                   # Full pipeline tests
├── docker-compose.yml         # Qdrant + Redis
├── docker-compose.langfuse.yml # Optional: Langfuse observability
└── postman_collection.json    # Importable Postman collection
```

---

## Postman Collection

Import `postman_collection.json` into Postman.

Set the `base_url` collection variable to `http://localhost:8000`.

The collection includes:
- **Happy Path** folder: health check, 6 HR queries, cache hit demonstrations
- **Guardrails — Input Blocked**: injection attacks, toxic input, validation errors
- **Guardrails — Edge Cases**: off-topic questions, long queries

Each request includes test assertions (`pm.test`) verifying status codes and response structure.
