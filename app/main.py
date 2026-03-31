from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router
from app.api.middleware import setup_rate_limiter
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.bm25_index import BM25Index
from app.services.semantic_cache import SemanticCache
from app.services.reranker import RerankerService
from app.services.llm import LLMService, build_llm
from app.services.prompt_registry import PromptRegistry
from app.graph.builder import build_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup / shutdown lifecycle.

    Startup sequence (blocking, in order):
      1. Load embedding model   (~2s, downloads ~90MB on first run)
      2. Connect Qdrant, ensure collection exists
      3. Connect Redis
      4. Load cross-encoder reranker (~1s, downloads ~68MB on first run)
      5. Fetch all docs from Qdrant → build BM25 index
      6. Load prompt registry (YAML)
      7. Build LLM client (OpenAI or Ollama)
      8. Compile LangGraph pipeline
    """
    print("🚀 Starting Advanced RAG service...")

    # 1. Embedding model
    print(f"  Loading embedding model: {settings.embedding_model}")
    embedding_service = EmbeddingService(settings.embedding_model)
    print(f"  ✓ Embedding model loaded (dim={embedding_service.dim})")

    # 2. Qdrant
    print(f"  Connecting to Qdrant: {settings.qdrant_url}")
    vector_store = VectorStoreService(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dim=embedding_service.dim,
    )
    vector_store.create_collection_if_not_exists()
    print(f"  ✓ Qdrant connected (collection: {settings.qdrant_collection})")

    # 3. Redis
    print(f"  Connecting to Redis: {settings.redis_url}")
    semantic_cache = SemanticCache(
        redis_url=settings.redis_url,
        similarity_threshold=settings.cache_similarity_threshold,
        ttl_seconds=settings.cache_ttl_seconds,
    )
    print("  ✓ Redis connected")

    # 4. Reranker
    print("  Loading cross-encoder reranker...")
    reranker = RerankerService()
    print("  ✓ Reranker loaded")

    # 5. BM25 index (rebuilt from Qdrant)
    print("  Building BM25 index from Qdrant documents...")
    bm25_index = BM25Index()
    all_docs = vector_store.get_all_documents()
    if all_docs:
        bm25_index.build(all_docs)
        print(f"  ✓ BM25 index built ({bm25_index.size} chunks)")
    else:
        print("  ⚠ BM25 index empty — run `make ingest` to load the knowledge base")

    # 6. Prompt registry
    print(f"  Loading prompt registry: {settings.prompt_templates_path}")
    prompt_registry = PromptRegistry(settings.prompt_templates_path)
    print(f"  ✓ Prompt registry loaded (versions: {prompt_registry.available_versions()})")

    # 7. LLM
    print(f"  Initializing LLM: provider={settings.llm_provider}")
    raw_model = build_llm(
        provider=settings.llm_provider,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
    )
    model_name = (
        settings.openai_model
        if settings.llm_provider == "openai"
        else settings.ollama_model
    )
    llm_service = LLMService(raw_model, model_name)
    print(f"  ✓ LLM ready ({model_name})")

    # 8. LangGraph pipeline
    print("  Compiling LangGraph pipeline...")
    rag_graph = build_graph(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_index=bm25_index,
        semantic_cache=semantic_cache,
        reranker=reranker,
        llm_service=llm_service,
        prompt_registry=prompt_registry,
        settings=settings,
    )
    print("  ✓ LangGraph pipeline compiled")

    # Store in app.state for dependency injection in routes
    app.state.rag_graph = rag_graph
    app.state.services = {
        "embedding_service": embedding_service,
        "vector_store": vector_store,
        "bm25_index": bm25_index,
        "semantic_cache": semantic_cache,
        "reranker": reranker,
        "llm_service": llm_service,
        "prompt_registry": prompt_registry,
    }

    # 9. Langfuse (optional observability)
    if settings.langfuse_enabled:
        from langfuse import Langfuse
        lf_client = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
        )
        print(f"  ✓ Langfuse tracing enabled → {settings.langfuse_host}")
    else:
        lf_client = None
    app.state.langfuse = lf_client

    print("\n✅ Advanced RAG service ready!")
    print(f"   API docs: http://localhost:8000/docs\n")

    yield

    # Shutdown
    print("Shutting down...")
    await semantic_cache.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Advanced RAG API",
        description=(
            "Reference implementation of Advanced RAG with LangGraph, "
            "hybrid retrieval, semantic cache, and guardrails."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    setup_rate_limiter(app, settings.rate_limit_per_minute)

    app.include_router(router)

    return app


app = create_app()
