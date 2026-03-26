from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = "ollama"          # "ollama" (default) | "openai"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"        # free, ~2GB; alternatives: mistral, gemma2:2b
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "acme_hr_docs"

    # ── Redis / Semantic Cache ────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600

    # ── Retrieval ─────────────────────────────────────────────────────────────
    vector_top_k: int = 10
    bm25_top_k: int = 10
    rerank_top_k: int = 5

    # ── Guardrails ────────────────────────────────────────────────────────────
    max_llm_retries: int = 2
    hallucination_threshold: float = 0.25

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = 60

    # ── Prompt registry ───────────────────────────────────────────────────────
    prompt_template_version: str = "v2"
    prompt_templates_path: str = "prompts/templates.yaml"

    # ── Admin ─────────────────────────────────────────────────────────────────
    admin_token: str = "change_me_before_running"

    # ── Langfuse ─────────────────────────────────────────────────────────────
    langfuse_enabled: bool = False
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
