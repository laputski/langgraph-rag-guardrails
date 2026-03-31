from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse

from app.api.schemas import (
    QueryRequest,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    ErrorResponse,
)
from app.api.middleware import limiter
from app.graph.state import initial_state
from app.config import settings

router = APIRouter()


# ── Dependency helpers ────────────────────────────────────────────────────────

def get_rag_graph(request: Request):
    return request.app.state.rag_graph


def get_services(request: Request):
    return request.app.state.services


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_from_state(state: dict) -> AsyncIterator[str]:
    """
    Stream the final answer token by token, then emit a 'done' event.

    For cache hits or error states the answer is emitted in a single chunk
    (no token-by-token breakdown needed — the answer already exists in full).
    """
    if state.get("error_code"):
        yield _sse_event({
            "type": "error",
            "error_code": state["error_code"],
            "detail": state.get("error", "Unknown error"),
        })
        return

    answer: str = state.get("final_answer") or ""

    # Emit tokens (word-by-word for a natural streaming feel)
    words = answer.split(" ")
    for i, word in enumerate(words):
        token = word if i == len(words) - 1 else word + " "
        yield _sse_event({"type": "token", "content": token})

    # Final metadata event
    yield _sse_event({
        "type": "done",
        "sources": state.get("sources", []),
        "from_cache": state.get("from_cache", False),
        "request_id": state.get("request_id", ""),
        "retry_count": max(0, state.get("llm_retry_count", 0) - 1),
        "model": state.get("llm_model_used", ""),
    })


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/query")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def query(
    body: QueryRequest,
    request: Request,
    rag_graph=Depends(get_rag_graph),
):
    """
    Main RAG query endpoint. Returns a Server-Sent Events (SSE) stream.

    Event format:
        data: {"type": "token", "content": "..."}
        data: {"type": "done", "sources": [...], "from_cache": bool, "request_id": "..."}
        data: {"type": "error", "error_code": "...", "detail": "..."}

    Errors that occur BEFORE the stream starts (blocked input) are returned
    as a plain JSON 400 response.
    """
    request_id = str(uuid.uuid4())
    prompt_version = body.prompt_version or settings.prompt_template_version

    # Build initial state
    state = initial_state(
        query=body.query,
        request_id=request_id,
        user_id=body.user_id,
        prompt_version=prompt_version,
    )

    # Start Langfuse trace (no-op when langfuse is disabled)
    lf_client = request.app.state.langfuse
    lf_trace = None
    if lf_client:
        lf_trace = lf_client.trace(
            id=request_id,
            name="rag-query",
            input=body.query,
            user_id=body.user_id,
        )

    # Run the LangGraph pipeline
    final_state = await rag_graph.ainvoke(state)

    if lf_trace:
        lf_trace.update(
            output=final_state.get("final_answer", ""),
            metadata={
                "from_cache": final_state.get("from_cache", False),
                "sources": final_state.get("sources", []),
                "retry_count": max(0, final_state.get("llm_retry_count", 0) - 1),
                "model": final_state.get("llm_model_used", ""),
                "error_code": final_state.get("error_code"),
            },
        )
        lf_client.flush()

    # If input was blocked, return a 400 before starting any SSE stream
    if not final_state.get("input_guard_passed", True) or final_state.get(
        "error_code"
    ) in ("content_filtered", "injection_detected"):
        error_code = final_state.get("error_code", "content_filtered")
        detail_map = {
            "content_filtered": "Input contains prohibited content",
            "injection_detected": "Input appears to be a prompt injection attempt",
        }
        raise HTTPException(
            status_code=400,
            detail={
                "error": error_code,
                "detail": detail_map.get(error_code, "Input blocked"),
                "request_id": request_id,
            },
        )

    return StreamingResponse(
        _stream_from_state(final_state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    body: IngestRequest,
    request: Request,
    services=Depends(get_services),
):
    """
    Trigger document ingestion (admin only).

    Loads all markdown files from knowledge_base/ and indexes them into Qdrant.
    Rebuilds the in-memory BM25 index after ingestion.
    """
    if body.admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    from ingest.loader import run_ingest

    vector_store = services["vector_store"]
    embedding_service = services["embedding_service"]
    bm25_index = services["bm25_index"]

    chunks_indexed, documents = await run_ingest(
        vector_store=vector_store,
        embedding_service=embedding_service,
        force_rebuild=body.force_rebuild,
    )

    # Rebuild BM25 index from the freshly ingested data
    all_docs = vector_store.get_all_documents()
    bm25_index.build(all_docs)

    return IngestResponse(
        status="ok",
        chunks_indexed=chunks_indexed,
        documents=documents,
    )


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    services=Depends(get_services),
):
    """Return the health status of all dependent services."""
    vector_store = services["vector_store"]
    semantic_cache = services["semantic_cache"]
    bm25_index = services["bm25_index"]

    qdrant_ok = vector_store.health()
    redis_ok = await semantic_cache.health()

    return HealthResponse(
        status="ok" if (qdrant_ok and redis_ok) else "degraded",
        services={
            "qdrant": "ok" if qdrant_ok else "unavailable",
            "redis": "ok" if redis_ok else "unavailable",
            "embedding_model": "loaded",
            "bm25_index": f"loaded ({bm25_index.size} chunks)" if bm25_index.is_ready else "empty",
        },
    )
