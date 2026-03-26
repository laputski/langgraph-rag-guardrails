from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.graph.state import RAGState
from app.graph.nodes.input_rails import input_rails_node
from app.graph.nodes.cache_lookup import make_cache_lookup_node
from app.graph.nodes.retrieve import make_retrieve_node
from app.graph.nodes.fuse_rerank import make_fuse_rerank_node
from app.graph.nodes.build_prompt import make_build_prompt_node
from app.graph.nodes.llm_generate import make_llm_generate_node
from app.graph.nodes.output_guardrails import make_output_guardrails_node
from app.graph.nodes.cache_store import make_cache_store_node
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.bm25_index import BM25Index
from app.services.semantic_cache import SemanticCache
from app.services.reranker import RerankerService
from app.services.llm import LLMService
from app.services.prompt_registry import PromptRegistry
from app.guardrails.output_guard import OutputGuard
from app.config import Settings


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_after_input_rails(state: RAGState) -> str:
    """Route to error terminal if input was blocked, else proceed to cache lookup."""
    return "cache_lookup" if state["input_guard_passed"] else END


def _route_after_cache_lookup(state: RAGState) -> str:
    """
    3a — Cache hit:  skip retrieval, go straight to END (answer is in state).
    3b — Cache miss: proceed through the full RAG pipeline.

    Note: cache_lookup already computed and stored query_embedding in state,
    so the retrieve node can use it directly without a separate embed step.
    """
    return END if state["cache_hit"] else "retrieve"


def _route_after_output_guardrails(state: RAGState) -> str:
    """
    Pass  → save to cache, then END.
    Retry → loop back to LLM (retry count is checked inside the node).
    Fail  → max retries exceeded, go to END with error in state.
    """
    if state["output_guard_passed"]:
        return "cache_store"
    if state.get("error_code") == "max_retries_exceeded":
        return END
    # Still retrying
    return "llm_generate"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    embedding_service: EmbeddingService,
    vector_store: VectorStoreService,
    bm25_index: BM25Index,
    semantic_cache: SemanticCache,
    reranker: RerankerService,
    llm_service: LLMService,
    prompt_registry: PromptRegistry,
    settings: Settings,
):
    """
    Assemble and compile the LangGraph RAG pipeline.

    Pipeline flow:
        START
          └─► input_rails
                ├─[blocked]─► END  (error in state)
                └─[pass]──► cache_lookup
                              ├─[hit]───► END  (answer from cache)
                              └─[miss]─► retrieve  (query_embedding already in state)
                                                 └─► fuse_rerank
                                                       └─► build_prompt
                                                             └─► llm_generate ◄─┐
                                                                   └─► output_guardrails
                                                                         ├─[pass]──► cache_store → END
                                                                         ├─[retry]──────────────────┘
                                                                         └─[max retries]──► END
    """
    output_guard = OutputGuard(
        hallucination_threshold=settings.hallucination_threshold
    )

    graph = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("input_rails", input_rails_node)

    graph.add_node(
        "cache_lookup",
        make_cache_lookup_node(semantic_cache, embedding_service),
    )
    graph.add_node(
        "retrieve",
        make_retrieve_node(
            vector_store,
            bm25_index,
            vector_top_k=settings.vector_top_k,
            bm25_top_k=settings.bm25_top_k,
        ),
    )
    graph.add_node(
        "fuse_rerank",
        make_fuse_rerank_node(reranker, rerank_top_k=settings.rerank_top_k),
    )
    graph.add_node(
        "build_prompt",
        make_build_prompt_node(prompt_registry),
    )
    graph.add_node(
        "llm_generate",
        make_llm_generate_node(llm_service, prompt_registry),
    )
    graph.add_node(
        "output_guardrails",
        make_output_guardrails_node(output_guard, max_retries=settings.max_llm_retries),
    )
    graph.add_node(
        "cache_store",
        make_cache_store_node(semantic_cache),
    )

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_edge(START, "input_rails")

    graph.add_conditional_edges(
        "input_rails",
        _route_after_input_rails,
        {"cache_lookup": "cache_lookup", END: END},
    )

    graph.add_conditional_edges(
        "cache_lookup",
        _route_after_cache_lookup,
        {"retrieve": "retrieve", END: END},
    )

    # Linear chain: retrieve → fuse_rerank → build_prompt → llm_generate
    # (query_embedding is already in state from cache_lookup)
    graph.add_edge("retrieve", "fuse_rerank")
    graph.add_edge("fuse_rerank", "build_prompt")
    graph.add_edge("build_prompt", "llm_generate")
    graph.add_edge("llm_generate", "output_guardrails")

    graph.add_conditional_edges(
        "output_guardrails",
        _route_after_output_guardrails,
        {
            "cache_store": "cache_store",
            "llm_generate": "llm_generate",   # retry loop
            END: END,
        },
    )

    graph.add_edge("cache_store", END)

    return graph.compile()
