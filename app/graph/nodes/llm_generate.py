from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from app.graph.state import RAGState
from app.services.llm import LLMService
from app.services.prompt_registry import PromptRegistry


def make_llm_generate_node(
    llm_service: LLMService,
    prompt_registry: PromptRegistry,
):
    """
    Step 9 — LLM inference.

    Generates an answer using the constructed prompt. On retry (called again
    after a failed guardrail check), increments llm_retry_count and regenerates.

    Note: In a full streaming implementation the tokens would be yielded directly
    to the SSE stream. Here we collect the full answer in state so that
    output_guardrails can inspect it before delivery. The streaming happens in
    the API layer (routes.py) after the graph completes successfully.
    """
    async def llm_generate_node(state: RAGState) -> dict:
        retry_count = state.get("llm_retry_count", 0)

        # Rebuild messages from context + query (handles retry without
        # needing to re-run retrieval/reranking)
        version = state.get("prompt_version", "v2") or "v2"
        messages = prompt_registry.render(
            version=version,
            context=state["context_text"] or "",
            question=state["query"],
        )

        # Collect the full streamed response
        tokens = []
        async for token in llm_service.astream(messages):
            tokens.append(token)
        answer = "".join(tokens)

        return {
            "llm_answer": answer,
            "llm_retry_count": retry_count + 1,
            "llm_model_used": llm_service.model_name,
        }

    return llm_generate_node
