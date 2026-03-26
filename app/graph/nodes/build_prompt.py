from __future__ import annotations

from app.graph.state import RAGState
from app.services.prompt_registry import PromptRegistry


def make_build_prompt_node(prompt_registry: PromptRegistry):
    """
    Step 8 — Prompt construction from the Prompt Registry.

    Formats the reranked chunks into a context string and renders the
    versioned prompt template. The resulting messages list is stored in
    state for the LLM generation node.
    """
    def build_prompt_node(state: RAGState) -> dict:
        chunks = state["reranked_chunks"]

        # Build context string: each chunk prefixed with its document title
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[{chunk.title}]\n{chunk.text}"
            )
        context_text = "\n\n---\n\n".join(context_parts)

        # Render the prompt template
        version = state.get("prompt_version", "v2") or "v2"
        messages = prompt_registry.render(
            version=version,
            context=context_text,
            question=state["query"],
        )

        # Serialize messages to dicts for state storage (LangChain messages
        # are not JSON-serializable by default in all configurations)
        serialized = [
            {"type": m.__class__.__name__, "content": m.content}
            for m in messages
        ]

        return {
            "context_text": context_text,
            "constructed_prompt": serialized,
        }

    return build_prompt_node
