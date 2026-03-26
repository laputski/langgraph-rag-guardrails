from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel


def build_llm(
    provider: str,
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.2",
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
) -> BaseChatModel:
    """
    Factory that returns a LangChain chat model.

    Supported providers:
      - "ollama"  → ChatOllama  (default; free local models, no API key needed)
                    Recommended free models: llama3.2, mistral, gemma2:2b
                    Pull with: ollama pull llama3.2
      - "openai"  → ChatOpenAI  (requires OPENAI_API_KEY in .env)
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=openai_api_key,
            model=openai_model,
            temperature=0.0,
            streaming=True,
        )
    else:
        # Default: Ollama — local, free, sovereign
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.0,
        )


class LLMService:
    """Thin wrapper providing a streaming interface over any LangChain chat model."""

    def __init__(self, model: BaseChatModel, model_name: str) -> None:
        self._model = model
        self.model_name = model_name

    async def astream(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        """Yield token strings as they are generated."""
        async for chunk in self._model.astream(messages):
            if chunk.content:
                yield chunk.content

    async def ainvoke(self, messages: list[BaseMessage]) -> str:
        """Invoke the model and return the full response string."""
        response = await self._model.ainvoke(messages)
        return str(response.content)
