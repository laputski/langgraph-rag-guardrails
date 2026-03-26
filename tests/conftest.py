"""
Shared pytest fixtures for unit and e2e tests.

For unit tests: services are mocked via pytest-mock.
For e2e tests: a real FastAPI app is used with a real httpx.AsyncClient.
  - Requires Docker services (Qdrant + Redis) to be running.
  - Requires knowledge base to be ingested (make ingest).
  - Requires OPENAI_API_KEY in the environment (or Ollama running).
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.graph.state import ScoredChunk


# ── Event loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── FastAPI app client ────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def app_client() -> AsyncIterator[AsyncClient]:
    """
    Session-scoped async HTTP client against the real FastAPI app.

    All services (Qdrant, Redis) must be up and the knowledge base must be
    ingested before e2e tests run.
    """
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# ── Sample chunks ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_chunks() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            doc_id="03_pto_policy",
            title="PTO Policy",
            chunk_index=0,
            text=(
                "Full-time employees receive 15 PTO days per year for 0-2 years of service, "
                "20 days for 3-5 years, 25 days for 6-10 years, and 30 days for 10+ years. "
                "PTO accrues monthly from the first day of employment."
            ),
            source_file="03_pto_policy.md",
            score=0.9,
        ),
        ScoredChunk(
            doc_id="12_parental_leave_policy",
            title="Parental Leave Policy",
            chunk_index=0,
            text=(
                "Primary caregivers receive 16 weeks of fully paid parental leave. "
                "Secondary caregivers receive 6 weeks of fully paid parental leave. "
                "Employees must have completed 6 months of continuous employment to be eligible."
            ),
            source_file="12_parental_leave_policy.md",
            score=0.85,
        ),
        ScoredChunk(
            doc_id="11_401k_retirement_plan",
            title="401k Retirement Plan",
            chunk_index=0,
            text=(
                "Acme Corp matches 100% of the first 4% of salary contributed to the 401k. "
                "This match vests immediately. The employee contribution limit for 2025 is $23,500."
            ),
            source_file="11_401k_retirement_plan.md",
            score=0.8,
        ),
    ]


# ── Mock LLM factories ────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_grounded(sample_chunks):
    """Returns an LLMService whose answer contains words from the context."""
    service = MagicMock()
    service.model_name = "mock-model"

    grounded_answer = (
        "Full-time employees receive 15 PTO days per year for the first 2 years of service, "
        "20 days for years 3–5, 25 days for years 6–10, and 30 days for 10+ years of service. "
        "[PTO Policy]"
    )

    async def fake_astream(messages):
        for word in grounded_answer.split():
            yield word + " "

    async def fake_ainvoke(messages):
        return grounded_answer

    service.astream = fake_astream
    service.ainvoke = AsyncMock(return_value=grounded_answer)
    return service


@pytest.fixture
def mock_llm_hallucinating():
    """Returns an LLMService whose answer has zero overlap with retrieved context."""
    service = MagicMock()
    service.model_name = "mock-model"

    hallucinated_answer = (
        "Employees receive unlimited vacation days and can take as much time off "
        "as they want with no restrictions whatsoever and zero approval needed."
    )

    async def fake_astream(messages):
        for word in hallucinated_answer.split():
            yield word + " "

    service.astream = fake_astream
    service.ainvoke = AsyncMock(return_value=hallucinated_answer)
    return service


@pytest.fixture
def mock_llm_first_hallucinate_then_ground(sample_chunks):
    """First call hallucinates; second call returns a grounded answer."""
    service = MagicMock()
    service.model_name = "mock-model"

    hallucinated = (
        "Employees receive unlimited vacation days and complete unrestricted freedom "
        "with absolutely no policies whatsoever anywhere ever."
    )
    grounded = (
        "Full-time employees receive 15 PTO days per year for the first 2 years of service. "
        "PTO accrues monthly from the first day of employment. [PTO Policy]"
    )
    call_count = {"n": 0}

    async def fake_astream(messages):
        answer = hallucinated if call_count["n"] == 0 else grounded
        call_count["n"] += 1
        for word in answer.split():
            yield word + " "

    service.astream = fake_astream
    service.ainvoke = AsyncMock(return_value=grounded)
    return service
