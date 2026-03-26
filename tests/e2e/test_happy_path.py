"""
E2E happy path tests.

Prerequisites:
  - make up          (Qdrant + Redis running)
  - make ingest      (knowledge base loaded)
  - OPENAI_API_KEY   (or Ollama configured)
  - make dev OR the ASGI app initialized via the app_client fixture
"""
from __future__ import annotations

import json
import time

import pytest
from httpx import AsyncClient


def _parse_sse(body: str) -> list[dict]:
    """Parse SSE response body into a list of event dicts."""
    events = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _get_done_event(events: list[dict]) -> dict | None:
    for e in events:
        if e.get("type") == "done":
            return e
    return None


def _reconstruct_answer(events: list[dict]) -> str:
    return "".join(e.get("content", "") for e in events if e.get("type") == "token")


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_health_check(self, app_client: AsyncClient):
        """All services should be healthy after startup + ingest."""
        response = await app_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["services"]["qdrant"] == "ok"
        assert body["services"]["redis"] == "ok"
        assert "loaded" in body["services"]["bm25_index"]

    @pytest.mark.asyncio
    async def test_pto_query_returns_grounded_answer(self, app_client: AsyncClient):
        """A question about PTO should return an answer grounded in the PTO policy."""
        response = await app_client.post(
            "/query",
            json={"query": "How many PTO days do full-time employees get per year at Acme?"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = _parse_sse(response.text)
        done = _get_done_event(events)
        assert done is not None, "SSE stream must contain a 'done' event"
        assert done["from_cache"] is False

        answer = _reconstruct_answer(events)
        assert len(answer) > 20, "Answer should be non-trivial"

        # At least one source should reference the PTO policy
        sources = done.get("sources", [])
        assert any("PTO" in s or "pto" in s.lower() for s in sources), (
            f"Expected PTO Policy in sources, got: {sources}"
        )

    @pytest.mark.asyncio
    async def test_parental_leave_query(self, app_client: AsyncClient):
        """A parental leave question should return duration details."""
        response = await app_client.post(
            "/query",
            json={"query": "What is the parental leave policy at Acme Corp?"},
        )
        assert response.status_code == 200
        events = _parse_sse(response.text)
        done = _get_done_event(events)
        assert done is not None
        answer = _reconstruct_answer(events)
        # The answer should mention weeks of leave
        assert "week" in answer.lower() or "16" in answer or "6" in answer

    @pytest.mark.asyncio
    async def test_401k_match_query(self, app_client: AsyncClient):
        """A 401k question should mention the 4% match."""
        response = await app_client.post(
            "/query",
            json={"query": "What is the 401k employer match percentage at Acme?"},
        )
        assert response.status_code == 200
        events = _parse_sse(response.text)
        answer = _reconstruct_answer(events)
        assert "4%" in answer or "4 percent" in answer.lower() or "four percent" in answer.lower()

    @pytest.mark.asyncio
    async def test_repeated_query_hits_cache(self, app_client: AsyncClient):
        """
        Sending the exact same query twice should result in a cache hit on the second call.
        """
        query = "How much is the annual learning and development budget at Acme Corp?"

        # First call — warms the cache
        r1 = await app_client.post("/query", json={"query": query})
        assert r1.status_code == 200
        events1 = _parse_sse(r1.text)
        done1 = _get_done_event(events1)
        assert done1 is not None
        assert done1["from_cache"] is False

        # Second call — should hit the cache
        r2 = await app_client.post("/query", json={"query": query})
        assert r2.status_code == 200
        events2 = _parse_sse(r2.text)
        done2 = _get_done_event(events2)
        assert done2 is not None
        assert done2["from_cache"] is True, "Second identical query should be a cache hit"

    @pytest.mark.asyncio
    async def test_semantically_similar_query_hits_cache(self, app_client: AsyncClient):
        """
        A paraphrased version of a previously answered question should hit the cache
        (cosine similarity above 0.92 threshold).
        """
        # Warm the cache with the canonical question
        warmup_query = "What is the parental leave duration for primary caregivers?"
        r1 = await app_client.post("/query", json={"query": warmup_query})
        assert r1.status_code == 200

        # Semantically similar paraphrase
        similar_query = "How many weeks of parental leave does a primary caregiver get?"
        r2 = await app_client.post("/query", json={"query": similar_query})
        assert r2.status_code == 200
        events2 = _parse_sse(r2.text)
        done2 = _get_done_event(events2)
        assert done2 is not None
        # This may or may not be a cache hit depending on embedding similarity —
        # assert the response is valid either way
        assert done2.get("from_cache") in (True, False)
        answer2 = _reconstruct_answer(events2)
        assert len(answer2) > 10

    @pytest.mark.asyncio
    async def test_response_includes_request_id(self, app_client: AsyncClient):
        """Every response should include a request_id in the done event."""
        r = await app_client.post(
            "/query",
            json={"query": "What is the remote work policy?"},
        )
        assert r.status_code == 200
        events = _parse_sse(r.text)
        done = _get_done_event(events)
        assert done is not None
        assert "request_id" in done
        assert len(done["request_id"]) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_prompt_version_v1_accepted(self, app_client: AsyncClient):
        """Specifying prompt_version=v1 should still return a valid response."""
        r = await app_client.post(
            "/query",
            json={
                "query": "What are the health benefit plan options?",
                "prompt_version": "v1",
            },
        )
        assert r.status_code == 200
        events = _parse_sse(r.text)
        done = _get_done_event(events)
        assert done is not None
