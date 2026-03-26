"""Unit tests for the Prompt Registry."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest
import yaml

from app.services.prompt_registry import PromptRegistry
from langchain_core.messages import SystemMessage, HumanMessage


SAMPLE_YAML = textwrap.dedent("""
    templates:
      v1:
        description: "Basic"
        system: "You are an HR assistant."
        user: "Context:\\n{context}\\n\\nQuestion: {question}"
      v2:
        description: "Detailed with citations"
        system: "You are a helpful HR assistant. Cite sources in [Policy Name] format."
        user: "Relevant HR Policy Documents:\\n{context}\\n\\nEmployee Question: {question}"
""").strip()


@pytest.fixture
def registry(tmp_path) -> PromptRegistry:
    """Create a PromptRegistry backed by a real temp YAML file."""
    template_file = tmp_path / "templates.yaml"
    template_file.write_text(SAMPLE_YAML, encoding="utf-8")
    return PromptRegistry(str(template_file))


class TestPromptRegistry:
    def test_available_versions(self, registry):
        versions = registry.available_versions()
        assert "v1" in versions
        assert "v2" in versions

    def test_render_v1_returns_two_messages(self, registry):
        messages = registry.render("v1", context="Some context", question="What is PTO?")
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    def test_render_v2_injects_context_and_question(self, registry):
        context = "You get 15 PTO days."
        question = "How many PTO days do I get?"
        messages = registry.render("v2", context=context, question=question)
        user_content = messages[1].content
        assert context in user_content
        assert question in user_content

    def test_render_v1_system_matches_yaml(self, registry):
        messages = registry.render("v1", context="ctx", question="q")
        assert messages[0].content == "You are an HR assistant."

    def test_render_unknown_version_falls_back(self, registry):
        """Unknown version should fall back to the first available template."""
        messages = registry.render("v99", context="ctx", question="q")
        assert len(messages) == 2  # no error, graceful fallback

    def test_render_v2_system_mentions_citation(self, registry):
        messages = registry.render("v2", context="ctx", question="q")
        assert "[Policy Name]" in messages[0].content
