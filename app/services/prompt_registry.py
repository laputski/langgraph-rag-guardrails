from __future__ import annotations

import yaml
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage


class PromptRegistry:
    """
    Loads versioned prompt templates from a YAML file.

    YAML format (see prompts/templates.yaml):
        templates:
          v1:
            system: "..."
            user: "..."
          v2:
            system: "..."
            user: "..."
    """

    def __init__(self, path: str = "prompts/templates.yaml") -> None:
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        self._templates: dict[str, dict] = data["templates"]

    def render(
        self,
        version: str,
        context: str,
        question: str,
    ) -> list[BaseMessage]:
        """
        Return a list of LangChain messages ready for the LLM.

        Falls back to the first available version if the requested one
        is not found — prevents hard failures on misconfiguration.
        """
        tpl = self._templates.get(version)
        if tpl is None:
            version = next(iter(self._templates))
            tpl = self._templates[version]

        system_content = tpl["system"].strip()
        user_content = tpl["user"].format(context=context, question=question).strip()

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]

    def available_versions(self) -> list[str]:
        return list(self._templates.keys())
