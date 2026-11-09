"""LLM provider abstraction.

``get_provider("auto")`` picks Anthropic if ``ANTHROPIC_API_KEY`` is set, else OpenAI if
``OPENAI_API_KEY`` is set, else the offline template provider. SDKs are imported lazily so
they're optional dependencies (``pip install retention-engine[llm]``).
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from retention_engine.config import LLMConfig
from retention_engine.crm import templates
from retention_engine.crm.prompts import build_messages
from retention_engine.crm.schemas import AudienceBrief


class ProviderError(RuntimeError):
    pass


class BaseProvider(ABC):
    name: str = "base"
    model: str | None = None

    @abstractmethod
    def generate(self, brief: AudienceBrief, feedback: list[str] | None = None) -> dict[str, Any]:
        """Return a dict matching :class:`CampaignCopy`."""


class TemplateProvider(BaseProvider):
    name = "template"

    def generate(self, brief: AudienceBrief, feedback: list[str] | None = None) -> dict[str, Any]:
        return templates.render(brief)


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences or stray prose."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ProviderError(f"No JSON object in model output: {text[:200]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Invalid JSON from model: {exc}") from exc
    if not isinstance(data, dict):
        raise ProviderError("Model output JSON is not an object")
    return data


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, cfg: LLMConfig, client: Any = None):
        self.cfg = cfg
        self.model = cfg.openai_model
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise ProviderError("pip install 'retention-engine[llm]' to use OpenAI") from exc
            client = OpenAI(timeout=cfg.timeout_s)
        self.client = client

    def generate(self, brief: AudienceBrief, feedback: list[str] | None = None) -> dict[str, Any]:
        system, user = build_messages(brief, feedback)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_completion_tokens=self.cfg.max_output_tokens,
        )
        return extract_json(resp.choices[0].message.content or "")


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, cfg: LLMConfig, client: Any = None):
        self.cfg = cfg
        self.model = cfg.anthropic_model
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover
                raise ProviderError("pip install 'retention-engine[llm]' to use Anthropic") from exc
            client = Anthropic(timeout=cfg.timeout_s)
        self.client = client

    def generate(self, brief: AudienceBrief, feedback: list[str] | None = None) -> dict[str, Any]:
        system, user = build_messages(brief, feedback)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.cfg.max_output_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        return extract_json(text)


def available_providers() -> list[str]:
    out = ["template"]
    if os.getenv("OPENAI_API_KEY"):
        out.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        out.append("anthropic")
    return out


def get_provider(name: str = "auto", cfg: LLMConfig | None = None) -> BaseProvider:
    cfg = cfg or LLMConfig()
    if name == "auto":
        name = (
            cfg.provider
            if cfg.provider != "auto"
            else (
                "anthropic"
                if os.getenv("ANTHROPIC_API_KEY")
                else "openai"
                if os.getenv("OPENAI_API_KEY")
                else "template"
            )
        )
    if name == "anthropic":
        return AnthropicProvider(cfg)
    if name == "openai":
        return OpenAIProvider(cfg)
    if name == "template":
        return TemplateProvider()
    raise ValueError(f"Unknown provider {name!r}")
