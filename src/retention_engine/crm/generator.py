"""Campaign generation loop: draft → validate → (retry with feedback) → fallback → cache."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from pydantic import ValidationError

from retention_engine.config import LLMConfig
from retention_engine.crm.guardrails import check
from retention_engine.crm.prompts import PROMPT_VERSION
from retention_engine.crm.providers import BaseProvider, TemplateProvider, get_provider
from retention_engine.crm.schemas import AudienceBrief, Campaign, CampaignCopy, GuardrailReport
from retention_engine.logging_utils import get_logger

log = get_logger(__name__)


class CampaignGenerator:
    def __init__(
        self,
        provider: BaseProvider | None = None,
        cfg: LLMConfig | None = None,
        cache_dir: Path | None = None,
    ):
        self.cfg = cfg or LLMConfig()
        self.provider = provider or get_provider("auto", self.cfg)
        self.cache_dir = cache_dir if self.cfg.cache else None
        self._fallback = TemplateProvider()

    # -- caching ------------------------------------------------------------------------
    def _cache_key(self, brief: AudienceBrief) -> str:
        payload = json.dumps(
            [
                self.provider.name,
                self.provider.model,
                PROMPT_VERSION,
                brief.model_dump(mode="json"),
            ],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def _cache_get(self, key: str) -> Campaign | None:
        if not self.cache_dir or self.provider.name == "template":
            return None
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            c = Campaign.model_validate_json(path.read_text())
            return c.model_copy(update={"cached": True, "latency_ms": 0})
        return None

    def _cache_put(self, key: str, campaign: Campaign) -> None:
        if not self.cache_dir or campaign.provider == "template":
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(campaign.model_dump_json(indent=2))

    # -- generation ---------------------------------------------------------------------
    def generate(self, brief: AudienceBrief) -> Campaign:
        key = self._cache_key(brief)
        if cached := self._cache_get(key):
            return cached

        t0 = time.perf_counter()
        feedback: list[str] | None = None
        fallback_reason: str | None = None
        attempts = 0
        copy: CampaignCopy | None = None
        warnings: list[str] = []

        for _ in range(1 + self.cfg.max_retries):
            attempts += 1
            try:
                draft = CampaignCopy.model_validate(self.provider.generate(brief, feedback))
            except ValidationError as exc:
                feedback = [f"output did not match the schema: {exc.errors()[0]['msg']}"]
                fallback_reason = "schema validation failed"
                continue
            except Exception as exc:
                log.warning("Provider %s failed: %s", self.provider.name, exc)
                fallback_reason = f"provider error: {type(exc).__name__}: {exc}"[:300]
                break
            violations, warnings = check(draft, brief)
            if not violations:
                copy, fallback_reason = draft, None
                break
            feedback = violations
            fallback_reason = f"guardrail violations: {'; '.join(violations[:3])}"

        provider_name, model = self.provider.name, self.provider.model
        if copy is None:
            copy = CampaignCopy.model_validate(self._fallback.generate(brief))
            provider_name, model = "template", None
        violations, warnings = check(copy, brief)

        campaign = Campaign(
            brief=brief,
            content=copy,
            provider=provider_name,
            model=model,
            guardrails=GuardrailReport(
                passed=not violations, violations=violations, warnings=warnings
            ),
            attempts=attempts,
            fallback_reason=fallback_reason
            if provider_name == "template" and self.provider.name != "template"
            else None,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        if campaign.guardrails.passed:
            self._cache_put(key, campaign)
        return campaign

    def generate_all(self, briefs: list[AudienceBrief]) -> list[Campaign]:
        return [self.generate(b) for b in briefs]
