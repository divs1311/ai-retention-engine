from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from retention_engine.config import LLMConfig
from retention_engine.crm import templates
from retention_engine.crm.generator import CampaignGenerator
from retention_engine.crm.guardrails import check
from retention_engine.crm.playbook import assign_stage, build_audiences, incentive_policy
from retention_engine.crm.prompts import build_messages
from retention_engine.crm.providers import (
    AnthropicProvider,
    BaseProvider,
    OpenAIProvider,
    extract_json,
    get_provider,
)
from retention_engine.crm.schemas import AudienceBrief, CampaignCopy, Incentive, LifecycleStage

REASONS = [
    "LATE_DELIVERY",
    "OVERDUE",
    "FREQ_DECLINE",
    "NARROW_BASKET",
    "EARLY_TENURE",
    "DISCOUNT_DEPENDENT",
    "NONE",
]
INCENTIVES = [
    Incentive(),
    Incentive(type="free_delivery"),
    Incentive(type="percent_off", value=15, max_cap=100),
    Incentive(type="flat_off", value=75),
]


def brief(**kw) -> AudienceBrief:
    base = dict(
        audience_id="churn_mitigation__overdue",
        stage=LifecycleStage.CHURN_MITIGATION,
        primary_reason="OVERDUE",
        reason_summary="21 days since last order vs a usual 7-day cadence",
        size=120,
        avg_churn_probability=0.62,
        value_at_risk=55000,
        avg_order_value=450,
        avg_category_width=2.4,
        top_categories=["fruits vegetables", "dairy bread eggs"],
        dominant_rfm_segments=["At Risk"],
        incentive=Incentive(type="free_delivery"),
    )
    base.update(kw)
    return AudienceBrief(**base)


GOOD = {
    "push_title": "Running low on the usual?",
    "push_body": "{first_name}, restock in a few taps. Free delivery on your next order.",
    "whatsapp_message": "Hi {first_name}, your usual basket is saved. Free delivery on your next order.",
    "email_subject": "{first_name}, your usual basket is ready",
    "email_preheader": "Restock in a few taps.",
    "cta": "Restock now",
    "rationale": "Overdue users respond to convenience.",
}


# --- playbook ----------------------------------------------------------------------------
def _row(**kw):
    base = dict(
        risk_tier="high",
        tenure_days=200,
        rfm_segment="At Risk",
        recency_days=10,
        avg_category_width_90d=4,
    )
    base.update(kw)
    return pd.Series(base)


@pytest.mark.parametrize(
    ("row", "stage"),
    [
        (_row(tenure_days=20), LifecycleStage.ONBOARDING),
        (_row(rfm_segment="Champions"), LifecycleStage.VIP_RETENTION),
        (_row(recency_days=25), LifecycleStage.WIN_BACK),
        (_row(), LifecycleStage.CHURN_MITIGATION),
        (_row(risk_tier="low", avg_category_width_90d=2), LifecycleStage.CROSS_SELL),
        (_row(risk_tier="low"), LifecycleStage.HABIT_FORMATION),
    ],
)
def test_assign_stage(row, stage):
    assert assign_stage(row) == stage


def test_incentive_policy_never_discounts_low_risk_stages():
    for reason in REASONS:
        for stage in (LifecycleStage.HABIT_FORMATION, LifecycleStage.CROSS_SELL):
            assert incentive_policy(stage, reason).type == "none"
    assert incentive_policy(LifecycleStage.VIP_RETENTION, "LATE_DELIVERY").type == "flat_off"
    assert incentive_policy(LifecycleStage.WIN_BACK, "DISCOUNT_DEPENDENT").type == "free_delivery"


def test_build_audiences(trained, dataset, settings):
    from retention_engine.churn.model import score

    users, orders = dataset
    scores = score(trained.pipeline, orders, users, settings.churn)
    briefs = build_audiences(scores, orders, min_size=5)
    assert briefs
    assert [b.value_at_risk for b in briefs] == sorted(
        (b.value_at_risk for b in briefs), reverse=True
    )
    assert sum(b.size for b in briefs) <= len(scores)
    assert all(b.top_categories for b in briefs)


# --- guardrails --------------------------------------------------------------------------
def test_good_copy_passes():
    v, w = check(CampaignCopy(**GOOD), brief())
    assert v == [] and w == []


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"push_title": "x" * 60}, "push_title is 60 chars"),
        ({"push_body": "Act now before it's gone"}, "banned phrase 'act now'"),
        ({"email_subject": "HUGE savings inside"}, "ALL-CAPS"),
        ({"cta": "Go 🎉🎉🎉"}, "more than 2 emoji"),
        ({"push_body": "Hi {last_name}"}, "unknown tokens: last_name"),
        ({"push_body": "Great!! deals"}, "repeated exclamation"),
        ({"cta": ""}, "cta is empty"),
        ({"whatsapp_message": "Get 20% off today"}, "only allowed offer is free delivery"),
    ],
)
def test_guardrail_violations(patch, expected):
    v, _ = check(CampaignCopy(**{**GOOD, **patch}), brief())
    assert any(expected in x for x in v), v


def test_offer_must_match_policy_exactly():
    pct = brief(incentive=Incentive(type="percent_off", value=15, max_cap=100))
    ok = {
        **GOOD,
        "push_body": "{first_name}, enjoy 15% off your next order.",
        "whatsapp_message": "Hi {first_name}, 15% off.",
    }
    assert check(CampaignCopy(**ok), pct)[0] == []
    wrong = {**ok, "push_body": "{first_name}, enjoy 25% off."}
    assert any("exactly 15%" in v for v in check(CampaignCopy(**wrong), pct)[0])
    none = brief(incentive=Incentive())
    assert any("no incentive" in v for v in check(CampaignCopy(**GOOD), none)[0])


@pytest.mark.parametrize(
    ("stage", "reason", "inc"), itertools.product(LifecycleStage, REASONS, INCENTIVES)
)
def test_every_template_passes_guardrails(stage, reason, inc):
    b = brief(stage=stage, primary_reason=reason, incentive=inc)
    v, _ = check(CampaignCopy(**templates.render(b)), b)
    assert v == []


# --- prompts & providers -----------------------------------------------------------------
def test_prompt_contains_policy_and_feedback():
    system, user = build_messages(brief(), feedback=["push_title is 60 chars (limit 45)"])
    assert "push_title ≤ 45" in system
    assert "Free delivery on the next order." in user
    assert "push_title is 60 chars" in user
    assert "value_at_risk" not in user  # internal economics are not sent to the model


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(GOOD),
        f"```json\n{json.dumps(GOOD)}\n```",
        f"Sure! Here you go:\n{json.dumps(GOOD)}\nThanks",
    ],
)
def test_extract_json(text):
    assert extract_json(text)["cta"] == "Restock now"


def test_openai_provider_with_fake_client():
    calls = {}

    def create(**kw):
        calls.update(kw)
        msg = SimpleNamespace(content=json.dumps(GOOD))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    out = OpenAIProvider(LLMConfig(openai_model="test-model"), client=client).generate(brief())
    assert out["cta"] == "Restock now"
    assert calls["model"] == "test-model"
    assert calls["response_format"] == {"type": "json_object"}


def test_anthropic_provider_with_fake_client():
    calls = {}

    def create(**kw):
        calls.update(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(GOOD))])

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    out = AnthropicProvider(LLMConfig(anthropic_model="claude-test"), client=client).generate(
        brief()
    )
    assert out["push_title"] == GOOD["push_title"]
    assert calls["model"] == "claude-test" and "JSON" in calls["system"]


def test_auto_provider_selection(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert get_provider("auto").name == "template"


# --- generator loop ----------------------------------------------------------------------
class Scripted(BaseProvider):
    name, model = "scripted", "m1"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.feedback_seen: list = []

    def generate(self, brief, feedback=None):
        self.feedback_seen.append(feedback)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_retry_with_feedback_then_pass():
    p = Scripted([{**GOOD, "push_body": "ACT NOW!!"}, GOOD])
    c = CampaignGenerator(p, LLMConfig(max_retries=1, cache=False)).generate(brief())
    assert c.provider == "scripted" and c.attempts == 2 and c.guardrails.passed
    assert p.feedback_seen[0] is None and p.feedback_seen[1]


def test_falls_back_to_template_after_repeated_violations():
    p = Scripted([{**GOOD, "cta": "x" * 99}] * 2)
    c = CampaignGenerator(p, LLMConfig(max_retries=1, cache=False)).generate(brief())
    assert c.provider == "template" and c.guardrails.passed
    assert "guardrail" in c.fallback_reason


def test_falls_back_on_provider_error():
    p = Scripted([TimeoutError("boom")])
    c = CampaignGenerator(p, LLMConfig(cache=False)).generate(brief())
    assert c.provider == "template" and "TimeoutError" in c.fallback_reason


def test_schema_errors_are_retried():
    p = Scripted([{"push_title": "only one field"}, GOOD])
    c = CampaignGenerator(p, LLMConfig(max_retries=1, cache=False)).generate(brief())
    assert c.provider == "scripted" and c.attempts == 2


def test_cache_hit(tmp_path):
    p = Scripted([GOOD])
    gen = CampaignGenerator(p, LLMConfig(cache=True), cache_dir=tmp_path)
    first = gen.generate(brief())
    second = gen.generate(brief())  # would raise IndexError if the provider were called
    assert not first.cached and second.cached
    assert second.content == first.content
