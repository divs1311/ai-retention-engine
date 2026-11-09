"""Prompt construction. Versioned so cached generations invalidate when prompts change."""

from __future__ import annotations

import json

from retention_engine.crm.guardrails import ALLOWED_TOKENS, LIMITS
from retention_engine.crm.schemas import AudienceBrief, LifecycleStage

PROMPT_VERSION = "2026-11-v1"

STAGE_GOALS: dict[LifecycleStage, str] = {
    LifecycleStage.ONBOARDING: "Get a new user to their 2nd–3rd order and show the breadth of "
    "what can be delivered in minutes.",
    LifecycleStage.HABIT_FORMATION: "Anchor ordering to a recurring moment (breakfast restock, "
    "weekend cooking) to build a weekly habit.",
    LifecycleStage.CROSS_SELL: "Expand basket breadth into one adjacent category the user "
    "hasn't tried. Broader baskets retain ~2x better.",
    LifecycleStage.CHURN_MITIGATION: "Re-engage a slipping user before they lapse, addressing "
    "the specific friction behind the slowdown.",
    LifecycleStage.VIP_RETENTION: "Protect a high-value customer showing early risk; make them "
    "feel recognised. Service recovery over discounts.",
    LifecycleStage.WIN_BACK: "Bring back a lapsed user with a clear, honest reason to return.",
}

SYSTEM_PROMPT = """You are a senior lifecycle CRM writer for {brand}, a 10-minute grocery \
delivery app in India. You write concise, warm, specific copy that respects the customer.

Hard rules:
- Return ONLY a JSON object with exactly these string keys: {keys}.
- Character limits: {limits}.
- Mention ONLY the incentive given in the brief. If it says no incentive, do not mention \
discounts, coupons, cashback, free delivery or any offer.
- No manipulative urgency or spam language (e.g. "act now", "last chance", "guaranteed", \
"you won"), no ALL-CAPS words, at most 2 emoji per field, no "!!!".
- Personalisation tokens allowed: {tokens}. Do not invent any other {{token}}.
- Never claim facts not in the brief (no fake stock levels, prices or deadlines).
- "rationale" is internal: one or two sentences on why this angle fits the audience."""

USER_PROMPT = """Write a {stage_label} campaign for this audience.

Goal: {goal}

Audience brief (JSON):
{brief_json}

Incentive you may mention: {incentive}
{feedback}"""


def build_messages(brief: AudienceBrief, feedback: list[str] | None = None) -> tuple[str, str]:
    system = SYSTEM_PROMPT.format(
        brand=brief.brand_name,
        keys=", ".join(LIMITS) + ", rationale",
        limits="; ".join(f"{k} ≤ {v}" for k, v in LIMITS.items()),
        tokens=", ".join("{" + t + "}" for t in sorted(ALLOWED_TOKENS)),
    )
    brief_view = brief.model_dump(
        mode="json",
        include={
            "stage",
            "primary_reason",
            "reason_summary",
            "size",
            "avg_churn_probability",
            "avg_order_value",
            "avg_category_width",
            "top_categories",
            "dominant_rfm_segments",
            "currency_symbol",
        },
    )
    fb = ""
    if feedback:
        fb = (
            "\nYour previous draft was rejected by automated checks. Fix ALL of these:\n- "
            + "\n- ".join(feedback)
        )
    user = USER_PROMPT.format(
        stage_label=brief.stage.label,
        goal=STAGE_GOALS[brief.stage],
        brief_json=json.dumps(brief_view, indent=2, ensure_ascii=False),
        incentive=brief.incentive.describe(brief.currency_symbol),
        feedback=fb,
    )
    return system, user
