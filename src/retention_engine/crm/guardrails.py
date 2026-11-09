"""Deterministic checks every message must pass before it can be sent.

LLM output is treated as untrusted: a draft that violates a rule is sent back with the
violations as feedback, and after ``max_retries`` the generator falls back to a safe
template. Warnings are surfaced in the UI but don't block.
"""

from __future__ import annotations

import re

from retention_engine.crm.schemas import AudienceBrief, CampaignCopy

LIMITS: dict[str, int] = {
    "push_title": 45,
    "push_body": 120,
    "whatsapp_message": 400,
    "email_subject": 60,
    "email_preheader": 100,
    "cta": 25,
}

ALLOWED_TOKENS = {"first_name", "top_category"}

BANNED_PHRASES = [
    "act now",
    "last chance",
    "guaranteed",
    "risk-free",
    "risk free",
    "you won",
    "you've won",
    "click here",
    "limited time only",
    "100% free",
    "urgent",
    "hurry",
    "don't miss out",
    "final warning",
]

_OFFER_PATTERNS = {
    "percent": re.compile(r"(\d{1,2})\s?%\s?(off|discount)", re.I),
    "flat": re.compile(r"(₹|rs\.?|inr)\s?(\d+)\s?(off|credit)", re.I),
    "free_delivery": re.compile(r"free\s+deliver(y|ies)", re.I),
    "generic": re.compile(r"\b(discount|coupon|cashback|promo code|voucher)\b", re.I),
}
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff\U0001f900-\U0001f9ff]"
)
_TOKEN = re.compile(r"\{([a-zA-Z_]+)\}")
_CAPS_WORD = re.compile(r"\b[A-Z]{4,}\b")
_CAPS_ALLOWLIST = {"UPI", "COD", "OTP", "ASAP"}


def check(copy: CampaignCopy, brief: AudienceBrief) -> tuple[list[str], list[str]]:
    """Return (violations, warnings)."""
    violations: list[str] = []
    warnings: list[str] = []
    fields = {k: getattr(copy, k) for k in LIMITS}

    for name, text in fields.items():
        if not text or not text.strip():
            violations.append(f"{name} is empty")
            continue
        if len(text) > LIMITS[name]:
            violations.append(f"{name} is {len(text)} chars (limit {LIMITS[name]})")
        lowered = text.lower()
        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                violations.append(f"{name} uses banned phrase '{phrase}'")
        if "!!" in text:
            violations.append(f"{name} uses repeated exclamation marks")
        caps = [w for w in _CAPS_WORD.findall(text) if w not in _CAPS_ALLOWLIST]
        if caps and brief.brand_name.upper() not in caps:
            violations.append(f"{name} has ALL-CAPS words: {', '.join(caps[:3])}")
        if len(_EMOJI.findall(text)) > 2:
            violations.append(f"{name} has more than 2 emoji")
        unknown = set(_TOKEN.findall(text)) - ALLOWED_TOKENS
        if unknown:
            violations.append(f"{name} uses unknown tokens: {', '.join(sorted(unknown))}")

    violations += _offer_violations(" ".join(fields.values()), brief)

    if not copy.rationale.strip():
        warnings.append("rationale is empty")
    if "{first_name}" not in " ".join(fields.values()):
        warnings.append("no {first_name} personalisation")
    return violations, warnings


def _offer_violations(text: str, brief: AudienceBrief) -> list[str]:
    inc = brief.incentive
    out: list[str] = []
    pct = [int(m.group(1)) for m in _OFFER_PATTERNS["percent"].finditer(text)]
    flat = [int(m.group(2)) for m in _OFFER_PATTERNS["flat"].finditer(text)]
    free_del = bool(_OFFER_PATTERNS["free_delivery"].search(text))
    generic = bool(_OFFER_PATTERNS["generic"].search(text))

    if inc.type == "none" and (pct or flat or free_del or generic):
        out.append("mentions an offer, but this audience has no incentive")
    if inc.type == "free_delivery" and (pct or flat):
        out.append("mentions money off, but the only allowed offer is free delivery")
    if inc.type == "percent_off":
        if any(p != int(inc.value) for p in pct):
            out.append(f"percentage off must be exactly {inc.value:.0f}%")
        if flat or free_del:
            out.append("mentions an offer other than the allowed percentage")
    if inc.type == "flat_off":
        if any(v != int(inc.value) for v in flat):
            out.append(f"credit amount must be exactly {brief.currency_symbol}{inc.value:.0f}")
        if pct or free_del:
            out.append("mentions an offer other than the allowed credit")
    return out
