"""Deterministic, guardrail-compliant copy.

Used when no API key is configured (so the demo works offline), and as the safe fallback
whenever an LLM draft can't pass the guardrails.
"""

from __future__ import annotations

from retention_engine.crm.schemas import AudienceBrief, Incentive, LifecycleStage

REASON_OPENERS: dict[str, str] = {
    "LATE_DELIVERY": "A few of your recent orders took longer than they should have, and we're sorry.",
    "OVERDUE": "Your kitchen might be running low on the usual.",
    "FREQ_DECLINE": "We haven't seen you as often lately, and we'd love to help with the next restock.",
    "NARROW_BASKET": "Beyond your regulars, we deliver staples, household and personal-care essentials too.",
    "EARLY_TENURE": "Glad you tried us.",
    "DISCOUNT_DEPENDENT": "Your everyday essentials, at everyday prices.",
    "NONE": "Your regular essentials are a tap away.",
}

ADJACENT_CATEGORIES = [
    "staples",
    "household",
    "personal care",
    "dairy bread eggs",
    "fruits vegetables",
]


def offer_line(inc: Incentive, currency: str = "₹") -> str:
    if inc.type == "free_delivery":
        return "Free delivery on your next order."
    if inc.type == "percent_off":
        return f"Enjoy {inc.value:.0f}% off your next order (up to {currency}{inc.max_cap:.0f})."
    if inc.type == "flat_off":
        return f"We've added a {currency}{inc.value:.0f} credit to your next order."
    return ""


def _suggest_category(brief: AudienceBrief) -> str:
    for cat in ADJACENT_CATEGORIES:
        if cat not in brief.top_categories:
            return cat
    return "household"


def render(brief: AudienceBrief) -> dict[str, str]:
    brand = brief.brand_name
    offer = offer_line(brief.incentive, brief.currency_symbol)
    opener = REASON_OPENERS.get(brief.primary_reason, REASON_OPENERS["NONE"])
    stage = brief.stage

    if stage == LifecycleStage.ONBOARDING:
        copy = {
            "push_title": "Your second basket, sorted",
            "push_body": f"{{first_name}}, fruit, eggs, bread and more in about 10 minutes. {offer}",
            "whatsapp_message": (
                f"Hi {{first_name}}, thanks for your first {brand} order. Most people use us for "
                "daily top-ups: fruit, bread, eggs and a little something for later. "
                f"{offer} Your next basket is a tap away."
            ),
            "email_subject": f"{{first_name}}, here's what else {brand} delivers",
            "email_preheader": "From breakfast staples to cleaning supplies, in one quick order.",
            "cta": "Start my basket",
        }
    elif stage == LifecycleStage.HABIT_FORMATION:
        copy = {
            "push_title": "Make your weekly restock easy",
            "push_body": "{first_name}, save your usual basket and reorder it in two taps.",
            "whatsapp_message": (
                "Hi {first_name}, your usual {top_category} order could be a two-tap habit. "
                f"Save it as a favourite basket in the {brand} app and reorder whenever you run low."
            ),
            "email_subject": "Your usual basket, now in two taps",
            "email_preheader": "Save a favourite basket and restock in seconds.",
            "cta": "Save my basket",
        }
    elif stage == LifecycleStage.CROSS_SELL:
        cat = _suggest_category(brief)
        copy = {
            "push_title": f"Add {cat} to your next order"[:45],
            "push_body": f"{{first_name}}, {cat} arrive with your {{top_category}} in the same 10 minutes.",
            "whatsapp_message": (
                f"Hi {{first_name}}, did you know {brand} also delivers {cat}? Add a few to your "
                "next {top_category} order and skip the extra trip to the store."
            ),
            "email_subject": f"One basket for {cat} and more",
            "email_preheader": "Fewer trips, one quick delivery.",
            "cta": f"Shop {cat}"[:25],
        }
    elif stage == LifecycleStage.VIP_RETENTION:
        copy = {
            "push_title": "Thanks for being a regular",
            "push_body": f"{{first_name}}, a thank-you from us. {offer or 'Your favourites are ready.'}",
            "whatsapp_message": (
                f"Hi {{first_name}}, you're one of our most loyal customers. {opener} "
                f"{offer} Your favourites are saved and ready to reorder."
            ),
            "email_subject": "{first_name}, a thank-you from us",
            "email_preheader": "For being one of our regulars.",
            "cta": "Reorder favourites",
        }
    elif stage == LifecycleStage.WIN_BACK:
        copy = {
            "push_title": "It's been a while",
            "push_body": f"{{first_name}}, we'd love to deliver your next basket. {offer}",
            "whatsapp_message": (
                f"Hi {{first_name}}, it's been a while since your last {brand} order. {opener} "
                f"{offer} Fresh produce and daily essentials, at your door in minutes."
            ),
            "email_subject": "{first_name}, your favourites are still here",
            "email_preheader": "Daily essentials, delivered in minutes.",
            "cta": "Order now",
        }
    else:  # CHURN_MITIGATION
        copy = {
            "push_title": "Running low on the usual?",
            "push_body": f"{{first_name}}, restock in a few taps. {offer}",
            "whatsapp_message": (
                f"Hi {{first_name}}, {opener[0].lower() + opener[1:]} {offer} "
                "Your usual items are saved for a quick reorder."
            ),
            "email_subject": "{first_name}, your usual basket is ready",
            "email_preheader": "Restock your essentials in a few taps.",
            "cta": "Restock now",
        }

    copy = {k: " ".join(v.split()) for k, v in copy.items()}
    copy["rationale"] = (
        f"Template for {stage.label} / {brief.primary_reason}: lead with the main friction, "
        f"mention only the policy incentive ({brief.incentive.type})."
    )
    return copy
