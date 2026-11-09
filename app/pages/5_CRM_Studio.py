"""Generate, inspect and export lifecycle campaigns per audience."""

from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

from retention_engine.crm.generator import CampaignGenerator
from retention_engine.crm.providers import available_providers, get_provider
from retention_engine.crm.schemas import AudienceBrief
from retention_engine.pipeline import audiences

from _shared import load_artifacts, money, page_setup, settings

page_setup("CRM studio", "✉️")
art = load_artifacts()


@st.cache_data(show_spinner=False)
def _briefs_json() -> list[dict]:
    return [b.model_dump(mode="json") for b in audiences(art.scores, art.orders, settings)]


briefs = [AudienceBrief.model_validate(b) for b in _briefs_json()]

st.title("CRM studio")
st.markdown(
    '<p class="re-caption">Each audience is a lifecycle stage × primary churn reason. Incentives are '
    "set by policy; the LLM only writes copy, and every draft must pass deterministic guardrails "
    "or it's regenerated with feedback, then replaced by a safe template.</p>",
    unsafe_allow_html=True,
)

table = pd.DataFrame(
    [
        {
            "audience": b.audience_id,
            "stage": b.stage.label,
            "reason": b.primary_reason.replace("_", " ").title(),
            "users": b.size,
            "value_at_risk": b.value_at_risk,
            "avg_churn_p": b.avg_churn_probability,
            "incentive": b.incentive.describe(settings.currency_symbol),
        }
        for b in briefs
    ]
)
st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "value_at_risk": st.column_config.NumberColumn("Value at risk", format="%.0f"),
        "avg_churn_p": st.column_config.ProgressColumn(
            "Avg churn p", format="%.2f", min_value=0, max_value=1
        ),
    },
)

provs = available_providers()
c1, c2 = st.columns([3, 1])
choice = c1.selectbox(
    "Audience",
    range(len(briefs)),
    format_func=lambda i: (
        f"{briefs[i].stage.label} · {briefs[i].primary_reason.replace('_', ' ').title()} "
        f"({briefs[i].size} users, {money(briefs[i].value_at_risk)} at risk)"
    ),
)
provider = c2.selectbox(
    "Provider", provs[::-1], help="Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable LLM providers."
)
if provs == ["template"]:
    st.caption(
        "No API key detected — using the offline template provider. Add a key to `.env` or Streamlit secrets to use Claude or GPT."
    )

brief = briefs[choice]
with st.expander("Audience brief sent to the model"):
    st.json(brief.model_dump(mode="json"))

if st.button("Generate campaign", type="primary"):
    gen = CampaignGenerator(
        get_provider(provider, settings.llm), settings.llm, cache_dir=settings.llm_cache_dir
    )
    with st.spinner(f"Generating with {provider}…"):
        st.session_state["campaign"] = gen.generate(brief)

camp = st.session_state.get("campaign")
if camp and camp.brief.audience_id == brief.audience_id:
    cp = camp.content
    esc = lambda s: html.escape(s).replace("{", "<b>{").replace("}", "}</b>")  # noqa: E731
    p, w, e = st.columns(3)
    with p:
        st.markdown("**Push notification**")
        st.markdown(
            f'<div class="re-card"><b>{esc(cp.push_title)}</b><br>{esc(cp.push_body)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**CTA:** {html.escape(cp.cta)}")
    with w:
        st.markdown("**WhatsApp**")
        st.markdown(
            f'<div class="re-card" style="background:rgba(27,175,122,.08)">{esc(cp.whatsapp_message)}</div>',
            unsafe_allow_html=True,
        )
    with e:
        st.markdown("**Email**")
        st.markdown(
            f'<div class="re-card"><b>{esc(cp.email_subject)}</b><br><span style="opacity:.7">{esc(cp.email_preheader)}</span></div>',
            unsafe_allow_html=True,
        )

    g = camp.guardrails
    status = "✅ Passed all guardrails" if g.passed else "❌ Guardrail violations"
    st.markdown(
        f"{status} · provider **{camp.provider}**{f' ({camp.model})' if camp.model else ''} · "
        f"{camp.attempts} attempt(s) · {camp.latency_ms} ms{' · cached' if camp.cached else ''}"
    )
    if camp.fallback_reason:
        st.warning(f"Fell back to template: {camp.fallback_reason}")
    for v in g.violations:
        st.error(v)
    for wmsg in g.warnings:
        st.caption(f"⚠️ {wmsg}")
    if cp.rationale:
        st.caption(f"Rationale: {cp.rationale}")
    st.download_button(
        "Download campaign (JSON)",
        json.dumps(camp.model_dump(mode="json"), indent=2, ensure_ascii=False),
        f"{brief.audience_id}.json",
        "application/json",
    )
