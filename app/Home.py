"""Executive overview — the landing page of the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from _shared import CUR, SERIES, load_artifacts, money, page_setup, show, sidebar_filters, style

from retention_engine.analytics.cohorts import category_breadth_retention
from retention_engine.analytics.kpis import category_mix, kpi_summary, weekly_trends

page_setup("Overview")
art = load_artifacts()
orders = sidebar_filters(art.orders, art.users)

st.title("AI Retention Intelligence Engine")
st.markdown(
    '<p class="re-caption">Quick-commerce retention analytics → churn prediction → '
    "LLM-generated lifecycle CRM. Simulated marketplace; every number below is computed live.</p>",
    unsafe_allow_html=True,
)

if orders.empty:
    st.warning("No orders match the current filters.")
    st.stop()

k = kpi_summary(orders)
as_of = orders["order_ts"].max()
cur = orders[orders["order_ts"] > as_of - pd.Timedelta(days=30)]
prev = orders[
    (orders["order_ts"] > as_of - pd.Timedelta(days=60))
    & (orders["order_ts"] <= as_of - pd.Timedelta(days=30))
]


def delta(a: float, b: float) -> str | None:
    return f"{(a / b - 1) * 100:+.1f}% vs prior 30d" if b else None


c = st.columns(5)
c[0].metric("GMV (30d)", money(cur["order_amount"].sum()), delta(cur["order_amount"].sum(), prev["order_amount"].sum()))
c[1].metric("Active users (30d)", f"{cur['user_id'].nunique():,}", delta(cur["user_id"].nunique(), prev["user_id"].nunique()))
c[2].metric("AOV (30d)", money(cur["order_amount"].mean()), delta(cur["order_amount"].mean(), prev["order_amount"].mean()))
c[3].metric("M1 retention", f"{k.m1_retention_pct:.1f}%", help="Cohort-size-weighted share of users who order again in their 2nd calendar month.")
c[4].metric("Late deliveries (30d)", f"{cur['is_late'].mean() * 100:.1f}%", delta(cur["is_late"].mean(), prev["is_late"].mean()), delta_color="inverse")

# --- Insight callouts ---------------------------------------------------------------------
scores = art.scores[art.scores["user_id"].isin(orders["user_id"])]
breadth = category_breadth_retention(orders)
high = scores[scores["risk_tier"] == "high"]
st.markdown("#### What matters right now")
i1, i2, i3 = st.columns(3)
with i1:
    st.markdown(
        f'<div class="re-card"><b>{money(high["value_at_risk"].sum())}</b> of 90-day spend is at '
        f"high risk across <b>{len(high):,}</b> active users. "
        f"<a href='Risk_Queue' target='_self'>Open the risk queue →</a></div>",
        unsafe_allow_html=True,
    )
with i2:
    if len(breadth) >= 2:
        top, low = breadth.iloc[-1], breadth.iloc[0]
        st.markdown(
            f'<div class="re-card">Users who shop <b>{top.breadth_bucket}</b> categories in month 1 '
            f"retain at <b>{top.retention_pct:.0f}%</b> by day 90 vs <b>{low.retention_pct:.0f}%</b> "
            f"for {low.breadth_bucket}. Basket breadth is a retention lever.</div>",
            unsafe_allow_html=True,
        )
with i3:
    b = art.card["baselines"]
    st.markdown(
        f'<div class="re-card">At the same outreach budget as the old "inactive &gt; 14 days" rule, '
        f"the model lifts precision <b>{b['inactive_14d_rule']['precision']:.0%} → "
        f"{b['model_at_same_budget']['precision']:.0%}</b> and recall "
        f"<b>{b['inactive_14d_rule']['recall']:.0%} → {b['model_at_same_budget']['recall']:.0%}</b>.</div>",
        unsafe_allow_html=True,
    )

# --- Trends -------------------------------------------------------------------------------
w = weekly_trends(orders)
left, right = st.columns(2)
with left:
    fig = go.Figure(
        go.Scatter(
            x=w["week"], y=w["gmv"], mode="lines", line_color=SERIES[0], name="GMV",
            fill="tozeroy", fillcolor="rgba(42,120,214,.10)",
            hovertemplate="Week of %{x|%d %b %Y}<br>GMV %{y:,.0f}<extra></extra>",
        )
    )
    show(style(fig, title="Weekly GMV", yaxis_title=f"GMV ({CUR})"))
with right:
    fig = go.Figure()
    fig.add_bar(x=w["week"], y=w["returning_users"], name="Returning", marker_color=SERIES[0],
                hovertemplate="%{y:,} returning<extra></extra>")
    fig.add_bar(x=w["week"], y=w["new_users"], name="New", marker_color=SERIES[1],
                hovertemplate="%{y:,} new<extra></extra>")
    show(style(fig, title="Weekly active users — new vs returning", barmode="stack", hovermode="x unified"))

left, right = st.columns(2)
with left:
    mix = category_mix(orders).sort_values("gmv")
    fig = go.Figure(
        go.Bar(
            x=mix["gmv"], y=mix["primary_category"].str.replace("_", " "), orientation="h",
            marker_color=SERIES[0], text=mix["gmv_share_pct"].map(lambda v: f"{v:.0f}%"),
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>GMV %{x:,.0f}<extra></extra>",
        )
    )
    show(style(fig, title="GMV by primary category", height=380, xaxis_title=None))
with right:
    fig = go.Figure(
        go.Bar(
            x=breadth["breadth_bucket"].astype(str), y=breadth["retention_pct"],
            marker_color=SERIES[0], text=breadth["retention_pct"].map(lambda v: f"{v:.0f}%"),
            textposition="outside", cliponaxis=False,
            customdata=breadth[["users", "lift_vs_avg"]],
            hovertemplate="%{x} categories<br>%{y:.1f}% retained<br>%{customdata[0]:,} users · "
            "%{customdata[1]:.2f}× avg<extra></extra>",
        )
    )
    show(style(fig, title="Day 60–90 retention by month-1 basket breadth", height=380,
               xaxis_title="Avg categories per order in first 30 days", yaxis_title="% retained"))

with st.expander("About this project"):
    st.markdown(
        """
        * **Data** — agent-based simulation of a 10-minute grocery marketplace (latent engagement,
          tenure-dependent churn hazard, late-delivery shocks, promo-acquired users, win-backs).
        * **Analytics** — censoring-aware cohort retention, revenue retention, RFM segmentation.
        * **Churn ML** — point-in-time features, out-of-time validation, calibrated gradient boosting,
          reason codes, value-at-risk prioritisation.
        * **CRM** — policy-controlled incentives, Anthropic/OpenAI/template providers, guardrails with
          retry-and-fallback.

        Source: [github.com/divs1311/ai-retention-engine](https://github.com/divs1311/ai-retention-engine)
        """
    )
