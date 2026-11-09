"""The prioritised list a retention team actually works from."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from _shared import SERIES, STATUS, load_artifacts, money, page_setup, show, style, tier_pill
from retention_engine.crm.schemas import LifecycleStage

page_setup("Risk queue", "🚨")
art = load_artifacts()
scores = art.scores

st.title("Risk queue")
st.markdown(
    f'<p class="re-caption">{len(scores):,} currently-active users, scored as of '
    f"{scores['scored_at'].iloc[0]:%d %b %Y} and ranked by value at risk "
    "(churn probability × trailing 90-day spend).</p>",
    unsafe_allow_html=True,
)

f1, f2, f3, f4 = st.columns(4)
tiers = f1.multiselect("Risk tier", ["high", "medium", "low"], default=["high", "medium"])
stages = f2.multiselect(
    "Lifecycle stage",
    [s.value for s in LifecycleStage],
    format_func=lambda v: LifecycleStage(v).label,
)
reasons = f3.multiselect("Primary reason", sorted(scores["primary_reason"].unique()))
segments = f4.multiselect("RFM segment", sorted(scores["rfm_segment"].dropna().unique()))

q = scores
if tiers:
    q = q[q["risk_tier"].isin(tiers)]
if stages:
    q = q[q["lifecycle_stage"].isin(stages)]
if reasons:
    q = q[q["primary_reason"].isin(reasons)]
if segments:
    q = q[q["rfm_segment"].isin(segments)]

c = st.columns(4)
c[0].metric("Users in queue", f"{len(q):,}")
c[1].metric("Value at risk", money(q["value_at_risk"].sum()))
c[2].metric("Avg churn probability", f"{q['churn_probability'].mean():.0%}" if len(q) else "–")
top10 = q["value_at_risk"].head(max(1, len(q) // 10)).sum() / max(q["value_at_risk"].sum(), 1)
c[3].metric(
    "VaR in top 10% of queue",
    f"{top10:.0%}",
    help="How concentrated the risk is — work the top of the list first.",
)

left, right = st.columns(2)
with left:
    fig = go.Figure()
    for tier in ["low", "medium", "high"]:
        sub = scores[scores["risk_tier"] == tier]
        fig.add_histogram(
            x=sub["churn_probability"],
            name=tier.title(),
            marker_color=STATUS[tier],
            xbins=dict(start=0, end=1, size=0.05),
            hovertemplate=f"{tier.title()} · %{{x}}: %{{y}} users<extra></extra>",
        )
    show(
        style(
            fig,
            height=320,
            title="Churn probability distribution (all active users)",
            barmode="stack",
            xaxis_tickformat=".0%",
            bargap=0.06,
        )
    )
with right:
    by = q.groupby("primary_reason")["value_at_risk"].sum().sort_values()
    fig = go.Figure(
        go.Bar(
            x=by.values,
            y=by.index.str.replace("_", " ").str.title(),
            orientation="h",
            marker_color=SERIES[0],
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        )
    )
    show(style(fig, height=320, title="Value at risk by primary reason"))

st.subheader("Queue")
cols = [
    "user_id",
    "risk_tier",
    "churn_probability",
    "value_at_risk",
    "lifecycle_stage",
    "rfm_segment",
    "reason_text",
    "recency_days",
    "orders_90d",
]
view = (
    q[cols]
    .head(500)
    .assign(lifecycle_stage=lambda d: d["lifecycle_stage"].map(lambda v: LifecycleStage(v).label))
)
st.dataframe(
    view,
    hide_index=True,
    use_container_width=True,
    column_config={
        "churn_probability": st.column_config.ProgressColumn(
            "Churn p", format="%.2f", min_value=0, max_value=1
        ),
        "value_at_risk": st.column_config.NumberColumn("Value at risk", format="%.0f"),
        "recency_days": st.column_config.NumberColumn("Recency (d)", format="%.0f"),
        "reason_text": st.column_config.TextColumn("Why", width="large"),
        "lifecycle_stage": "Stage",
        "rfm_segment": "Segment",
        "risk_tier": "Tier",
    },
)
st.download_button(
    "Download queue (CSV)", q[cols].to_csv(index=False), "risk_queue.csv", "text/csv"
)

st.subheader("User drill-down")
if q.empty:
    st.stop()
uid = st.selectbox("User", q["user_id"].head(200), format_func=lambda u: f"User {u}")
row = q[q["user_id"] == uid].iloc[0]
u = art.users.set_index("user_id").loc[uid]
st.markdown(
    f"{tier_pill(row.risk_tier)} &nbsp; **{row.churn_probability:.0%}** churn probability · "
    f"{LifecycleStage(row.lifecycle_stage).label} · {row.rfm_segment} · {u.city} · via {u.acquisition_channel}",
    unsafe_allow_html=True,
)
st.markdown("**Why:** " + row.reason_text)
hist = art.orders[art.orders["user_id"] == uid]
fig = go.Figure(
    go.Scatter(
        x=hist["order_ts"],
        y=hist["order_amount"],
        mode="markers",
        marker=dict(
            size=9,
            color=[STATUS["high"] if late else SERIES[0] for late in hist["is_late"]],
            line=dict(width=2, color="rgba(255,255,255,.9)"),
        ),
        customdata=hist[["delivery_minutes", "primary_category"]],
        hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} · %{customdata[1]}<br>%{customdata[0]:.0f} min delivery<extra></extra>",
    )
)
show(style(fig, height=280, title="Order history (red = late delivery)", showlegend=False))
