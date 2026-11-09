"""RFM segmentation."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from _shared import SERIES, load_artifacts, money, page_setup, show, sidebar_filters, stat, style
from retention_engine.analytics.rfm import compute_rfm, segment_summary

page_setup("Segments", "🧩")
art = load_artifacts()
orders = sidebar_filters(art.orders, art.users)

st.title("RFM segments")
st.markdown(
    '<p class="re-caption">Recency, frequency and monetary quintiles mapped to the standard '
    "R×F segment grid, computed over the full order history.</p>",
    unsafe_allow_html=True,
)
if orders.empty:
    st.warning("No orders match the current filters.")
    st.stop()

rfm = compute_rfm(orders)
summary = segment_summary(rfm)

top = summary.sort_values("revenue_share_pct", ascending=False).iloc[0]
c = st.columns(3)
stat(c[0], "Customers", f"{len(rfm):,}")
stat(
    c[1],
    f"Revenue from {top.segment}",
    f"{top.revenue_share_pct:.0f}%",
    f"…from {top.user_share_pct:.0f}% of customers",
)
risky = summary[summary["segment"].isin(["At Risk", "Can't Lose"])]
stat(
    c[2],
    "Lifetime revenue in At Risk + Can't Lose",
    money(risky["revenue"].sum()),
    "Highest-value win-back pool",
)

s = summary.sort_values("users")
fig = go.Figure()
fig.add_bar(
    y=s["segment"],
    x=s["user_share_pct"],
    orientation="h",
    name="Share of customers",
    marker_color=SERIES[0],
    hovertemplate="%{y}: %{x:.1f}% of customers<extra></extra>",
)
fig.add_bar(
    y=s["segment"],
    x=s["revenue_share_pct"],
    orientation="h",
    name="Share of revenue",
    marker_color=SERIES[1],
    hovertemplate="%{y}: %{x:.1f}% of revenue<extra></extra>",
)
show(
    style(
        fig,
        height=430,
        title="Customers vs revenue by segment (%)",
        barmode="group",
        bargroupgap=0.08,
    )
)

st.subheader("Segment playbook")
st.dataframe(
    summary[
        [
            "segment",
            "users",
            "user_share_pct",
            "revenue_share_pct",
            "avg_recency_days",
            "avg_orders",
            "description",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "user_share_pct": st.column_config.ProgressColumn(
            "Customers %", format="%.1f%%", min_value=0, max_value=100
        ),
        "revenue_share_pct": st.column_config.ProgressColumn(
            "Revenue %", format="%.1f%%", min_value=0, max_value=100
        ),
        "avg_recency_days": st.column_config.NumberColumn("Avg recency (d)", format="%.0f"),
        "avg_orders": st.column_config.NumberColumn("Avg orders", format="%.1f"),
    },
)

st.subheader("R × F grid")
grid = rfm.pivot_table(
    index="f_score", columns="r_score", values="user_id", aggfunc="count"
).sort_index()
fig = go.Figure(
    go.Heatmap(
        z=grid.to_numpy(),
        x=[f"R{c}" for c in grid.columns],
        y=[f"F{i}" for i in grid.index],
        colorscale=[[0, "#cde2fb"], [1, "#184f95"]],
        xgap=2,
        ygap=2,
        text=grid.to_numpy(),
        texttemplate="%{text}",
        hovertemplate="%{y} · %{x}: %{z} customers<extra></extra>",
        showscale=False,
    )
)
show(
    style(
        fig,
        height=340,
        title="Customers by recency (→ more recent) and frequency (↑ more frequent) score",
    )
)
