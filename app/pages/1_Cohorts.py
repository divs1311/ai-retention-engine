"""Cohort retention: heatmaps and curves."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retention_engine.analytics.cohorts import (
    average_retention_curve,
    cohort_counts,
    retention_matrix,
)

from _shared import SEQUENTIAL, SERIES, load_artifacts, page_setup, show, sidebar_filters, style

page_setup("Cohorts", "📈")
art = load_artifacts()
orders = sidebar_filters(art.orders, art.users)

st.title("Cohort retention")
st.markdown(
    '<p class="re-caption">Rows are monthly acquisition cohorts (first order month). '
    "Cells in periods that haven't fully happened yet are left blank rather than shown as "
    "false drop-offs.</p>",
    unsafe_allow_html=True,
)
if orders.empty:
    st.warning("No orders match the current filters.")
    st.stop()

c1, c2 = st.columns([2, 1])
metric = c1.radio(
    "Metric",
    ["users", "revenue", "orders"],
    horizontal=True,
    format_func={
        "users": "User retention",
        "revenue": "Revenue retention",
        "orders": "Order retention",
    }.get,
)
max_months = c2.slider("Months shown", 6, 18, 12)

mat = retention_matrix(orders, metric).iloc[:, : max_months + 1]
sizes = cohort_counts(orders, "users")[0]
labels = [f"{idx}  (n={int(sizes[idx]):,})" for idx in mat.index]

zmax = 100 if metric == "users" else float(min(mat.iloc[:, 1:].max().max(), 150))
fig = go.Figure(
    go.Heatmap(
        z=mat.to_numpy(),
        x=[f"M{c}" for c in mat.columns],
        y=labels,
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        zmin=0,
        zmax=zmax,
        text=[["" if pd.isna(v) else f"{v:.0f}" for v in row] for row in mat.to_numpy()],
        texttemplate="%{text}",
        textfont=dict(size=10),
        xgap=2,
        ygap=2,
        hovertemplate="Cohort %{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="%", thickness=10),
    )
)
fig.update_yaxes(autorange="reversed")
show(style(fig, height=max(420, 26 * len(mat)), title=f"{metric.title()} retention by cohort (%)"))
if metric == "revenue":
    st.caption(
        "Revenue retention can exceed 100% when surviving users spend more than the cohort did in month 0."
    )

st.subheader("Average retention curve")
st.caption("Cohort-size-weighted, using only observed cells.")
split = st.radio(
    "Compare by", ["None", "acquisition_channel", "city", "signup_promo"], horizontal=True
)
fig = go.Figure()
if split == "None":
    curve = average_retention_curve(orders, metric)
    fig.add_scatter(
        x=curve.index,
        y=curve.values,
        mode="lines+markers",
        name="All users",
        line_color=SERIES[0],
        marker_size=8,
        hovertemplate="M%{x}: %{y:.1f}%<extra></extra>",
    )
else:
    groups = art.users.set_index("user_id")[split]
    values = sorted(groups.unique(), key=str)
    for i, g in enumerate(values[: len(SERIES)]):
        sub = orders[orders["user_id"].map(groups) == g]
        if sub["user_id"].nunique() < 50:
            continue
        curve = average_retention_curve(sub, metric)
        name = (
            {True: "Promo-acquired", False: "No promo"}.get(g, str(g))
            if split == "signup_promo"
            else str(g)
        )
        fig.add_scatter(
            x=curve.index,
            y=curve.values,
            mode="lines",
            name=name,
            line_color=SERIES[i],
            hovertemplate=f"{name} · M%{{x}}: %{{y:.1f}}%<extra></extra>",
        )
fig.update_xaxes(range=[-0.3, max_months + 0.3], dtick=1, title="Months since acquisition")
fig.update_yaxes(
    title="% of cohort", range=[0, 105] if metric == "users" else None, rangemode="tozero"
)
show(style(fig, height=420, hovermode="x unified"))

with st.expander("Table view"):
    st.dataframe(mat.round(1), use_container_width=True)
