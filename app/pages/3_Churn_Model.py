"""Model card: how good is the churn model, and how do we know?"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import MUTED, SERIES, load_artifacts, page_setup, show, stat, style

page_setup("Churn model", "🧠")
art = load_artifacts()
card = art.card
best = card["selected_model"]
m = card["metrics"][best]
b = card["baselines"]

st.title("Churn model")
st.markdown(
    f'<p class="re-caption">{card["churn_definition"]}<br>Validation: {card["validation"]}.</p>',
    unsafe_allow_html=True,
)

c = st.columns(5)
stat(c[0], "ROC-AUC", f"{m['roc_auc']:.3f}")
stat(c[1], "PR-AUC", f"{m['pr_auc']:.3f}", f"Base rate {m['base_rate']:.0%}")
stat(
    c[2],
    "Top-decile lift",
    f"{m['lift_at_top10pct']:.2f}×",
    f"{m['precision_at_top10pct']:.0%} precision in top 10%",
)
stat(
    c[3], "Recall @ top 20%", f"{m['recall_at_top20pct']:.0%}", "Churners caught contacting 1 in 5"
)
stat(c[4], "Calibration error", f"{m['ece']:.3f}", "ECE — lower is better")

st.subheader("Does the model beat simple rules?")
rows = [
    {
        "approach": f"Model — {r}",
        **{k: v[k] for k in ["roc_auc", "pr_auc", "lift_at_top10pct", "recall_at_top20pct"]},
    }
    for r, v in card["metrics"].items()
]
rows.append({"approach": "Rule — rank by recency", **b["recency_ranking"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.info(
    f"**Same outreach budget** ({b['inactive_14d_rule']['flagged_share']:.0%} of active users): "
    f"the original *inactive > 14 days* rule reaches precision {b['inactive_14d_rule']['precision']:.1%} / "
    f"recall {b['inactive_14d_rule']['recall']:.1%}; the model reaches "
    f"**{b['model_at_same_budget']['precision']:.1%} / {b['model_at_same_budget']['recall']:.1%}**."
)

cv = card["curves"]
left, right = st.columns(2)
with left:
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color=MUTED, width=1),
        name="Random",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=cv["roc"]["fpr"],
        y=cv["roc"]["tpr"],
        mode="lines",
        line_color=SERIES[0],
        name=best,
        hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>",
    )
    show(
        style(
            fig,
            height=340,
            title="ROC curve",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
        )
    )
with right:
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1],
        y=[m["base_rate"]] * 2,
        mode="lines",
        line=dict(color=MUTED, width=1),
        name="Base rate",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=cv["pr"]["recall"],
        y=cv["pr"]["precision"],
        mode="lines",
        line_color=SERIES[0],
        name=best,
        hovertemplate="Recall %{x:.2f} · Precision %{y:.2f}<extra></extra>",
    )
    show(
        style(
            fig,
            height=340,
            title="Precision–recall curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            yaxis_range=[0, 1.02],
        )
    )

left, right = st.columns(2)
with left:
    g = cv["gains"]
    fig = go.Figure(
        go.Bar(
            x=[f"D{d}" for d in g["decile"]],
            y=g["churn_rate"],
            marker_color=SERIES[0],
            customdata=g["lift"],
            hovertemplate="%{x}: %{y:.0%} churn · %{customdata:.2f}× lift<extra></extra>",
        )
    )
    fig.add_hline(
        y=m["base_rate"],
        line=dict(color=MUTED, width=1),
        annotation_text="base rate",
        annotation_position="top right",
    )
    show(
        style(
            fig,
            height=340,
            title="Observed churn rate by predicted-risk decile",
            yaxis_tickformat=".0%",
        )
    )
with right:
    cal = cv["calibration"]
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color=MUTED, width=1),
        name="Perfect",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=cal["predicted"],
        y=cal["observed"],
        mode="lines+markers",
        marker_size=9,
        line_color=SERIES[0],
        name="Model",
        customdata=cal["n"],
        hovertemplate="Predicted %{x:.0%} → observed %{y:.0%} (n=%{customdata})<extra></extra>",
    )
    show(
        style(
            fig,
            height=340,
            title="Calibration",
            xaxis_title="Predicted churn probability",
            yaxis_title="Observed churn rate",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
        )
    )

st.subheader("What drives the predictions")
imp = pd.Series(card["permutation_importance"]).sort_values().tail(12)
fig = go.Figure(
    go.Bar(
        x=imp.values,
        y=imp.index.str.replace("_", " "),
        orientation="h",
        marker_color=SERIES[0],
        hovertemplate="%{y}: −%{x:.4f} PR-AUC when shuffled<extra></extra>",
    )
)
show(style(fig, height=420, title="Permutation importance (drop in PR-AUC on held-out data)"))

with st.expander("Raw model card"):
    st.json({k: v for k, v in card.items() if k != "curves"})
