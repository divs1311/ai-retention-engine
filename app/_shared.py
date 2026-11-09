"""Shared state, styling and chart helpers for the Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run app/Home.py` from a source checkout without installing the package.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from retention_engine.config import get_settings  # noqa: E402
from retention_engine.pipeline import Artifacts, ensure_artifacts  # noqa: E402

# Validated categorical order (fixed — colour follows the entity, never its rank).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Single-hue sequential ramp (blue 100 → 700) for magnitude, e.g. retention heatmaps.
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
# Reserved status colours — risk tiers only, always shown with a text label.
STATUS = {"high": "#d03b3b", "medium": "#ec835a", "low": "#0ca30c"}
MUTED = "#8a8984"

settings = get_settings()
CUR = settings.currency_symbol


def page_setup(title: str, icon: str = "🛒") -> None:
    st.set_page_config(page_title=f"{title} · Retention Engine", page_icon=icon, layout="wide")
    st.markdown(
        """
        <style>
          .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
          [data-testid="stMetricValue"] {font-size: 1.6rem;}
          .re-caption {color: var(--text-color); opacity: .65; font-size: .9rem; margin-top: -.6rem;}
          .re-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
                    padding: 14px 16px; margin-bottom: 10px;}
          .re-pill {display:inline-block; padding:2px 10px; border-radius:999px;
                    font-size:.78rem; font-weight:600; color:#fff;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Building data, model and scores (first run only)…")
def load_artifacts() -> Artifacts:
    return ensure_artifacts(settings)


def sidebar_filters(orders: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """City / channel filters applied to the order history."""
    with st.sidebar:
        st.header("Filters")
        cities = st.multiselect("City", sorted(users["city"].unique()))
        channels = st.multiselect("Acquisition channel", sorted(users["acquisition_channel"].unique()))
        st.caption("Empty = all. Filters apply to analytics pages.")
    u = users
    if cities:
        u = u[u["city"].isin(cities)]
    if channels:
        u = u[u["acquisition_channel"].isin(channels)]
    return orders[orders["user_id"].isin(u["user_id"])]


def money(x: float, decimals: int = 0) -> str:
    if abs(x) >= 1e7:
        return f"{CUR}{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"{CUR}{x / 1e5:.1f} L"
    return f"{CUR}{x:,.{decimals}f}"


def style(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    """Recessive chrome: hairline grid, no chart junk, consistent margins."""
    title = layout.pop("title", None)
    has_legend = len(fig.data) > 1 and layout.get("showlegend", True)
    defaults = dict(
        height=height,
        margin=dict(l=8, r=8, t=70 if has_legend else 40, b=8),
        title=dict(text=title, x=0, xanchor="left", y=0.985, yanchor="top") if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None),
        hoverlabel=dict(namelength=-1),
        bargap=0.25,
    )
    fig.update_layout(**{**defaults, **layout})
    fig.update_xaxes(showgrid=False, zeroline=False, linewidth=1, linecolor="rgba(128,128,128,.35)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", gridwidth=1, zeroline=False)
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    fig.update_traces(selector=dict(type="scatter", mode="lines"), line_width=2)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def stat(col, label: str, value: str, note: str | None = None, help: str | None = None) -> None:
    """A metric tile with an optional neutral note (no arrow — it's context, not a change)."""
    col.metric(label, value, help=help)
    if note:
        col.caption(note)


def tier_pill(tier: str) -> str:
    icon = {"high": "▲", "medium": "◆", "low": "▼"}.get(tier, "")
    return f'<span class="re-pill" style="background:{STATUS.get(tier, MUTED)}">{icon} {tier}</span>'
