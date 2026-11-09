"""Business KPIs and time series for the executive view."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from retention_engine.analytics.cohorts import average_retention_curve


@dataclass(frozen=True)
class KPISummary:
    as_of: str
    total_users: int
    active_users_30d: int
    orders: int
    gmv: float
    aov: float
    orders_per_user: float
    repeat_rate_pct: float
    m1_retention_pct: float
    m3_retention_pct: float
    discount_rate_pct: float
    late_delivery_pct: float
    avg_delivery_minutes: float

    def to_dict(self) -> dict:
        return asdict(self)


def kpi_summary(orders: pd.DataFrame) -> KPISummary:
    as_of = orders["order_ts"].max()
    last_30 = orders[orders["order_ts"] > as_of - pd.Timedelta(days=30)]
    per_user = orders.groupby("user_id").size()
    curve = average_retention_curve(orders)
    return KPISummary(
        as_of=str(as_of.date()),
        total_users=int(orders["user_id"].nunique()),
        active_users_30d=int(last_30["user_id"].nunique()),
        orders=len(orders),
        gmv=round(float(orders["order_amount"].sum()), 2),
        aov=round(float(orders["order_amount"].mean()), 2),
        orders_per_user=round(float(per_user.mean()), 2),
        repeat_rate_pct=round(float((per_user > 1).mean() * 100), 1),
        m1_retention_pct=round(float(curve.get(1, float("nan"))), 1),
        m3_retention_pct=round(float(curve.get(3, float("nan"))), 1),
        discount_rate_pct=round(
            float(orders["discount_amount"].sum() / orders["gross_amount"].sum() * 100), 2
        ),
        late_delivery_pct=round(float(orders["is_late"].mean() * 100), 1),
        avg_delivery_minutes=round(float(orders["delivery_minutes"].mean()), 1),
    )


def weekly_trends(orders: pd.DataFrame) -> pd.DataFrame:
    """Weekly GMV, orders, active users and the new-vs-returning split."""
    df = orders.copy()
    df["week"] = df["order_ts"].dt.to_period("W-SUN").dt.start_time
    first_week = df.groupby("user_id")["week"].transform("min")
    df["is_new"] = df["week"] == first_week
    weekly = df.groupby("week").agg(
        gmv=("order_amount", "sum"),
        orders=("order_id", "count"),
        active_users=("user_id", "nunique"),
        late_rate=("is_late", "mean"),
    )
    new_users = df[df["is_new"]].groupby("week")["user_id"].nunique()
    weekly["new_users"] = new_users.reindex(weekly.index, fill_value=0)
    weekly["returning_users"] = weekly["active_users"] - weekly["new_users"]
    # Drop a trailing partial week so charts don't end with a fake cliff.
    last_ts = orders["order_ts"].max()
    if (last_ts - weekly.index.max()).days < 6:
        weekly = weekly.iloc[:-1]
    return weekly.reset_index()


def category_mix(orders: pd.DataFrame) -> pd.DataFrame:
    out = orders.groupby("primary_category").agg(
        orders=("order_id", "count"), gmv=("order_amount", "sum")
    )
    out["gmv_share_pct"] = (out["gmv"] / out["gmv"].sum() * 100).round(1)
    return out.sort_values("gmv", ascending=False).reset_index()
