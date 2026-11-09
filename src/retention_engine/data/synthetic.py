"""Synthetic quick-commerce marketplace simulator.

The original generator drew each month's retention independently from a fixed curve, so
nothing about a user's behaviour actually predicted whether they'd come back. This
simulator is agent-based instead: every user carries latent state that evolves weekly,
and churn is a *consequence* of that state. The signals a real retention team watches
(declining frequency, late deliveries, narrow baskets, promo-driven acquisition) therefore
carry real predictive signal, which is what makes the downstream churn model meaningful.

Generative story (per user, per week)
------------------------------------
* ``z`` – log-engagement, an AR(1) process. Late deliveries knock it down; good
  experiences nudge it up.
* order count ~ Poisson(base_rate · exp(0.8·z) · seasonality)
* weekly churn hazard = h0 · tenure_curve · channel · promo · breadth · exp(-1.2·z)
  (early-tenure hazard is high, which reproduces the steep Month-1 drop-off).
* churned ("dormant") users reactivate with a small weekly probability (organic win-back).

Everything is vectorised over users; the only Python loop is over ~78 weeks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retention_engine.config import SimulationConfig
from retention_engine.logging_utils import get_logger

log = get_logger(__name__)

CITIES = {
    # city: (population share, mean delivery minutes)
    "Bengaluru": (0.24, 10.8),
    "Mumbai": (0.21, 11.6),
    "Delhi NCR": (0.22, 12.6),
    "Hyderabad": (0.13, 10.4),
    "Pune": (0.11, 11.2),
    "Chennai": (0.09, 12.2),
}

CHANNELS = {
    # channel: (share, churn-hazard multiplier)
    "organic": (0.30, 0.80),
    "referral": (0.14, 0.70),
    "paid_social": (0.24, 1.25),
    "performance": (0.20, 1.10),
    "influencer": (0.12, 1.40),
}

ARCHETYPES = {
    # archetype: (share, weekly order rate, category breadth range, items per order)
    "fresh_only": (0.50, 0.75, (1, 2), 3.0),
    "mixed_basket": (0.38, 1.30, (3, 5), 5.5),
    "power_user": (0.12, 2.40, (6, 10), 8.5),
}

CATEGORIES = [
    "fruits_vegetables",
    "dairy_bread_eggs",
    "snacks_beverages",
    "staples",
    "personal_care",
    "household",
    "meat_seafood",
    "frozen_instant",
    "baby_care",
    "pharmacy",
]
# Category affinity by archetype (rows align with ARCHETYPES order).
_CATEGORY_P = np.array(
    [
        [0.38, 0.30, 0.18, 0.04, 0.02, 0.02, 0.03, 0.02, 0.005, 0.005],
        [0.20, 0.18, 0.20, 0.13, 0.08, 0.08, 0.06, 0.05, 0.01, 0.01],
        [0.14, 0.13, 0.15, 0.15, 0.11, 0.11, 0.08, 0.07, 0.03, 0.03],
    ]
)
_CATEGORY_P = _CATEGORY_P / _CATEGORY_P.sum(axis=1, keepdims=True)

PAYMENT_METHODS = {"upi": 0.68, "card": 0.14, "cod": 0.12, "wallet": 0.06}

# Simulation dynamics -----------------------------------------------------------------------
H0 = 0.042  # baseline weekly churn hazard
EARLY_TENURE_BOOST = 5.5  # hazard multiplier at week 0, decays with tenure
EARLY_TENURE_DECAY_WEEKS = 3.0
PROMO_HAZARD = 1.35
BREADTH_HAZARD_SLOPE = 0.09  # each extra category of breadth lowers hazard ~9%
Z_PERSISTENCE = 0.85
Z_NOISE = 0.35
LATE_SHOCK = 0.45
GOOD_ORDER_LIFT = 0.04
REACTIVATION_P = 0.012
DELIVERY_SIGMA = 0.23


@dataclass(frozen=True)
class SyntheticDataset:
    users: pd.DataFrame
    orders: pd.DataFrame


def _seasonality(week_starts: pd.DatetimeIndex) -> np.ndarray:
    """Order-rate multiplier per week: festive season, monsoon, and a mild year-end dip."""
    m = week_starts.month.to_numpy()
    mult = np.ones(len(week_starts))
    mult[np.isin(m, [10, 11])] *= 1.22  # Navratri / Diwali
    mult[np.isin(m, [7, 8])] *= 1.10  # monsoon — people stay in
    mult[m == 12] *= 0.95
    return mult


def _sample_users(cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_users
    start = pd.Timestamp(cfg.start_date)
    span_days = (pd.Timestamp(cfg.end_date) - start).days
    signup_span = max(int(span_days * cfg.signup_window_frac), 1)

    # Growth ramp: later days are ~2.5x more likely to see a signup than day 0.
    day_weights = np.linspace(1.0, 2.5, signup_span)
    signup_offset = rng.choice(signup_span, size=n, p=day_weights / day_weights.sum())

    archetype = rng.choice(list(ARCHETYPES), size=n, p=[v[0] for v in ARCHETYPES.values()])
    lo = np.array([ARCHETYPES[a][2][0] for a in archetype])
    hi = np.array([ARCHETYPES[a][2][1] for a in archetype])

    users = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "signup_date": start + pd.to_timedelta(np.sort(signup_offset), unit="D"),
            "city": rng.choice(list(CITIES), size=n, p=[v[0] for v in CITIES.values()]),
            "acquisition_channel": rng.choice(
                list(CHANNELS), size=n, p=[v[0] for v in CHANNELS.values()]
            ),
            "device_os": rng.choice(["android", "ios"], size=n, p=[0.78, 0.22]),
            "signup_promo": rng.random(n) < 0.45,
        }
    )
    # Latent attributes are kept separate from the published table.
    users["_archetype"] = archetype
    users["_breadth"] = rng.integers(lo, hi + 1)
    users["_base_rate"] = np.array([ARCHETYPES[a][1] for a in archetype]) * rng.gamma(4.0, 0.25, n)
    return users


def generate(cfg: SimulationConfig | None = None) -> SyntheticDataset:
    """Simulate users and orders. Deterministic for a given config/seed."""
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    users = _sample_users(cfg, rng)
    n = len(users)

    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date) + pd.Timedelta(hours=23, minutes=59)
    week_starts = pd.date_range(start, end, freq="7D")
    n_weeks = len(week_starts)
    season = _seasonality(week_starts)

    signup_week = ((users["signup_date"] - start).dt.days // 7).to_numpy()
    channel_mult = users["acquisition_channel"].map({k: v[1] for k, v in CHANNELS.items()})
    promo_mult = np.where(users["signup_promo"], PROMO_HAZARD, 1.0)
    breadth = users["_breadth"].to_numpy()
    static_hazard = (
        H0 * channel_mult.to_numpy() * promo_mult * np.exp(-BREADTH_HAZARD_SLOPE * (breadth - 3))
    )
    base_rate = users["_base_rate"].to_numpy()
    city_delivery = users["city"].map({k: v[1] for k, v in CITIES.items()}).to_numpy()

    z = rng.normal(0.0, 0.3, n)
    alive = np.zeros(n, dtype=bool)

    rec_user: list[np.ndarray] = []
    rec_week: list[np.ndarray] = []
    rec_mins: list[np.ndarray] = []
    rec_first: list[np.ndarray] = []

    for w in range(n_weeks):
        new = signup_week == w
        alive |= new
        tenure = w - signup_week

        # Orders this week. New users always place their first order (that's how they
        # entered the dataset) plus whatever their rate implies.
        lam = base_rate * np.exp(0.8 * z) * season[w]
        counts = np.where(alive, rng.poisson(lam), 0)
        counts = np.where(new, np.maximum(counts, 1), counts)

        total = int(counts.sum())
        if total:
            idx = np.repeat(np.arange(n), counts)
            mins = rng.lognormal(np.log(city_delivery[idx]), DELIVERY_SIGMA)
            late = mins > cfg.late_delivery_minutes
            late_per_user = np.bincount(idx, weights=late, minlength=n)
            good_per_user = counts - late_per_user
            first = np.zeros(total, dtype=bool)
            # mark the first order of each brand-new user
            new_idx = np.flatnonzero(new)
            if new_idx.size:
                starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
                first[starts[new_idx]] = True
            rec_user.append(idx)
            rec_week.append(np.full(total, w))
            rec_mins.append(mins)
            rec_first.append(first)
        else:
            late_per_user = good_per_user = np.zeros(n)

        # Engagement update (only meaningful for users who exist).
        z = (
            Z_PERSISTENCE * z
            + rng.normal(0.0, Z_NOISE, n)
            - LATE_SHOCK * late_per_user
            + GOOD_ORDER_LIFT * np.minimum(good_per_user, 3)
        )

        # Churn & reactivation.
        tenure_mult = 1 + EARLY_TENURE_BOOST * np.exp(
            -np.clip(tenure, 0, None) / EARLY_TENURE_DECAY_WEEKS
        )
        hazard = np.clip(static_hazard * tenure_mult * np.exp(-1.2 * z), 0, 0.95)
        churn_now = alive & (rng.random(n) < hazard)
        exists = signup_week <= w
        reactivate = exists & ~alive & (rng.random(n) < REACTIVATION_P)
        alive = (alive & ~churn_now) | reactivate
        z = np.where(reactivate, -0.3, z)

    orders = _materialise_orders(
        users,
        np.concatenate(rec_user),
        np.concatenate(rec_week),
        np.concatenate(rec_mins),
        np.concatenate(rec_first),
        week_starts,
        end,
        cfg,
        rng,
    )
    public_users = users.drop(columns=[c for c in users.columns if c.startswith("_")])
    log.info(
        "Simulated %s users and %s orders (%s → %s)",
        f"{n:,}",
        f"{len(orders):,}",
        orders["order_ts"].min().date(),
        orders["order_ts"].max().date(),
    )
    return SyntheticDataset(users=public_users, orders=orders)


def _materialise_orders(
    users: pd.DataFrame,
    user_idx: np.ndarray,
    week: np.ndarray,
    delivery_minutes: np.ndarray,
    is_first: np.ndarray,
    week_starts: pd.DatetimeIndex,
    end: pd.Timestamp,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    m = len(user_idx)
    signup = users["signup_date"].to_numpy()[user_idx]

    # Timestamp: weekend-weighted day, bimodal (breakfast / dinner) hour of day.
    day = rng.choice(7, size=m, p=[0.12, 0.12, 0.13, 0.13, 0.15, 0.18, 0.17])
    evening = rng.random(m) < 0.62
    hour = np.where(
        evening,
        rng.normal(20.0, 1.8, m),
        rng.normal(9.0, 1.6, m),
    ).clip(6, 23.9)
    ts = (
        week_starts.to_numpy()[week]
        + pd.to_timedelta(day, unit="D").to_numpy()
        + pd.to_timedelta(hour * 60, unit="min").to_numpy()
    )
    # First orders happen on the signup day; nothing happens before signup or after end.
    signup_ts = signup + pd.to_timedelta(hour * 60, unit="min").to_numpy()
    ts = np.where(is_first | (ts < signup), signup_ts, ts)
    keep = ts <= end.to_datetime64()

    archetype = users["_archetype"].to_numpy()[user_idx]
    arch_codes = pd.Categorical(archetype, categories=list(ARCHETYPES)).codes
    breadth = users["_breadth"].to_numpy()[user_idx]
    mean_items = np.array([ARCHETYPES[a][3] for a in ARCHETYPES])[arch_codes]

    items = 1 + rng.poisson(mean_items - 1)
    cat_width = np.clip(np.rint(rng.normal(breadth, 0.9)), 1, None).astype(int)
    cat_width = np.minimum(cat_width, np.minimum(items, len(CATEGORIES)))

    # Primary category per order, drawn from the archetype's affinity row.
    u = rng.random(m)
    cdf = np.cumsum(_CATEGORY_P, axis=1)[arch_codes]
    primary = np.asarray(CATEGORIES)[(u[:, None] > cdf).sum(axis=1).clip(max=len(CATEGORIES) - 1)]

    item_price = rng.lognormal(np.log(78), 0.35, m)
    gross = np.round(items * item_price, 2)

    promo = users["signup_promo"].to_numpy()[user_idx]
    order_rank = pd.Series(ts).groupby(user_idx).rank(method="first").to_numpy()
    promo_order = promo & (order_rank <= 3)
    coupon = rng.random(m) < 0.14
    discount_rate = np.where(promo_order, 0.25, np.where(coupon, 0.12, 0.0))
    discount = np.round(np.minimum(gross * discount_rate, np.where(promo_order, 150, 75)), 2)

    orders = pd.DataFrame(
        {
            "user_id": users["user_id"].to_numpy()[user_idx],
            "order_ts": pd.to_datetime(ts).floor("min"),
            "items_count": items,
            "category_width": cat_width,
            "primary_category": primary,
            "gross_amount": gross,
            "discount_amount": discount,
            "order_amount": np.round(gross - discount, 2),
            "delivery_minutes": np.round(delivery_minutes, 1),
            "is_late": delivery_minutes > cfg.late_delivery_minutes,
            "payment_method": rng.choice(
                list(PAYMENT_METHODS), size=m, p=list(PAYMENT_METHODS.values())
            ),
        }
    )[keep]
    orders = orders.sort_values(["order_ts", "user_id"], kind="stable").reset_index(drop=True)
    orders.insert(0, "order_id", [f"O{i:07d}" for i in range(1, len(orders) + 1)])
    return orders
