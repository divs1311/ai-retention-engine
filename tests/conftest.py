from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from retention_engine.config import Settings, SimulationConfig
from retention_engine.data.synthetic import generate


@pytest.fixture(scope="session")
def small_sim() -> SimulationConfig:
    return SimulationConfig(n_users=900, seed=7, end_date=date(2026, 3, 31))


@pytest.fixture(scope="session")
def settings(tmp_path_factory, small_sim) -> Settings:
    root = tmp_path_factory.mktemp("re")
    return Settings(
        data_dir=root / "data",
        artifacts_dir=root / "artifacts",
        simulation=small_sim,
    )


@pytest.fixture(scope="session")
def dataset(small_sim):
    ds = generate(small_sim)
    return ds.users, ds.orders


@pytest.fixture(scope="session")
def trained(dataset, settings):
    from retention_engine.churn import model

    users, orders = dataset
    return model.train(orders, users, settings.churn)


def make_orders(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    """Tiny hand-built order table: (user_id, timestamp, amount)."""
    df = pd.DataFrame(rows, columns=["user_id", "order_ts", "order_amount"])
    df["order_ts"] = pd.to_datetime(df["order_ts"], format="mixed")
    n = len(df)
    df.insert(0, "order_id", [f"T{i}" for i in range(n)])
    df["items_count"] = 3
    df["category_width"] = 2
    df["primary_category"] = "staples"
    df["gross_amount"] = df["order_amount"]
    df["discount_amount"] = 0.0
    df["delivery_minutes"] = 10.0
    df["is_late"] = False
    df["payment_method"] = "upi"
    return df


def make_users(ids: list[int], signup: str = "2025-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ids,
            "signup_date": pd.Timestamp(signup),
            "city": "Pune",
            "acquisition_channel": "organic",
            "device_os": "android",
            "signup_promo": False,
        }
    )
