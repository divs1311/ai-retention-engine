from __future__ import annotations

import pandas as pd
import pytest

from retention_engine.analytics.cohorts import average_retention_curve
from retention_engine.data.schema import DataValidationError, validate_orders, validate_users
from retention_engine.data.synthetic import generate


def test_generation_is_deterministic(small_sim):
    a, b = generate(small_sim), generate(small_sim)
    pd.testing.assert_frame_equal(a.orders, b.orders)
    pd.testing.assert_frame_equal(a.users, b.users)


def test_generated_data_passes_contracts(dataset):
    users, orders = dataset
    users = validate_users(users)
    validate_orders(orders, users)


def test_latent_columns_are_not_published(dataset):
    users, _ = dataset
    assert not [c for c in users.columns if c.startswith("_")]


def test_orders_inside_window_and_after_signup(dataset, small_sim):
    users, orders = dataset
    assert orders["order_ts"].min() >= pd.Timestamp(small_sim.start_date)
    assert orders["order_ts"].max() < pd.Timestamp(small_sim.end_date) + pd.Timedelta(days=1)
    signup = orders["user_id"].map(users.set_index("user_id")["signup_date"])
    assert (orders["order_ts"].dt.normalize() >= signup).all()


def test_every_user_has_a_first_order_on_signup_day(dataset):
    users, orders = dataset
    first = orders.groupby("user_id")["order_ts"].min().dt.normalize()
    signup = users.set_index("user_id")["signup_date"].reindex(first.index)
    assert (first == signup).mean() > 0.99


def test_marketplace_looks_realistic(dataset):
    _, orders = dataset
    curve = average_retention_curve(orders)
    assert 25 < curve[1] < 60, "Month-1 retention should show the classic steep drop"
    assert curve[1] > curve[3] > curve[6]
    assert 0.05 < orders["is_late"].mean() < 0.25
    assert 250 < orders["order_amount"].mean() < 700


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda o: o.drop(columns="order_amount"), "missing column 'order_amount'"),
        (lambda o: o.assign(order_amount=-1.0), "negative values in 'order_amount'"),
        (lambda o: o.assign(discount_amount=o["gross_amount"] + 5), "discount_amount exceeds"),
        (lambda o: pd.concat([o, o.head(1)]), "duplicate order_id"),
        (lambda o: o.assign(category_width=0), "category_width must be >= 1"),
    ],
)
def test_order_validation_catches_bad_data(dataset, mutate, message):
    users, orders = dataset
    with pytest.raises(DataValidationError, match=message):
        validate_orders(mutate(orders.head(50).copy()), users)


def test_validation_flags_unknown_users_and_pre_signup_orders(dataset):
    users, orders = dataset
    bad = orders.head(5).copy()
    bad.loc[bad.index[0], "user_id"] = 10**9
    bad.loc[bad.index[1], "order_ts"] = pd.Timestamp("2000-01-01")
    with pytest.raises(DataValidationError) as exc:
        validate_orders(bad, users)
    joined = " ".join(exc.value.problems)
    assert "unknown user_id" in joined and "before the user's signup_date" in joined


def test_csv_style_booleans_are_coerced(dataset):
    users, _ = dataset
    u = users.head(3).assign(signup_promo=["True", "false", "1"])
    assert validate_users(u)["signup_promo"].tolist() == [True, False, True]
