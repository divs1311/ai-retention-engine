"""Data contracts for the two input tables.

Validation is explicit and dependency-free so the engine can also run on real exports:
point it at your own ``users``/``orders`` files and it tells you exactly what's wrong.
"""

from __future__ import annotations

import pandas as pd

USERS_SCHEMA: dict[str, str] = {
    "user_id": "int",
    "signup_date": "datetime",
    "city": "str",
    "acquisition_channel": "str",
    "device_os": "str",
    "signup_promo": "bool",
}

ORDERS_SCHEMA: dict[str, str] = {
    "order_id": "str",
    "user_id": "int",
    "order_ts": "datetime",
    "items_count": "int",
    "category_width": "int",
    "primary_category": "str",
    "gross_amount": "float",
    "discount_amount": "float",
    "order_amount": "float",
    "delivery_minutes": "float",
    "is_late": "bool",
    "payment_method": "str",
}


class DataValidationError(ValueError):
    def __init__(self, table: str, problems: list[str]):
        self.table = table
        self.problems = problems
        super().__init__(f"{table} failed validation:\n  - " + "\n  - ".join(problems))


def _coerce(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for col, kind in schema.items():
        if col not in df:
            continue
        if kind == "datetime":
            df[col] = pd.to_datetime(df[col])
        elif kind == "int":
            df[col] = pd.to_numeric(df[col]).astype("int64")
        elif kind == "float":
            df[col] = pd.to_numeric(df[col]).astype("float64")
        elif kind == "bool":
            if not pd.api.types.is_bool_dtype(df[col]):
                df[col] = (
                    df[col].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
                )
            df[col] = df[col].astype(bool)
        else:
            df[col] = df[col].astype(str)
    return df


def validate_users(users: pd.DataFrame) -> pd.DataFrame:
    problems = [f"missing column '{c}'" for c in USERS_SCHEMA if c not in users]
    if problems:
        raise DataValidationError("users", problems)
    users = _coerce(users, USERS_SCHEMA)
    if users["user_id"].duplicated().any():
        problems.append(f"{int(users['user_id'].duplicated().sum())} duplicate user_id values")
    if users["signup_date"].isna().any():
        problems.append("null signup_date values")
    if problems:
        raise DataValidationError("users", problems)
    return users


def validate_orders(orders: pd.DataFrame, users: pd.DataFrame | None = None) -> pd.DataFrame:
    problems = [f"missing column '{c}'" for c in ORDERS_SCHEMA if c not in orders]
    if problems:
        raise DataValidationError("orders", problems)
    orders = _coerce(orders, ORDERS_SCHEMA)

    if orders["order_id"].duplicated().any():
        problems.append(f"{int(orders['order_id'].duplicated().sum())} duplicate order_id values")
    for col in ("gross_amount", "order_amount", "discount_amount", "delivery_minutes"):
        if (orders[col] < 0).any():
            problems.append(f"negative values in '{col}'")
    if (orders["discount_amount"] > orders["gross_amount"] + 1e-6).any():
        problems.append("discount_amount exceeds gross_amount")
    if (orders["category_width"] < 1).any():
        problems.append("category_width must be >= 1")
    if orders["order_ts"].isna().any():
        problems.append("null order_ts values")

    if users is not None:
        unknown = ~orders["user_id"].isin(users["user_id"])
        if unknown.any():
            problems.append(f"{int(unknown.sum())} orders reference unknown user_id")
        signup = orders["user_id"].map(users.set_index("user_id")["signup_date"])
        early = orders["order_ts"].dt.normalize() < signup
        if early.any():
            problems.append(f"{int(early.sum())} orders placed before the user's signup_date")

    if problems:
        raise DataValidationError("orders", problems)
    return orders.sort_values("order_ts", kind="stable").reset_index(drop=True)
