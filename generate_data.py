# ============================================================
# QUICK-COMMERCE TRANSACTION DATA SYNTHESIZER
# ============================================================
# Generates:
#   user_transactions.csv
#
# Characteristics:
# - 20K+ transactions
# - 2,500 users
# - Realistic quick-commerce retention decay
# - Steep Month-1 dropoff
# - Stabilized power cohorts
# ============================================================

import uuid
import random
from datetime import timedelta

import numpy as np
import pandas as pd

# -----------------------------
# GLOBAL CONFIG
# -----------------------------
np.random.seed(42)
random.seed(42)

TOTAL_USERS = 2500
TARGET_TRANSACTIONS = 20000

START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# Quick-commerce category archetypes
CATEGORY_ARCHETYPES = {
    "fresh_only": [1, 2],
    "mixed_basket": [3, 5],
    "power_user": [6, 10]
}

# Cohort decay assumptions
# Month 1 retention intentionally ~35%
RETENTION_CURVE = {
    0: 1.00,
    1: 0.36,
    2: 0.28,
    3: 0.24,
    4: 0.21,
    5: 0.19,
    6: 0.18,
    7: 0.16,
    8: 0.15,
    9: 0.14,
    10: 0.13,
    11: 0.12
}

# -----------------------------
# USER GENERATION
# -----------------------------
users = []

for user_id in range(1, TOTAL_USERS + 1):

    signup_offset = np.random.randint(0, 365)
    signup_date = START_DATE + timedelta(days=int(signup_offset))

    # User segment distribution
    segment = np.random.choice(
        ["fresh_only", "mixed_basket", "power_user"],
        p=[0.55, 0.35, 0.10]
    )

    category_width = random.randint(
        CATEGORY_ARCHETYPES[segment][0],
        CATEGORY_ARCHETYPES[segment][1]
    )

    users.append({
        "user_id": user_id,
        "signup_date": signup_date,
        "segment": segment,
        "base_category_width": category_width
    })

users_df = pd.DataFrame(users)

# -----------------------------
# TRANSACTION SYNTHESIS
# -----------------------------
transactions = []

for _, user in users_df.iterrows():

    user_id = user["user_id"]
    signup_date = user["signup_date"]
    base_category_width = user["base_category_width"]

    acquisition_month = signup_date.to_period("M")

    # Month 0 transactions
    initial_orders = np.random.poisson(3) + 1

    for _ in range(initial_orders):

        order_ts = signup_date + timedelta(
            days=np.random.randint(0, 28),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )

        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "user_id": user_id,
            "order_timestamp": order_ts,
            "order_amount": round(
                np.random.gamma(shape=2.2, scale=180),
                2
            ),
            "product_category_width": base_category_width,
            "signup_date": signup_date.date()
        })

    # Subsequent retention simulation
    for month_idx in range(1, 12):

        retention_probability = RETENTION_CURVE[month_idx]

        retained = np.random.rand() < retention_probability

        if retained:

            orders_this_month = np.random.poisson(2) + 1

            for _ in range(orders_this_month):

                base_date = signup_date + pd.DateOffset(months=month_idx)

                if base_date > END_DATE:
                    continue

                order_ts = base_date + timedelta(
                    days=np.random.randint(0, 28),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )

                dynamic_category_width = min(
                    base_category_width + np.random.binomial(2, 0.25),
                    12
                )

                transactions.append({
                    "transaction_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "order_timestamp": order_ts,
                    "order_amount": round(
                        np.random.gamma(shape=2.5, scale=220),
                        2
                    ),
                    "product_category_width": dynamic_category_width,
                    "signup_date": signup_date.date()
                })

# -----------------------------
# FINAL DATAFRAME
# -----------------------------
transactions_df = pd.DataFrame(transactions)

# Ensure target size
if len(transactions_df) > TARGET_TRANSACTIONS:
    transactions_df = transactions_df.sample(
        TARGET_TRANSACTIONS,
        random_state=42
    )

transactions_df = transactions_df.sort_values("order_timestamp")

# -----------------------------
# EXPORT CSV
# -----------------------------
transactions_df.to_csv(
    "user_transactions.csv",
    index=False
)

print("=" * 60)
print("DATASET GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"Transactions : {len(transactions_df):,}")
print(f"Unique Users : {transactions_df['user_id'].nunique():,}")
print(f"Date Range   : {transactions_df['order_timestamp'].min()} "
      f"to {transactions_df['order_timestamp'].max()}")
print("=" * 60)