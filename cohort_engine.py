# ============================================================
# COHORT RETENTION ENGINE
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("user_transactions.csv")

df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])

# ------------------------------------------------------------
# COHORT ASSIGNMENT
# ------------------------------------------------------------
df["order_month"] = df["order_timestamp"].dt.to_period("M")

# First transaction month = acquisition cohort
cohort_map = (
    df.groupby("user_id")["order_month"]
    .min()
    .rename("cohort_month")
)

df = df.merge(
    cohort_map,
    on="user_id",
    how="left"
)

# ------------------------------------------------------------
# COHORT INDEX CALCULATION
# ------------------------------------------------------------
df["cohort_index"] = (
    (df["order_month"].dt.year - df["cohort_month"].dt.year) * 12
    + (df["order_month"].dt.month - df["cohort_month"].dt.month)
)

# ------------------------------------------------------------
# ACTIVE USER AGGREGATION
# ------------------------------------------------------------
cohort_data = (
    df.groupby(["cohort_month", "cohort_index"])
    ["user_id"]
    .nunique()
    .reset_index()
)

# Absolute retention matrix
cohort_matrix = cohort_data.pivot(
    index="cohort_month",
    columns="cohort_index",
    values="user_id"
)

# ------------------------------------------------------------
# RETENTION %
# ------------------------------------------------------------
cohort_size = cohort_matrix.iloc[:, 0]

retention_matrix = cohort_matrix.divide(
    cohort_size,
    axis=0
)

retention_pct = (
    retention_matrix * 100
).round(1)

retention_display = retention_pct.astype(str) + "%"

# ------------------------------------------------------------
# EXPORTS
# ------------------------------------------------------------
cohort_matrix.to_csv("cohort_absolute_matrix.csv")
retention_pct.to_csv("cohort_retention_pct.csv")

print("\nABSOLUTE COHORT MATRIX")
print(cohort_matrix.head())

print("\nRETENTION MATRIX (%)")
print(retention_display.head())

# ------------------------------------------------------------
# HEATMAP VISUALIZATION
# ------------------------------------------------------------
def plot_retention_heatmap(retention_matrix_df):

    plt.figure(figsize=(14, 8))

    sns.heatmap(
        retention_matrix_df,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Retention %"}
    )

    plt.title(
        "Quick-Commerce Cohort Retention Heatmap",
        fontsize=18,
        fontweight="bold"
    )

    plt.ylabel("Acquisition Cohort")
    plt.xlabel("Months Since Acquisition")

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------
plot_retention_heatmap(retention_pct)