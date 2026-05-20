import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Retention Intelligence Engine",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("user_transactions.csv")

df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Executive Dashboard",
        "Cohort Retention",
        "Churn Detection",
        "AI CRM Generator"
    ]
)

# =====================================================
# EXECUTIVE DASHBOARD
# =====================================================

if page == "Executive Dashboard":

    st.title("🛒 AI Retention Intelligence Engine")

    st.markdown("""
    AI-native growth analytics platform simulating
    quick-commerce retention intelligence,
    churn prediction, and lifecycle CRM orchestration.
    """)

    total_users = df["user_id"].nunique()
    total_orders = len(df)
    total_gmv = df["order_amount"].sum()
    avg_order_value = df["order_amount"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Users", f"{total_users:,}")
    col2.metric("Orders", f"{total_orders:,}")
    col3.metric("GMV", f"${total_gmv:,.0f}")
    col4.metric("AOV", f"${avg_order_value:.2f}")

    st.divider()

    # -------------------------------------------------
    # ORDER DISTRIBUTION
    # -------------------------------------------------

    st.subheader("Order Amount Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        df["order_amount"],
        bins=40,
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------------------------------
    # CATEGORY WIDTH DISTRIBUTION
    # -------------------------------------------------

    st.subheader("Category Width Distribution")

    category_counts = (
        df["product_category_width"]
        .value_counts()
        .sort_index()
    )

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=category_counts.index,
        y=category_counts.values,
        ax=ax2
    )

    st.pyplot(fig2)

# =====================================================
# COHORT RETENTION PAGE
# =====================================================

elif page == "Cohort Retention":

    st.title("📈 Cohort Retention Heatmap")

    df["order_month"] = df["order_timestamp"].dt.to_period("M")

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

    df["cohort_index"] = (
        (df["order_month"].dt.year - df["cohort_month"].dt.year) * 12
        + (df["order_month"].dt.month - df["cohort_month"].dt.month)
    )

    cohort_data = (
        df.groupby(["cohort_month", "cohort_index"])
        ["user_id"]
        .nunique()
        .reset_index()
    )

    cohort_matrix = cohort_data.pivot(
        index="cohort_month",
        columns="cohort_index",
        values="user_id"
    )

    cohort_size = cohort_matrix.iloc[:, 0]

    retention_matrix = (
        cohort_matrix.divide(cohort_size, axis=0) * 100
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        retention_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)

# =====================================================
# CHURN DETECTION
# =====================================================

elif page == "Churn Detection":

    st.title("⚠️ Churn Risk Detection")

    latest_date = df["order_timestamp"].max()

    user_last_order = (
        df.groupby("user_id")["order_timestamp"]
        .max()
        .reset_index()
    )

    user_last_order["days_since_last_order"] = (
        latest_date - user_last_order["order_timestamp"]
    ).dt.days

    at_risk_users = user_last_order[
        user_last_order["days_since_last_order"] > 14
    ]

    st.metric(
        "At-Risk Users",
        f"{len(at_risk_users):,}"
    )

    st.dataframe(
        at_risk_users.sort_values(
            "days_since_last_order",
            ascending=False
        ).head(20)
    )

# =====================================================
# AI CRM GENERATOR
# =====================================================

elif page == "AI CRM Generator":

    st.title("🤖 AI Lifecycle CRM Generator")

    lifecycle_stage = st.selectbox(
        "Lifecycle Stage",
        [
            "Onboarding Friction",
            "Habit Formation",
            "Churn Risk",
            "Win Back"
        ]
    )

    category_width = st.slider(
        "Category Width",
        1,
        10,
        3
    )

    inactivity_days = st.slider(
        "Days Since Last Order",
        1,
        45,
        12
    )

    if st.button("Generate CRM Copy"):

        push = (
            f"Your next smart restock is waiting 🛒"
        )

        whatsapp = (
            f"You've explored {category_width} categories — "
            f"ready to try fresh staples and breakfast essentials?"
        )

        email = (
            "Your personalized convenience basket is ready"
        )

        st.success(f"📲 Push: {push}")

        st.info(f"💬 WhatsApp: {whatsapp}")

        st.warning(f"📧 Email: {email}")