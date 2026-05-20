import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Retention Intelligence Engine",
    layout="wide"
)

st.title("🛒 AI Retention Intelligence Engine")

df = pd.read_csv("user_transactions.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Core Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Users", df["user_id"].nunique())
col2.metric("Orders", len(df))
col3.metric("GMV", f"${df['order_amount'].sum():,.0f}")

st.subheader("Order Amount Distribution")

fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(df["order_amount"], bins=40, ax=ax)

st.pyplot(fig)