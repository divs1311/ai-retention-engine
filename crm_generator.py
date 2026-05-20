# ============================================================
# CHURN-RISK + LLM CRM GENERATOR
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime

# ------------------------------------------------------------
# LOAD FILES
# ------------------------------------------------------------
transactions = pd.read_csv("user_transactions.csv")
retention = pd.read_csv(
    "cohort_retention_pct.csv",
    index_col=0
)

transactions["order_timestamp"] = pd.to_datetime(
    transactions["order_timestamp"]
)

# ------------------------------------------------------------
# RISK DETECTION LOGIC
# ------------------------------------------------------------
def detect_at_risk_segments(retention_df, transactions_df):

    risk_segments = []

    # -----------------------------------------
    # Cohort-level retention collapse
    # -----------------------------------------
    for cohort in retention_df.index:

        row = retention_df.loc[cohort]

        try:
            m2 = row["2"]
            m3 = row["3"]

            if (m2 - m3) > 15:

                risk_segments.append({
                    "risk_type": "retention_drop",
                    "cohort": cohort,
                    "month2_retention": round(m2, 1),
                    "month3_retention": round(m3, 1),
                    "drop_pct": round(m2 - m3, 1)
                })

        except:
            continue

    # -----------------------------------------
    # User-level inactivity
    # -----------------------------------------
    latest_date = transactions_df["order_timestamp"].max()

    user_last_order = (
        transactions_df.groupby("user_id")
        ["order_timestamp"]
        .max()
        .reset_index()
    )

    user_last_order["days_since_last_order"] = (
        latest_date - user_last_order["order_timestamp"]
    ).dt.days

    inactive_users = user_last_order[
        user_last_order["days_since_last_order"] > 14
    ]

    category_width = (
        transactions_df.groupby("user_id")
        ["product_category_width"]
        .max()
        .reset_index()
    )

    inactive_users = inactive_users.merge(
        category_width,
        on="user_id",
        how="left"
    )

    for _, row in inactive_users.iterrows():

        risk_segments.append({
            "risk_type": "inactive_user",
            "user_id": int(row["user_id"]),
            "days_inactive": int(row["days_since_last_order"]),
            "category_width": int(row["product_category_width"])
        })

    return risk_segments

# ------------------------------------------------------------
# LLM CRM GENERATION ENGINE
# ------------------------------------------------------------
def generate_lifecycle_crm_campaign(at_risk_cohort_metadata):

    campaigns = []

    for risk in at_risk_cohort_metadata:

        # =====================================================
        # LIFECYCLE STAGE MAPPING
        # =====================================================
        if risk["risk_type"] == "retention_drop":

            lifecycle_stage = "Habit Formation"

            prompt = f"""
You are an elite CRM strategist at an AI-native quick-commerce startup.

Generate:
1. Push notification
2. WhatsApp copy
3. Email subject line

Context:
- Cohort retention dropped from {risk['month2_retention']}%
  to {risk['month3_retention']}%
- This indicates weak habit formation.
- Goal is to increase repeat order frequency.
- Encourage cross-category expansion.
- Avoid generic discounts and spam language.
- Use emotional resonance and convenience psychology.
- Suggest expanding from fruits/snacks into staples.

Tone:
Premium, warm, intelligent, habit-forming.
"""

        elif risk["risk_type"] == "inactive_user":

            lifecycle_stage = "Churn Mitigation"

            prompt = f"""
You are an elite retention marketer.

Generate:
1. Push notification
2. WhatsApp copy
3. Email subject line

Context:
- User inactive for {risk['days_inactive']} days.
- Current category width:
  {risk['category_width']} categories.
- Goal:
  Reactivate purchasing behavior.
- Encourage wider basket composition.
- Recommend staples, breakfast essentials,
  and impulse add-ons.
- Avoid discount addiction mechanics.

Tone:
Personalized, emotionally intelligent,
high-convenience quick-commerce positioning.
"""

        else:

            lifecycle_stage = "Win Back"

            prompt = """
Generate intelligent lifecycle CRM copy.
"""

        # =====================================================
        # MOCK LLM OUTPUT
        # Replace with actual OpenAI / Anthropic API call
        # =====================================================

        simulated_llm_response = {
            "lifecycle_stage": lifecycle_stage,

            "push_notification":
                "Your kitchen might be missing something fresh today 👀",

            "whatsapp":
                "You stocked fruits last week — "
                "want to complete the basket with breakfast staples "
                "and organic essentials in 10 mins?",

            "email_subject":
                "Your smart restock is waiting 🛒"
        }

        campaigns.append({
            "risk_metadata": risk,
            "prompt_used": prompt,
            "generated_campaign": simulated_llm_response
        })

    return campaigns

# ------------------------------------------------------------
# PIPELINE EXECUTION
# ------------------------------------------------------------
risk_segments = detect_at_risk_segments(
    retention,
    transactions
)

crm_campaigns = generate_lifecycle_crm_campaign(
    risk_segments
)

# ------------------------------------------------------------
# PREVIEW OUTPUT
# ------------------------------------------------------------
for idx, campaign in enumerate(crm_campaigns[:5]):

    print("\n" + "=" * 60)
    print(f"CAMPAIGN #{idx+1}")
    print("=" * 60)

    print("\nRISK METADATA")
    print(campaign["risk_metadata"])

    print("\nPUSH")
    print(campaign["generated_campaign"]["push_notification"])

    print("\nWHATSAPP")
    print(campaign["generated_campaign"]["whatsapp"])

    print("\nEMAIL")
    print(campaign["generated_campaign"]["email_subject"])