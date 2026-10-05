"""
feature_engineering.py
-----------------------
Creates derived features from raw customer data:
  - tenure_bucket        : New / Growing / Mature / Loyal
  - charges_per_month    : total_charges / tenure_months
  - usage_trend          : normalised composite of login + session activity
  - engagement_score     : feature_usage × login normalised
  - support_load         : tickets weighted by avg resolution time
  - payment_reliability  : inverse payment failure score
  - recency_score        : 1 / (1 + days_since_last_login)
  - loyalty_score        : tenure × referrals × auto_pay bonus
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a copy of df.
    Input df must contain the raw columns from churn_data.csv
    (minus customer_id / signup_date if already dropped).
    Returns a new DataFrame.
    """
    df = df.copy()

    # ── 1. Tenure bucket ──────────────────────────────────────────────────────
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins    = [0, 6, 24, 48, 9999],
        labels  = ["New", "Growing", "Mature", "Loyal"],
        right   = True,
    ).astype(str)

    # ── 2. Charges per month ──────────────────────────────────────────────────
    # Avoids division by zero for tenure = 0
    df["charges_per_month"] = df["total_charges"] / df["tenure_months"].clip(lower=1)

    # ── 3. Usage trend ────────────────────────────────────────────────────────
    # Normalise login count (0-60) and session minutes (rough cap at 2000)
    login_norm   = df["login_count_30d"].clip(0, 60) / 60
    session_norm = df["session_minutes_30d"].fillna(0).clip(0, 2000) / 2000
    df["usage_trend"] = ((login_norm + session_norm) / 2).round(4)

    # ── 4. Engagement score ───────────────────────────────────────────────────
    feature_norm = df["feature_usage_score"].fillna(50) / 100
    df["engagement_score"] = ((feature_norm + login_norm) / 2 * 100).round(2)

    # ── 5. Support load ───────────────────────────────────────────────────────
    # More tickets AND longer resolution → higher load
    tickets  = df["num_support_tickets_30d"].fillna(0).clip(0, 15)
    res_time = df["avg_resolution_time_hours"].fillna(df["avg_resolution_time_hours"].median()).clip(0, 96)
    df["support_load"] = (tickets * np.log1p(res_time)).round(4)

    # ── 6. Payment reliability ────────────────────────────────────────────────
    # 0 failures → 1.0 (fully reliable); 5+ failures → ~0.0
    failures = df["num_payment_failures_6m"].fillna(0).clip(0, 5)
    df["payment_reliability"] = (1 - failures / 5).round(4)

    # ── 7. Recency score ──────────────────────────────────────────────────────
    # Recent login → score near 1; stale → near 0
    days = df["days_since_last_login"].fillna(30).clip(0, 180)
    df["recency_score"] = (1 / (1 + days / 30)).round(4)

    # ── 8. Loyalty score ─────────────────────────────────────────────────────
    # Long tenure, referrals, and auto-pay all reduce churn risk
    tenure_norm   = df["tenure_months"].clip(0, 72) / 72
    referral_norm = df["referral_count"].clip(0, 10) / 10
    autopay_bonus = df["auto_pay"].fillna(0)
    df["loyalty_score"] = (
        (tenure_norm * 0.5 + referral_norm * 0.3 + autopay_bonus * 0.2) * 100
    ).round(2)

    return df


def get_engineered_feature_names() -> list:
    """Return list of all engineered feature column names."""
    return [
        "tenure_bucket",
        "charges_per_month",
        "usage_trend",
        "engagement_score",
        "support_load",
        "payment_reliability",
        "recency_score",
        "loyalty_score",
    ]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_generation import generate_churn_data

    df = generate_churn_data(n=1000)
    df.drop(columns=["customer_id", "signup_date", "churn"], inplace=True)
    df_fe = add_features(df)

    engineered = get_engineered_feature_names()
    print("Engineered features sample:")
    print(df_fe[engineered].describe().round(3))
    print(f"\nNew columns: {engineered}")
