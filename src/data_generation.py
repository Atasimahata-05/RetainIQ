"""
data_generation.py
------------------
Generates a synthetic customer churn dataset with 20,000 records.
Realistic churn logic, noise, missing values, outliers, and seasonality.
Saves to data/churn_data.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

N = 20_000          # total customers
CHURN_RATE = 0.20   # target ~20%

# ── Output path ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT = DATA_DIR / "churn_data.csv"


def generate_churn_data(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Return a DataFrame of synthetic customer churn data."""
    rng = np.random.default_rng(seed)

    # ── Demographics / Account ────────────────────────────────────────────────
    customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(1, n + 1)]

    # Tenures: skewed – many new customers, some long-term
    # Build segments and shuffle to avoid ordered bias
    seg1 = rng.integers(1,  6,  size=n // 3)       # new
    seg2 = rng.integers(6,  24, size=n // 3)        # growing
    seg3 = rng.integers(24, 72, size=n - 2*(n//3))  # mature/loyal
    tenure_months = np.concatenate([seg1, seg2, seg3])
    rng.shuffle(tenure_months)
    tenure_months = np.clip(tenure_months, 1, 72).astype(int)

    plan_type = rng.choice(["Basic", "Standard", "Premium"],
                           size=n, p=[0.30, 0.45, 0.25])

    # Monthly charges based on plan + noise
    base_charges = {"Basic": 29.99, "Standard": 59.99, "Premium": 99.99}
    monthly_charges = np.array([base_charges[p] for p in plan_type])
    monthly_charges += rng.normal(0, 5, size=n)
    monthly_charges = np.clip(monthly_charges, 15, 150).round(2)

    total_charges = (monthly_charges * tenure_months
                     + rng.normal(0, 20, size=n)).round(2)
    total_charges = np.clip(total_charges, 0, None)

    # ── Contract / Payment ─────────────────────────────────────────────────────
    contract_type = rng.choice(["Monthly", "Yearly"],
                               size=n, p=[0.55, 0.45])

    payment_method = rng.choice(["Card", "UPI", "Bank", "Wallet"],
                                size=n, p=[0.35, 0.30, 0.20, 0.15])

    auto_pay = rng.choice([0, 1], size=n, p=[0.35, 0.65]).astype(int)

    # ── Support ────────────────────────────────────────────────────────────────
    # Higher tickets for dissatisfied / churning customers (added later via correlation)
    num_support_tickets_30d = rng.negative_binomial(2, 0.6, size=n)
    num_support_tickets_30d = np.clip(num_support_tickets_30d, 0, 15).astype(int)

    avg_resolution_time_hours = rng.gamma(3, 8, size=n).round(1)
    avg_resolution_time_hours = np.clip(avg_resolution_time_hours, 0.5, 96)

    satisfaction_score = rng.choice([1, 2, 3, 4, 5],
                                    size=n, p=[0.10, 0.15, 0.30, 0.30, 0.15])

    # ── Engagement / Usage ─────────────────────────────────────────────────────
    login_count_30d = rng.poisson(12, size=n)
    login_count_30d = np.clip(login_count_30d, 0, 60).astype(int)

    session_minutes_30d = (login_count_30d * rng.uniform(10, 45, size=n)).round(1)

    feature_usage_score = rng.beta(2, 3, size=n) * 100
    feature_usage_score = feature_usage_score.round(1)

    days_since_last_login = rng.exponential(15, size=n).round(0).astype(int)
    days_since_last_login = np.clip(days_since_last_login, 0, 180)

    # ── Offers / Financials ────────────────────────────────────────────────────
    discount_applied = rng.choice([0, 1], size=n, p=[0.60, 0.40]).astype(int)

    price_increase_last_3m = rng.choice([0, 1], size=n, p=[0.75, 0.25]).astype(int)

    num_payment_failures_6m = rng.choice(
        [0, 1, 2, 3, 4, 5],
        size=n, p=[0.55, 0.20, 0.12, 0.07, 0.04, 0.02]
    ).astype(int)

    referral_count = rng.choice(range(0, 11), size=n,
                                p=[0.40, 0.20, 0.12, 0.08,
                                   0.06, 0.04, 0.04, 0.02,
                                   0.02, 0.01, 0.01]).astype(int)

    # ── Signup date (last 6 years, corr with tenure) ───────────────────────────
    reference_date = pd.Timestamp("2026-09-24")
    signup_dates = [
        (reference_date - pd.DateOffset(months=int(t))).strftime("%Y-%m-%d")
        for t in tenure_months
    ]

    # ── Churn probability (logistic-style combination) ─────────────────────────
    # Normalise each factor to [0, 1] contributing to churn risk
    contract_risk    = np.where(contract_type == "Monthly", 0.30, 0.0)
    charges_risk     = (monthly_charges - 15) / (150 - 15) * 0.20
    usage_risk       = (1 - login_count_30d / 60) * 0.15
    support_risk     = np.clip(num_support_tickets_30d / 15, 0, 1) * 0.15
    satisfaction_risk= (5 - satisfaction_score) / 4 * 0.12
    payment_risk     = np.clip(num_payment_failures_6m / 5, 0, 1) * 0.10
    price_risk       = price_increase_last_3m * 0.10
    recency_risk     = np.clip(days_since_last_login / 180, 0, 1) * 0.08
    autopay_risk     = np.where(auto_pay == 0, 0.05, 0.0)
    discount_protect = discount_applied * (-0.08)

    base_churn_prob = (
        contract_risk + charges_risk + usage_risk + support_risk
        + satisfaction_risk + payment_risk + price_risk
        + recency_risk + autopay_risk + discount_protect
    )

    # Add Gaussian noise
    base_churn_prob += rng.normal(0, 0.06, size=n)
    base_churn_prob = np.clip(base_churn_prob, 0.01, 0.98)

    # ── Seasonality: higher churn at month-end billing (simulate) ─────────────
    month_day = rng.integers(1, 29, size=n)   # billing day
    seasonal_bump = np.where(month_day >= 25, 0.05, 0.0)
    base_churn_prob = np.clip(base_churn_prob + seasonal_bump, 0.01, 0.98)

    # ── Sample binary churn label ──────────────────────────────────────────────
    churn = rng.binomial(1, base_churn_prob, size=n).astype(int)

    # Calibrate to target churn rate via rejection sampling tweak
    current_rate = churn.mean()
    if current_rate < 0.15:
        extra_idx = rng.choice(np.where(churn == 0)[0],
                               size=int((0.18 - current_rate) * n), replace=False)
        churn[extra_idx] = 1
    elif current_rate > 0.27:
        remove_idx = rng.choice(np.where(churn == 1)[0],
                                size=int((current_rate - 0.22) * n), replace=False)
        churn[remove_idx] = 0

    # ── Assemble DataFrame ─────────────────────────────────────────────────────
    df = pd.DataFrame({
        "customer_id":               customer_ids,
        "signup_date":               signup_dates,
        "tenure_months":             tenure_months,
        "plan_type":                 plan_type,
        "monthly_charges":           monthly_charges,
        "total_charges":             total_charges,
        "contract_type":             contract_type,
        "payment_method":            payment_method,
        "auto_pay":                  auto_pay,
        "num_support_tickets_30d":   num_support_tickets_30d,
        "avg_resolution_time_hours": avg_resolution_time_hours,
        "satisfaction_score":        satisfaction_score,
        "login_count_30d":           login_count_30d,
        "session_minutes_30d":       session_minutes_30d,
        "feature_usage_score":       feature_usage_score,
        "days_since_last_login":     days_since_last_login,
        "discount_applied":          discount_applied,
        "price_increase_last_3m":    price_increase_last_3m,
        "num_payment_failures_6m":   num_payment_failures_6m,
        "referral_count":            referral_count,
        "churn":                     churn,
    })

    # ── Add outliers (1-2%) ───────────────────────────────────────────────────
    outlier_idx = rng.choice(n, size=int(n * 0.015), replace=False)
    df.loc[outlier_idx, "monthly_charges"]           *= rng.uniform(1.5, 3, size=len(outlier_idx))
    df.loc[outlier_idx, "num_support_tickets_30d"]   = rng.integers(10, 30, size=len(outlier_idx))
    df.loc[outlier_idx, "session_minutes_30d"]       *= rng.uniform(0, 0.1, size=len(outlier_idx))

    # ── Inject Missing Values (~3-5% per column) ──────────────────────────────
    cols_with_na = [
        "avg_resolution_time_hours", "satisfaction_score",
        "session_minutes_30d", "feature_usage_score",
        "days_since_last_login", "num_payment_failures_6m",
    ]
    for col in cols_with_na:
        mask = rng.choice([True, False], size=n, p=[0.04, 0.96])
        df.loc[mask, col] = np.nan

    return df


def main():
    print("Generating synthetic churn dataset...")
    df = generate_churn_data()

    # Summary statistics
    print(f"  Total customers  : {len(df):,}")
    print(f"  Churn rate       : {df['churn'].mean():.1%}")
    print(f"  Missing values   : {df.isnull().sum().sum():,}")

    df.to_csv(OUTPUT, index=False)
    print(f"  Saved to         : {OUTPUT}")


if __name__ == "__main__":
    main()
