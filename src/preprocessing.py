"""
preprocessing.py
----------------
Builds a reusable scikit-learn ColumnTransformer pipeline:
  - Numeric: median imputation → StandardScaler
  - Categorical: most-frequent imputation → OneHotEncoder
Provides train/test split and SMOTE utility for imbalanced learning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = Path(__file__).parent.parent / "data" / "churn_data.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Column groups ─────────────────────────────────────────────────────────────
# These are the RAW columns (before feature engineering)
NUMERIC_COLS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets_30d",
    "avg_resolution_time_hours",
    "satisfaction_score",
    "login_count_30d",
    "session_minutes_30d",
    "feature_usage_score",
    "days_since_last_login",
    "num_payment_failures_6m",
    "referral_count",
    # Engineered numeric features (added after feature_engineering.py)
    "charges_per_month",
    "usage_trend",
    "engagement_score",
    "support_load",
    "payment_reliability",
    "recency_score",
    "loyalty_score",
]

CATEGORICAL_COLS = [
    "plan_type",
    "contract_type",
    "payment_method",
    "tenure_bucket",   # engineered
]

BINARY_COLS = [
    "auto_pay",
    "discount_applied",
    "price_increase_last_3m",
]

TARGET_COL = "churn"

# Columns to DROP before modelling (identifiers / dates / leak)
DROP_COLS = ["customer_id", "signup_date"]


def build_preprocessor(numeric_cols=None, categorical_cols=None):
    """
    Build and return a ColumnTransformer that handles:
      - numeric: median imputation + standard scaling
      - categorical: mode imputation + one-hot encoding
      - binary: median imputation (no scaling needed for tree models,
                 but StandardScaler applied for LR compatibility)
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler",  StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipeline,     numeric_cols),
            ("cat",  categorical_pipeline, categorical_cols),
            ("bin",  binary_pipeline,      BINARY_COLS),
        ],
        remainder="drop",   # drop customer_id, signup_date, etc.
        verbose_feature_names_out=True,
    )
    return preprocessor


def load_and_split(data_path=DATA_PATH, test_size=0.20, seed=42):
    """
    Load CSV, drop unused columns, split into train/test.
    Returns: X_train, X_test, y_train, y_test, feature_cols
    """
    df = pd.read_csv(data_path)

    # Drop identifier / date columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train_transformed, y_train, seed=42):
    """
    Apply SMOTE to the already-transformed training data.
    Returns resampled X, y as numpy arrays.
    """
    sm = SMOTE(random_state=seed, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train_transformed, y_train)
    print(f"  [SMOTE] Before: {y_train.value_counts().to_dict()}")
    print(f"  [SMOTE] After : {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


def get_feature_names(preprocessor):
    """Extract human-readable feature names from the fitted ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())


if __name__ == "__main__":
    # Smoke test
    from feature_engineering import add_features

    X_train, X_test, y_train, y_test = load_and_split()
    X_train = add_features(X_train)
    X_test  = add_features(X_test)

    preproc = build_preprocessor()
    X_tr = preproc.fit_transform(X_train)
    X_te = preproc.transform(X_test)

    print(f"X_train transformed shape : {X_tr.shape}")
    print(f"X_test  transformed shape : {X_te.shape}")
    print(f"Churn rate (train) : {y_train.mean():.1%}")
    print(f"Churn rate (test)  : {y_test.mean():.1%}")

    joblib.dump(preproc, MODELS_DIR / "preprocessor.pkl")
    print("Preprocessor saved to models/preprocessor.pkl")
