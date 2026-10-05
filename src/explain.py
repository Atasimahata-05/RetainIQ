"""
explain.py
----------
SHAP-based model explainability:
  - Global: SHAP summary plot (beeswarm + bar)
  - Individual: waterfall plot for any customer index
  - Churn reason codes: top-3 SHAP features for each high-risk customer
Saves plots to notebooks/.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from feature_engineering import add_features
from preprocessing       import load_and_split, MODELS_DIR

NOTEBOOKS  = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "churn_model.pkl"

# Risk threshold for "high-risk" classification
HIGH_RISK_THRESHOLD = 0.5


def load_bundle():
    return joblib.load(MODEL_PATH)


def get_shap_explainer(bundle, X_transformed: np.ndarray):
    """Return the appropriate SHAP explainer based on model type."""
    model      = bundle["model"]
    model_name = bundle["model_name"]

    if model_name in ("XGBoost", "LightGBM", "RandomForest"):
        explainer = shap.TreeExplainer(model)
    else:
        # Linear model
        explainer = shap.LinearExplainer(model, X_transformed)
    return explainer


def compute_shap_values(explainer, X_transformed: np.ndarray):
    """Compute SHAP values. Returns array of shape (n_samples, n_features)."""
    shap_values = explainer.shap_values(X_transformed)
    # For binary classifiers returning a list of two arrays, take class-1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values


def plot_shap_summary(shap_values, X_transformed, feature_names, n_features=20):
    """Save SHAP beeswarm + bar summary plots."""
    # Beeswarm
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values[:1000],          # sample for speed
        X_transformed[:1000],
        feature_names=feature_names,
        max_display=n_features,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Summary (Beeswarm) – Top Features", fontsize=13)
    plt.tight_layout()
    path_beeswarm = NOTEBOOKS / "shap_summary_beeswarm.png"
    plt.savefig(path_beeswarm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP beeswarm saved → {path_beeswarm}")

    # Bar summary
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values[:1000],
        X_transformed[:1000],
        feature_names=feature_names,
        max_display=n_features,
        show=False,
        plot_type="bar",
    )
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=13)
    plt.tight_layout()
    path_bar = NOTEBOOKS / "shap_summary_bar.png"
    plt.savefig(path_bar, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP bar chart saved → {path_bar}")

    return path_beeswarm, path_bar


def plot_waterfall(explainer, shap_values, X_transformed,
                   feature_names, customer_idx=0):
    """Waterfall plot for a single customer."""
    vals   = shap_values[customer_idx]
    base   = (explainer.expected_value
               if np.isscalar(explainer.expected_value)
               else explainer.expected_value[1])

    # Top contributing features (by |SHAP|)
    top_idx  = np.argsort(np.abs(vals))[::-1][:15]
    top_vals = vals[top_idx]
    top_feat = [feature_names[i] for i in top_idx]

    colours = ["#C0504D" if v > 0 else "#4F81BD" for v in top_vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top_vals)), top_vals[::-1], color=colours[::-1])
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_feat[::-1], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (impact on churn probability)", fontsize=11)
    ax.set_title(f"Customer #{customer_idx} – Churn Explanation", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = NOTEBOOKS / f"shap_waterfall_{customer_idx}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Waterfall saved → {path}")
    return str(path)


def get_churn_reason_codes(shap_values, feature_names,
                           y_proba, threshold=HIGH_RISK_THRESHOLD,
                           top_n=3) -> pd.DataFrame:
    """
    Generate a DataFrame with top churn reason codes for high-risk customers.
    Columns: customer_index, churn_probability, reason_1..reason_n, shap_1..shap_n
    """
    high_risk_idx = np.where(y_proba >= threshold)[0]
    rows = []
    columns = (
        ["customer_index", "churn_probability"]
        + [f"reason_{i}" for i in range(1, top_n + 1)]
        + [f"shap_{i}" for i in range(1, top_n + 1)]
    )
    for idx in high_risk_idx:
        vals      = shap_values[idx]
        top_pos   = np.argsort(vals)[::-1][:top_n]   # top positive SHAP
        reasons   = {f"reason_{i+1}": feature_names[top_pos[i]] for i in range(top_n)}
        shap_vals = {f"shap_{i+1}":   round(float(vals[top_pos[i]]), 4) for i in range(top_n)}
        rows.append({
            "customer_index":    int(idx),
            "churn_probability": round(float(y_proba[idx]), 4),
            **reasons,
            **shap_vals,
        })
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values("churn_probability", ascending=False)


def get_individual_explanation(bundle, customer_features: dict) -> dict:
    """
    Given raw customer feature dict, return churn probability + top SHAP reasons.
    Used internally by the retention engine and API.
    """
    from feature_engineering import add_features
    df_single = pd.DataFrame([customer_features])
    df_fe     = add_features(df_single)

    preprocessor = bundle["preprocessor"]
    X_t          = preprocessor.transform(df_fe)
    model        = bundle["model"]
    proba        = float(model.predict_proba(X_t)[0, 1])

    explainer    = get_shap_explainer(bundle, X_t)
    shap_vals    = compute_shap_values(explainer, X_t)
    feat_names   = bundle["feature_names"]

    # Flatten if 2-D
    vals = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

    top_idx  = np.argsort(np.abs(vals))[::-1][:5]
    top_reasons = [
        {"feature": feat_names[i], "shap_value": round(float(vals[i]), 4)}
        for i in top_idx
    ]

    return {"churn_probability": round(proba, 4), "top_reasons": top_reasons}


def main():
    print("="*60)
    print(" SHAP EXPLAINABILITY")
    print("="*60)

    bundle = load_bundle()
    print(f"\nModel: {bundle['model_name']}")

    # Prepare test data
    _, X_test_raw, _, y_test = load_and_split()
    X_test_fe = add_features(X_test_raw)
    preprocessor = bundle["preprocessor"]
    X_test_t = preprocessor.transform(X_test_fe)
    y_proba  = bundle["model"].predict_proba(X_test_t)[:, 1]

    feature_names = bundle["feature_names"]

    print("\nComputing SHAP values (this may take a moment)...")
    explainer  = get_shap_explainer(bundle, X_test_t)
    shap_vals  = compute_shap_values(explainer, X_test_t)

    print(f"  SHAP values shape: {shap_vals.shape}")

    # Global plots
    plot_shap_summary(shap_vals, X_test_t, feature_names)

    # Individual waterfall for first high-risk customer
    high_risk_idx = np.where(y_proba >= HIGH_RISK_THRESHOLD)[0]
    if len(high_risk_idx) > 0:
        plot_waterfall(explainer, shap_vals, X_test_t,
                       feature_names, customer_idx=high_risk_idx[0])

    # Churn reason codes
    reason_df = get_churn_reason_codes(shap_vals, feature_names, y_proba)
    out = NOTEBOOKS / "churn_reason_codes.csv"
    reason_df.to_csv(out, index=False)
    print(f"\n  Churn reason codes saved → {out}")
    print(f"  High-risk customers: {len(reason_df)}")
    print("\nTop 5 high-risk customers:")
    print(reason_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
