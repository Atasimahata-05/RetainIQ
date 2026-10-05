"""
evaluate.py
-----------
Loads the trained model bundle from models/churn_model.pkl and the
held-out test set, then computes:
  - Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
  - Confusion matrix
  - ROC curve  → notebooks/roc_curve.png
  - PR curve   → notebooks/pr_curve.png
  - Lift chart → notebooks/lift_chart.png
  - Decile analysis table
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)

from feature_engineering import add_features
from preprocessing       import load_and_split, MODELS_DIR

NOTEBOOKS = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "churn_model.pkl"


def load_bundle_and_test():
    """Load model bundle and prepare test data."""
    bundle = joblib.load(MODEL_PATH)
    model        = bundle["model"]
    preprocessor = bundle["preprocessor"]
    threshold    = bundle.get("threshold", 0.5)

    _, X_test_raw, _, y_test = load_and_split()
    X_test_fe = add_features(X_test_raw)
    X_test_t  = preprocessor.transform(X_test_fe)

    y_proba = model.predict_proba(X_test_t)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    return y_test, y_pred, y_proba, bundle


def print_metrics(y_test, y_pred, y_proba):
    """Print classification metrics to console."""
    print("="*60)
    print(" EVALUATION METRICS")
    print("="*60)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1        : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  PR-AUC    : {average_precision_score(y_test, y_proba):.4f}")
    print()
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"  FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["No Churn", "Churn"]))

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc":    round(average_precision_score(y_test, y_proba), 4),
    }


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#4F81BD", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve – Churn Prediction", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = NOTEBOOKS / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved → {path}")


def plot_pr_curve(y_test, y_proba):
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="#C0504D", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.axhline(y=y_test.mean(), color="k", ls="--", lw=1, label="Baseline")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve – Churn Prediction", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = NOTEBOOKS / "pr_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  PR  curve saved → {path}")


def decile_analysis(y_test, y_proba) -> pd.DataFrame:
    """
    Compute lift by decile.
    Returns a DataFrame with decile stats.
    Also saves a lift chart PNG.
    """
    df = pd.DataFrame({"actual": y_test.values, "proba": y_proba})
    df = df.sort_values("proba", ascending=False).reset_index(drop=True)
    n  = len(df)
    if n == 0:
        raise ValueError("Cannot calculate deciles for an empty test set.")
    df["decile"] = np.clip(np.ceil((df.index + 1) / n * 10).astype(int), 1, 10)

    overall_rate = y_test.mean()

    decile_df = (
        df.groupby("decile")["actual"]
        .agg(customers="count", churners="sum")
        .assign(churn_rate=lambda d: d["churners"] / d["customers"])
        .assign(lift=lambda d: d["churn_rate"] / overall_rate)
        .assign(cumulative_churn_rate=lambda d:
                d["churners"].cumsum() / d["customers"].cumsum())
    )

    print("\nDecile Analysis:")
    print(decile_df.round(3).to_string())

    # Lift chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(decile_df.index, decile_df["lift"],
           color="#9BBB59", edgecolor="white")
    ax.axhline(y=1, color="red", ls="--", lw=1.5, label="Baseline (Lift=1)")
    ax.set_xlabel("Decile (1 = highest risk)", fontsize=12)
    ax.set_ylabel("Lift", fontsize=12)
    ax.set_title("Lift Chart by Decile", fontsize=14)
    ax.set_xticks(decile_df.index)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = NOTEBOOKS / "lift_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Lift chart saved → {path}")

    return decile_df


def main():
    y_test, y_pred, y_proba, bundle = load_bundle_and_test()
    print(f"\nModel: {bundle['model_name']}")

    metrics = print_metrics(y_test, y_pred, y_proba)

    print("\nGenerating plots...")
    plot_roc_curve(y_test, y_proba)
    plot_pr_curve(y_test, y_proba)
    decile_df = decile_analysis(y_test, y_proba)
    decile_df.to_csv(NOTEBOOKS / "decile_analysis.csv")

    # Save metrics JSON
    import json
    (MODELS_DIR / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print("\nMetrics saved to models/eval_metrics.json")


if __name__ == "__main__":
    main()
