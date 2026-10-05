"""
train.py
--------
Trains four churn prediction models with StratifiedKFold CV:
  1. Logistic Regression  (baseline)
  2. Random Forest
  3. XGBoost
  4. LightGBM

Class imbalance handled via estimator-specific class weights.
Best model (by ROC-AUC) saved to models/churn_model.pkl.
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import roc_auc_score, recall_score, classification_report
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier

from feature_engineering import add_features
from preprocessing       import (
    load_and_split, build_preprocessor,
    get_feature_names, MODELS_DIR
)

SEED   = 42
N_FOLDS = 5

# ── Model definitions ─────────────────────────────────────────────────────────
def get_models(scale_pos_weight: float = 4.0):
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
            C=0.5,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=SEED,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            is_unbalance=True,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbose=-1,
        ),
    }


def train_with_cv(name, model, X_train_raw, y_train,
                  preprocessor, n_folds=N_FOLDS):
    """Run StratifiedKFold CV using each estimator's class-weight strategy."""
    skf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    aucs, recalls = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train), 1):
        X_tr_raw = X_train_raw.iloc[tr_idx]
        y_tr     = y_train.iloc[tr_idx]
        X_val_raw = X_train_raw.iloc[val_idx]
        y_val     = y_train.iloc[val_idx]

        # Feature engineering per fold
        X_tr_fe  = add_features(X_tr_raw)
        X_val_fe = add_features(X_val_raw)

        # Fit preprocessor on training fold only → transform both
        preproc_fold = build_preprocessor()
        X_tr_t  = preproc_fold.fit_transform(X_tr_fe)
        X_val_t = preproc_fold.transform(X_val_fe)

        model.fit(X_tr_t, y_tr)
        proba = model.predict_proba(X_val_t)[:, 1]
        preds = (proba >= 0.4).astype(int)   # lower threshold → higher recall

        auc = roc_auc_score(y_val, proba)
        rec = recall_score(y_val, preds, zero_division=0)
        aucs.append(auc)
        recalls.append(rec)
        print(f"    Fold {fold}: ROC-AUC={auc:.4f}  Recall={rec:.4f}")

    mean_auc    = np.mean(aucs)
    mean_recall = np.mean(recalls)
    print(f"  >> {name}: mean AUC={mean_auc:.4f}  mean Recall={mean_recall:.4f}\n")
    return mean_auc, mean_recall


def main():
    print("="*60)
    print(" CHURN PREDICTION – MODEL TRAINING")
    print("="*60)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = load_and_split()
    print(f"\nTrain size: {len(X_train_raw):,}  |  Test size: {len(X_test_raw):,}")
    print(f"Churn rate - Train: {y_train.mean():.1%}  Test: {y_test.mean():.1%}\n")

    # Class imbalance weight for XGBoost
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = round(neg / pos, 2)
    print(f"scale_pos_weight = {scale_pos_weight}\n")

    models   = get_models(scale_pos_weight)
    cv_scores = {}

    # ── Cross-validation ─────────────────────────────────────────────────────
    for name, model in models.items():
        print(f"[{name}] - StratifiedKFold CV")
        auc, rec = train_with_cv(name, model, X_train_raw, y_train,
                                 build_preprocessor())
        cv_scores[name] = {"roc_auc": auc, "recall": rec}

    # ── Pick best model by ROC-AUC ────────────────────────────────────────────
    best_name = max(cv_scores, key=lambda k: cv_scores[k]["roc_auc"])
    print(f"\n✅ Best model: {best_name}  (ROC-AUC={cv_scores[best_name]['roc_auc']:.4f})")

    # ── Retrain best model on full training set ───────────────────────────────
    print("\nRetraining best model on full training set...")
    X_train_fe = add_features(X_train_raw)
    X_test_fe  = add_features(X_test_raw)

    final_preprocessor = build_preprocessor()
    X_train_t = final_preprocessor.fit_transform(X_train_fe)
    X_test_t  = final_preprocessor.transform(X_test_fe)

    final_model = get_models(scale_pos_weight)[best_name]
    final_model.fit(X_train_t, y_train)

    # Quick hold-out evaluation
    y_proba = final_model.predict_proba(X_test_t)[:, 1]
    y_pred  = (y_proba >= 0.4).astype(int)
    print(f"\nHold-out ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Hold-out Recall  : {recall_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ── Save artefacts ────────────────────────────────────────────────────────
    feature_names = get_feature_names(final_preprocessor)

    bundle = {
        "model":          final_model,
        "preprocessor":   final_preprocessor,
        "model_name":     best_name,
        "feature_names":  feature_names,
        "cv_scores":      cv_scores,
        "threshold":      0.4,
    }
    out_path = MODELS_DIR / "churn_model.pkl"
    joblib.dump(bundle, out_path)
    print(f"\n💾 Model bundle saved to {out_path}")

    # Save CV results to JSON for reference
    (MODELS_DIR / "cv_results.json").write_text(
        json.dumps(cv_scores, indent=2)
    )
    print(f"   CV results saved to models/cv_results.json")


if __name__ == "__main__":
    main()
