"""
api.py
------
FastAPI REST API for real-time churn prediction.

Endpoints:
  GET  /health           → service health check
  POST /predict          → churn probability + risk level + reasons + offer
  GET  /customers/{id}   → fetch customer record from SQLite
  GET  /customers        → paginated list of customers with churn probabilities
"""

import sys, json, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from fastapi            import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic           import BaseModel, Field
from typing             import Optional, List

from feature_engineering import add_features
from retention_engine    import recommend, _classify_risk

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
DATA_PATH  = ROOT / "data"   / "churn_data.csv"
DB_PATH    = ROOT / "data"   / "churn.db"

# ── Load model bundle on first scoring request ────────────────────────────────
bundle = None


def get_bundle():
    """Load the trained artifact lazily so health/docs work before training."""
    global bundle
    if bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model artifact is missing. Run `python src/train.py` first.",
            )
        bundle = joblib.load(MODEL_PATH)
    return bundle

# ── SQLite init ───────────────────────────────────────────────────────────────
def init_db():
    """Load churn_data.csv into an in-process SQLite DB for demo lookups."""
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Customer dataset is missing. Run `python src/data_generation.py` first.",
        )
    df = pd.read_csv(DATA_PATH)
    con = sqlite3.connect(DB_PATH)
    df.to_sql("customers", con, if_exists="replace", index=False)
    con.close()


def ensure_db():
    """Create the demo database on demand, not as an import-time side effect."""
    if not DB_PATH.exists():
        init_db()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RetainIQ API",
    description="RetainIQ predicts customer churn risk and recommends personalised retention offers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────
class CustomerIn(BaseModel):
    customer_id:               Optional[str] = "UNKNOWN"
    tenure_months:             int   = Field(..., ge=0)
    plan_type:                 str   = Field(..., pattern="^(Basic|Standard|Premium)$")
    monthly_charges:           float = Field(..., gt=0)
    total_charges:             float = Field(..., ge=0)
    contract_type:             str   = Field(..., pattern="^(Monthly|Yearly)$")
    payment_method:            str   = Field(..., pattern="^(Card|UPI|Bank|Wallet)$")
    auto_pay:                  int   = Field(..., ge=0, le=1)
    num_support_tickets_30d:   int   = Field(..., ge=0)
    avg_resolution_time_hours: Optional[float] = None
    satisfaction_score:        Optional[int]   = Field(None, ge=1, le=5)
    login_count_30d:           int   = Field(..., ge=0)
    session_minutes_30d:       Optional[float] = None
    feature_usage_score:       Optional[float] = Field(None, ge=0, le=100)
    days_since_last_login:     Optional[int]   = None
    discount_applied:          int   = Field(..., ge=0, le=1)
    price_increase_last_3m:    int   = Field(..., ge=0, le=1)
    num_payment_failures_6m:   Optional[int]   = None
    referral_count:            int   = Field(0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id":               "CUST000042",
                "tenure_months":             7,
                "plan_type":                 "Basic",
                "monthly_charges":           45.99,
                "total_charges":             321.93,
                "contract_type":             "Monthly",
                "payment_method":            "Card",
                "auto_pay":                  0,
                "num_support_tickets_30d":   4,
                "avg_resolution_time_hours": 18.5,
                "satisfaction_score":        2,
                "login_count_30d":           3,
                "session_minutes_30d":       45.0,
                "feature_usage_score":       22.0,
                "days_since_last_login":     21,
                "discount_applied":          0,
                "price_increase_last_3m":    1,
                "num_payment_failures_6m":   2,
                "referral_count":            0,
            }
        }


class RetentionOfferOut(BaseModel):
    offer_type:               str
    description:              str
    estimated_retention_prob: float
    offer_cost_usd:           float
    estimated_revenue_saved:  float
    roi_pct:                  float


class PredictResponse(BaseModel):
    customer_id:         str
    churn_probability:   float
    risk_level:          str
    top_reasons:         List[str]
    retention_offer:     RetentionOfferOut
    estimated_monthly_revenue: float


# ── Helper ─────────────────────────────────────────────────────────────────────
def _score_customer(data: dict) -> tuple[float, list]:
    """Run feature engineering + preprocessing + model prediction."""
    model_bundle = get_bundle()
    df    = pd.DataFrame([data])
    df_fe = add_features(df)
    X_t   = model_bundle["preprocessor"].transform(df_fe)
    proba = float(model_bundle["model"].predict_proba(X_t)[0, 1])

    # Simple SHAP-free reason extraction for real-time speed
    # Uses domain heuristics ranked by contribution
    reasons = []
    if data.get("price_increase_last_3m", 0) == 1:
        reasons.append("price_increase_last_3m")
    if data.get("contract_type") == "Monthly":
        reasons.append("monthly_contract")
    if (data.get("satisfaction_score") or 3) <= 2:
        reasons.append("low_satisfaction")
    if data.get("num_support_tickets_30d", 0) >= 3:
        reasons.append("support_load")
    if data.get("login_count_30d", 10) <= 3:
        reasons.append("low_engagement")
    payment_failures = data.get("num_payment_failures_6m")
    if pd.notna(payment_failures) and payment_failures >= 2:
        reasons.append("payment_failures")
    days_since_login = data.get("days_since_last_login")
    if pd.notna(days_since_login) and days_since_login >= 20:
        reasons.append("low_recency")
    if data.get("auto_pay", 1) == 0:
        reasons.append("no_autopay")

    # Fallback if no rules triggered
    if not reasons:
        reasons = ["general_high_risk"]

    return proba, reasons[:5]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    if not MODEL_PATH.exists():
        return {"status": "model_not_trained", "model": None, "version": "1.0.0"}
    model_bundle = get_bundle()
    return {"status": "ok", "model": model_bundle["model_name"], "version": "1.0.0"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(customer: CustomerIn):
    """
    Predict churn probability for a single customer and return
    a personalised retention offer.
    """
    data = customer.model_dump()
    cust_id = data.pop("customer_id", "UNKNOWN")

    # Score
    churn_prob, reasons = _score_customer(data)
    risk_level = _classify_risk(churn_prob)

    # Retention offer
    rec = recommend(
        churn_probability=churn_prob,
        top_reasons=reasons,
        customer=data,
        monthly_charges=data.get("monthly_charges"),
    )

    return PredictResponse(
        customer_id=cust_id,
        churn_probability=round(churn_prob, 4),
        risk_level=risk_level,
        top_reasons=reasons,
        estimated_monthly_revenue=float(data.get("monthly_charges", 0)),
        retention_offer=RetentionOfferOut(
            offer_type=rec.offer_type,
            description=rec.description,
            estimated_retention_prob=rec.estimated_retention_prob,
            offer_cost_usd=rec.offer_cost_usd,
            estimated_revenue_saved=rec.estimated_revenue_saved,
            roi_pct=rec.roi_pct,
        )
    )


@app.get("/customers", tags=["Data"])
def list_customers(limit: int = 100, offset: int = 0, risk: Optional[str] = None):
    """
    Return a paginated list of customers with pre-computed churn scores.
    Optionally filter by risk level: low / medium / high.
    """
    ensure_db()
    model_bundle = get_bundle()

    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql("SELECT * FROM customers", con)
    con.close()

    # Drop identifier columns before scoring
    id_col   = df["customer_id"].copy()
    date_col = df["signup_date"].copy()
    df_score = df.drop(columns=["customer_id", "signup_date", "churn"], errors="ignore")

    df_fe = add_features(df_score)
    X_t   = model_bundle["preprocessor"].transform(df_fe)
    probas = model_bundle["model"].predict_proba(X_t)[:, 1]

    df["churn_probability"] = probas.round(4)
    df["risk_level"]        = [_classify_risk(p) for p in probas]

    if risk:
        df = df[df["risk_level"] == risk.lower()]

    df = df.sort_values("churn_probability", ascending=False)
    records = df.iloc[offset: offset + limit].replace([np.inf, -np.inf], np.nan)
    records = records.astype(object).where(pd.notna(records), None)
    result = records.to_dict(orient="records")
    return {"total": len(df), "limit": limit, "offset": offset, "customers": result}


@app.get("/customers/{customer_id}", tags=["Data"])
def get_customer(customer_id: str):
    """Fetch a single customer record by ID and score them."""
    ensure_db()

    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql(
        "SELECT * FROM customers WHERE customer_id = ?",
        con, params=(customer_id,)
    )
    con.close()

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")

    safe_row = df.replace([np.inf, -np.inf], np.nan).astype(object)
    safe_row = safe_row.where(pd.notna(safe_row), None)
    row = safe_row.iloc[0].to_dict()
    data = {k: v for k, v in row.items()
            if k not in ("customer_id", "signup_date", "churn")}
    churn_prob, reasons = _score_customer(data)
    risk_level = _classify_risk(churn_prob)

    return {
        **row,
        "churn_probability": round(churn_prob, 4),
        "risk_level":        risk_level,
        "top_reasons":       reasons,
    }
