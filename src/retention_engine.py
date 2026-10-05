"""
retention_engine.py
-------------------
Rule-based + ML-driven engine that maps a customer's churn risk profile
to a personalised retention offer.

Offer catalogue:
  - DISCOUNT         : 10-15% off for 3 months
  - PLAN_UPGRADE     : Complimentary upgrade with waived fee
  - FREE_MONTH       : One free month credited
  - PRIORITY_SUPPORT : Dedicated support SLA (< 2 h response)
  - PERSONAL_CALL    : Personal account manager outreach
  - LOYALTY_REWARD   : Loyalty points / referral bonus

Business metrics:
  - Estimated retention probability per offer
  - Estimated monthly revenue saved
  - Simple ROI calculation
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np

# ── Offer definitions ─────────────────────────────────────────────────────────
OFFERS: Dict[str, Dict[str, Any]] = {
    "DISCOUNT": {
        "description":             "10–15% discount for the next 3 months",
        "estimated_retention_prob": 0.68,
        "offer_cost_pct":          0.125,   # 12.5% of monthly_charges × 3
        "duration_months":         3,
        "priority":                1,
    },
    "FREE_MONTH": {
        "description":             "One free month credited to your account",
        "estimated_retention_prob": 0.62,
        "offer_cost_pct":          1.0,
        "duration_months":         1,
        "priority":                2,
    },
    "PLAN_UPGRADE": {
        "description":             "Complimentary upgrade to the next plan tier (3 months)",
        "estimated_retention_prob": 0.72,
        "offer_cost_pct":          0.20,
        "duration_months":         3,
        "priority":                3,
    },
    "PRIORITY_SUPPORT": {
        "description":             "Dedicated Priority Support SLA: response < 2 hours",
        "estimated_retention_prob": 0.55,
        "offer_cost_pct":          0.05,
        "duration_months":         3,
        "priority":                4,
    },
    "PERSONAL_CALL": {
        "description":             "Personal account manager call for account review",
        "estimated_retention_prob": 0.50,
        "offer_cost_pct":          0.02,
        "duration_months":         1,
        "priority":                5,
    },
    "LOYALTY_REWARD": {
        "description":             "Loyalty reward: 500 bonus points + referral bonus",
        "estimated_retention_prob": 0.45,
        "offer_cost_pct":          0.08,
        "duration_months":         1,
        "priority":                6,
    },
}


@dataclass
class RetentionRecommendation:
    offer_type:                    str
    description:                   str
    estimated_retention_prob:      float
    offer_cost_usd:                float    # one-time or 3-month cost
    estimated_monthly_revenue:     float
    estimated_revenue_saved:       float    # if retained
    roi_pct:                       float
    risk_level:                    str
    churn_probability:             float
    top_reasons:                   List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_risk(churn_prob: float) -> str:
    if churn_prob >= 0.65:
        return "high"
    elif churn_prob >= 0.40:
        return "medium"
    return "low"


def _select_offer(churn_prob: float, top_reasons: List[str],
                  customer: Dict[str, Any]) -> str:
    """
    Rule-based offer selection driven by top SHAP-derived churn reasons
    and customer profile. Falls back to PERSONAL_CALL for low-risk cases.
    """
    reasons_lower = [r.lower() for r in top_reasons]

    # --- Rule priority cascade ---

    # Price-sensitive churn
    if any(k in r for r in reasons_lower
           for k in ("price_increase", "monthly_charges", "charges_per_month")):
        return "DISCOUNT"

    # Plan upgrade opportunity (Basic plan + high charges relative to value)
    if (customer.get("plan_type", "").lower() == "basic"
            and customer.get("monthly_charges", 0) >= 35):
        return "PLAN_UPGRADE"

    # Support dissatisfaction
    if any(k in r for r in reasons_lower
           for k in ("support", "satisfaction", "resolution")):
        return "PRIORITY_SUPPORT"

    # Engagement / usage decline
    if any(k in r for r in reasons_lower
           for k in ("usage", "engagement", "login", "recency", "session")):
        return "FREE_MONTH"

    # Payment reliability issues
    if any(k in r for r in reasons_lower
           for k in ("payment", "reliability", "failure")):
        return "DISCOUNT"

    # Long-tenured customer about to leave → loyalty reward
    if customer.get("tenure_months", 0) >= 24:
        return "LOYALTY_REWARD"

    # Default fallback
    if churn_prob >= 0.65:
        return "PERSONAL_CALL"

    return "LOYALTY_REWARD"


def recommend(
    churn_probability: float,
    top_reasons: List[str],
    customer: Dict[str, Any],
    monthly_charges: float | None = None,
) -> RetentionRecommendation:
    """
    Given churn probability, SHAP reason codes, and raw customer features,
    return a personalised RetentionRecommendation.

    Parameters
    ----------
    churn_probability : float
        Model output probability of churn (0–1).
    top_reasons : list of str
        Top SHAP feature names driving the prediction.
    customer : dict
        Raw customer feature values (used for rule matching).
    monthly_charges : float, optional
        Explicit monthly charge override; falls back to customer['monthly_charges'].
    """
    if monthly_charges is None:
        monthly_charges = float(customer.get("monthly_charges", 50.0))

    risk_level  = _classify_risk(churn_probability)
    offer_key   = _select_offer(churn_probability, top_reasons, customer)
    offer_meta  = OFFERS[offer_key]

    # Cost of offer
    offer_cost = round(
        monthly_charges
        * offer_meta["offer_cost_pct"]
        * offer_meta["duration_months"],
        2,
    )

    # Revenue at risk if customer churns (12-month LTV proxy)
    revenue_at_risk = round(monthly_charges * 12, 2)

    # Only customers who would churn and are retained by the offer count as saved.
    estimated_revenue_saved = round(
        churn_probability
        * offer_meta["estimated_retention_prob"]
        * revenue_at_risk,
        2,
    )

    # ROI = (revenue_saved - offer_cost) / offer_cost × 100
    roi = round(
        ((estimated_revenue_saved - offer_cost) / offer_cost) * 100
        if offer_cost > 0 else 0.0,
        1,
    )

    return RetentionRecommendation(
        offer_type=offer_key,
        description=offer_meta["description"],
        estimated_retention_prob=round(offer_meta["estimated_retention_prob"], 2),
        offer_cost_usd=offer_cost,
        estimated_monthly_revenue=monthly_charges,
        estimated_revenue_saved=estimated_revenue_saved,
        roi_pct=roi,
        risk_level=risk_level,
        churn_probability=round(churn_probability, 4),
        top_reasons=top_reasons[:3],
    )


def bulk_recommend(customers_df, probas: np.ndarray,
                   reason_codes_df=None) -> list:
    """
    Generate recommendations for all customers in a DataFrame.
    Returns list of dicts.
    """
    results = []
    for i, (_, row) in enumerate(customers_df.iterrows()):
        customer = row.to_dict()
        churn_prob = float(probas[i])

        if reason_codes_df is not None and i in reason_codes_df["customer_index"].values:
            rc = reason_codes_df[reason_codes_df["customer_index"] == i].iloc[0]
            reasons = [rc.get("reason_1", ""), rc.get("reason_2", ""),
                       rc.get("reason_3", "")]
            reasons = [r for r in reasons if r]
        else:
            reasons = ["high_risk_profile"]

        rec = recommend(churn_prob, reasons, customer)
        results.append(rec.to_dict())
    return results


# ── Business ROI Summary ─────────────────────────────────────────────────────
def compute_campaign_roi(recommendations: list) -> dict:
    """
    Aggregate campaign ROI metrics.
    """
    total_customers = len(recommendations)
    high_risk       = sum(1 for r in recommendations if r["risk_level"] == "high")
    med_risk        = sum(1 for r in recommendations if r["risk_level"] == "medium")

    total_mrr_at_risk = sum(
        r["estimated_monthly_revenue"] * r["churn_probability"]
        for r in recommendations
        if r["risk_level"] in ("high", "medium")
    )

    total_revenue_saved = sum(r["estimated_revenue_saved"] for r in recommendations)
    total_offer_cost    = sum(r["offer_cost_usd"]           for r in recommendations)
    campaign_roi = round(
        ((total_revenue_saved - total_offer_cost) / total_offer_cost) * 100
        if total_offer_cost > 0 else 0.0,
        1,
    )

    return {
        "total_customers":        total_customers,
        "high_risk_customers":    high_risk,
        "medium_risk_customers":  med_risk,
        "total_mrr_at_risk_usd":  round(total_mrr_at_risk, 2),
        "total_revenue_saved_usd":round(total_revenue_saved, 2),
        "total_offer_cost_usd":   round(total_offer_cost, 2),
        "campaign_roi_pct":       campaign_roi,
    }


if __name__ == "__main__":
    # Quick smoke test
    test_customer = {
        "plan_type":              "Basic",
        "monthly_charges":        45.99,
        "tenure_months":          8,
        "num_support_tickets_30d":4,
        "satisfaction_score":     2,
        "price_increase_last_3m": 1,
        "num_payment_failures_6m":2,
    }
    rec = recommend(
        churn_probability=0.78,
        top_reasons=["price_increase_last_3m", "satisfaction_score", "support_load"],
        customer=test_customer,
    )
    print("Retention Recommendation:")
    for k, v in rec.to_dict().items():
        print(f"  {k:<32}: {v}")
