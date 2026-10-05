"""
app/app.py
----------
Streamlit multi-page dashboard for the Customer Churn Prediction system.

Pages:
  1. Overview          – KPI cards + churn trend over time
  2. Customer Explorer – search/filter, per-customer churn probability
  3. High-Risk Alerts  – top high-risk table with reasons + offers
  4. Real-Time Scoring – manual input form → live prediction
  5. Model Insights    – SHAP plots, ROC/PR curves, decile lift, ROI calculator
"""

import sys, json, sqlite3, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from feature_engineering import add_features
from retention_engine    import recommend, _classify_risk, compute_campaign_roi

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
DATA_PATH  = ROOT / "data"   / "churn_data.csv"
NOTEBOOKS  = ROOT / "notebooks"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetainIQ | Customer Retention Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark background */
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* KPI cards */
    .kpi-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #a78bfa; }
    .kpi-label { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 4px; }
    .kpi-delta { font-size: 0.8rem; margin-top: 6px; }

    /* Section headings */
    h1, h2, h3 { color: #e2e8f0 !important; }

    /* Risk badges */
    .badge-high   { background:#ef4444; color:white; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-medium { background:#f59e0b; color:white; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-low    { background:#22c55e; color:white; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }

    /* Dividers */
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* Metric override */
    [data-testid="metric-container"] { background: rgba(255,255,255,0.06); border-radius:12px; padding:10px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

if not MODEL_PATH.exists():
    st.error("The churn model has not been trained yet. Run `python src/train.py` from the project root, then restart the dashboard.")
    st.stop()
if not DATA_PATH.exists():
    st.error("The customer dataset is missing. Run `python src/data_generation.py` from the project root.")
    st.stop()

@st.cache_resource(show_spinner="Loading model…")
def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner="Loading data…")
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["signup_date"])
    return df

@st.cache_data(show_spinner="Scoring customers…")
def score_all_customers(_bundle, _df):
    """Score entire dataset and attach churn_probability + risk_level."""
    df = _df.copy()
    df_score = df.drop(columns=["customer_id", "signup_date", "churn"], errors="ignore")
    df_fe    = add_features(df_score)
    X_t      = _bundle["preprocessor"].transform(df_fe)
    probas   = _bundle["model"].predict_proba(X_t)[:, 1]
    df["churn_probability"] = probas.round(4)
    df["risk_level"]        = [_classify_risk(p) for p in probas]
    return df


def get_retention_offer(row, bundle):
    """Get retention recommendation for a single row."""
    reasons = []
    if row.get("price_increase_last_3m", 0) == 1:
        reasons.append("price_increase_last_3m")
    if row.get("contract_type") == "Monthly":
        reasons.append("monthly_contract")
    if (row.get("satisfaction_score") or 3) <= 2:
        reasons.append("low_satisfaction")
    if row.get("num_support_tickets_30d", 0) >= 3:
        reasons.append("support_load")
    if row.get("login_count_30d", 10) <= 3:
        reasons.append("low_engagement")
    if row.get("num_payment_failures_6m", 0) >= 2:
        reasons.append("payment_failures")
    if not reasons:
        reasons = ["general_high_risk"]

    rec = recommend(
        churn_probability=row["churn_probability"],
        top_reasons=reasons,
        customer=row,
        monthly_charges=float(row.get("monthly_charges", 50)),
    )
    return rec, reasons


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

bundle = load_bundle()
df_raw = load_data()
df     = score_all_customers(bundle, df_raw)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/churn.png", width=60)
    st.title("RetainIQ")
    st.caption(f"Model: **{bundle['model_name']}**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Overview",
         "🔍 Customer Explorer",
         "🚨 High-Risk Alerts",
         "⚡ Real-Time Scoring",
         "🧠 Model Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("© 2026 RetainIQ | Customer Retention Intelligence")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("📊 RetainIQ — Churn Overview")
    st.caption("Real-time KPIs and churn trend analytics for your customer base.")
    st.markdown("---")

    # KPIs
    total_customers = len(df)
    churn_rate      = df["churn"].mean()
    high_risk_count = (df["risk_level"] == "high").sum()
    at_risk = df[df["risk_level"].isin(["high", "medium"])]
    mrr_at_risk = (at_risk["monthly_charges"] * at_risk["churn_probability"]).sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-label">Total Customers</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{churn_rate:.1%}</div>
            <div class="kpi-label">Historical Churn Rate</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#ef4444;">{high_risk_count:,}</div>
            <div class="kpi-label">High-Risk Customers</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#f59e0b;">${mrr_at_risk:,.0f}</div>
            <div class="kpi-label">MRR at Risk ($/mo)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Churn trend over time (by signup cohort)
    st.subheader("📈 Churn by Signup Cohort (Monthly)")
    df_trend = df.copy()
    df_trend["signup_month"] = df_trend["signup_date"].dt.to_period("M").astype(str)
    trend = (
        df_trend.groupby("signup_month")["churn"]
        .agg(churned="sum", total="count")
        .reset_index()
        .assign(churn_rate=lambda d: d["churned"] / d["total"])
        .tail(24)  # last 24 months
    )
    fig_trend = px.area(
        trend, x="signup_month", y="churn_rate",
        title="Monthly Churn Rate by Cohort",
        labels={"churn_rate": "Churn Rate", "signup_month": "Signup Month"},
        color_discrete_sequence=["#a78bfa"],
        template="plotly_dark",
    )
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_tickformat=".0%", height=350,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Risk distribution + Plan type breakdown
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🎯 Risk Level Distribution")
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig_risk = px.pie(
            risk_counts, names="Risk Level", values="Count",
            color="Risk Level",
            color_discrete_map={"high":"#ef4444","medium":"#f59e0b","low":"#22c55e"},
            template="plotly_dark",
        )
        fig_risk.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_r:
        st.subheader("📦 Churn Rate by Plan Type")
        by_plan = df.groupby("plan_type")["churn"].mean().reset_index()
        fig_plan = px.bar(
            by_plan, x="plan_type", y="churn",
            color="plan_type",
            color_discrete_sequence=["#a78bfa","#60a5fa","#34d399"],
            template="plotly_dark",
        )
        fig_plan.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_tickformat=".0%", showlegend=False, height=300,
        )
        st.plotly_chart(fig_plan, use_container_width=True)

    # Contract type churn
    st.subheader("📋 Churn Rate by Contract Type")
    by_contract = df.groupby("contract_type")["churn"].mean().reset_index()
    fig_cont = px.bar(
        by_contract, x="contract_type", y="churn",
        color="contract_type",
        color_discrete_sequence=["#f87171","#34d399"],
        template="plotly_dark",
    )
    fig_cont.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_tickformat=".0%", showlegend=False, height=280,
    )
    st.plotly_chart(fig_cont, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: CUSTOMER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Customer Explorer":
    st.title("🔍 Customer Explorer")
    st.caption("Search and filter customers to inspect individual churn risk.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        plan_filter  = st.multiselect("Plan Type", options=df["plan_type"].unique().tolist(), default=[])
    with col2:
        contr_filter = st.multiselect("Contract", options=df["contract_type"].unique().tolist(), default=[])
    with col3:
        risk_filter  = st.multiselect("Risk Level", options=["high","medium","low"], default=[])
    with col4:
        search_id = st.text_input("Search by Customer ID", placeholder="CUST000042")

    filtered = df.copy()
    if plan_filter:
        filtered = filtered[filtered["plan_type"].isin(plan_filter)]
    if contr_filter:
        filtered = filtered[filtered["contract_type"].isin(contr_filter)]
    if risk_filter:
        filtered = filtered[filtered["risk_level"].isin(risk_filter)]
    if search_id.strip():
        filtered = filtered[filtered["customer_id"].str.contains(search_id.strip(), case=False, regex=False)]

    st.markdown(f"**{len(filtered):,} customers** match your filters.")
    if filtered.empty:
        st.info("No customers match the current filters.")
        st.stop()

    display_cols = ["customer_id","plan_type","contract_type","monthly_charges",
                    "tenure_months","satisfaction_score",
                    "churn_probability","risk_level","churn"]
    filtered_show = filtered[display_cols].sort_values("churn_probability", ascending=False)
    st.dataframe(
        filtered_show.head(200).style.background_gradient(
            subset=["churn_probability"], cmap="RdYlGn_r"
        ).format({"churn_probability": "{:.2%}", "monthly_charges": "${:.2f}"}),
        height=420, use_container_width=True,
    )

    # Individual customer deep dive
    st.markdown("---")
    st.subheader("📋 Customer Deep Dive")
    selected_cid = st.selectbox("Select a customer ID", options=filtered["customer_id"].head(200).tolist())
    row = filtered[filtered["customer_id"] == selected_cid].iloc[0]

    col_a, col_b = st.columns([1, 2])
    with col_a:
        risk_color = {"high":"#ef4444","medium":"#f59e0b","low":"#22c55e"}
        st.metric("Churn Probability", f"{row['churn_probability']:.1%}")
        st.metric("Risk Level", row["risk_level"].upper())
        st.metric("Monthly Charges", f"${row['monthly_charges']:.2f}")
        st.metric("Tenure (months)", int(row["tenure_months"]))
        st.metric("Satisfaction Score", int(row.get("satisfaction_score", 0)) if pd.notna(row.get("satisfaction_score")) else "N/A")

    with col_b:
        # Churn probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=row["churn_probability"] * 100,
            delta={"reference": 20, "valueformat": ".1f"},
            title={"text": f"Churn Risk – {row['customer_id']}", "font": {"size": 14, "color": "white"}},
            gauge={
                "axis":   {"range": [0, 100], "tickcolor": "white"},
                "bar":    {"color": risk_color.get(row["risk_level"], "#a78bfa")},
                "steps": [
                    {"range": [0,  40], "color": "rgba(34,197,94,0.2)"},
                    {"range": [40, 65], "color": "rgba(245,158,11,0.2)"},
                    {"range": [65,100], "color": "rgba(239,68,68,0.2)"},
                ],
                "threshold": {"line": {"color": "white","width": 2}, "value": row["churn_probability"]*100},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=250,
            font={"color": "white"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: HIGH-RISK ALERTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🚨 High-Risk Alerts":
    st.title("🚨 High-Risk Customer Alerts")
    st.caption("Top customers at highest churn risk — with reasons and recommended retention offers.")
    st.markdown("---")

    high_risk = df[df["risk_level"] == "high"].sort_values("churn_probability", ascending=False).head(100)

    # Compute offers
    offer_rows = []
    for _, row in high_risk.iterrows():
        rec, reasons = get_retention_offer(row.to_dict(), bundle)
        offer_rows.append({
            "customer_id":          row["customer_id"],
            "churn_probability":    row["churn_probability"],
            "risk_level":           row["risk_level"],
            "plan_type":            row["plan_type"],
            "monthly_charges":      row["monthly_charges"],
            "top_reason_1":         reasons[0] if len(reasons) > 0 else "",
            "top_reason_2":         reasons[1] if len(reasons) > 1 else "",
            "recommended_offer":    rec.offer_type,
            "offer_description":    rec.description,
            "est_revenue_saved":    rec.estimated_revenue_saved,
            "roi_pct":              rec.roi_pct,
        })

    alerts_df = pd.DataFrame(offer_rows)

    # ROI summary banner
    total_rev = alerts_df["est_revenue_saved"].sum()
    st.info(f"💰 Targeting these {len(alerts_df)} high-risk customers could save **${total_rev:,.0f}** in annual revenue.")

    # Offer distribution
    st.subheader("Recommended Offer Distribution")
    offer_dist = alerts_df["recommended_offer"].value_counts().reset_index()
    offer_dist.columns = ["Offer", "Count"]
    fig_offer = px.bar(
        offer_dist, x="Offer", y="Count",
        color="Offer",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template="plotly_dark",
    )
    fig_offer.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False, height=280)
    st.plotly_chart(fig_offer, use_container_width=True)

    # Alert table
    st.subheader(f"Top 100 High-Risk Customers")
    st.dataframe(
        alerts_df.style
        .background_gradient(subset=["churn_probability"], cmap="Reds")
        .background_gradient(subset=["est_revenue_saved"], cmap="Greens")
        .format({
            "churn_probability":  "{:.1%}",
            "monthly_charges":    "${:.2f}",
            "est_revenue_saved":  "${:,.2f}",
            "roi_pct":            "{:.1f}%",
        }),
        height=500, use_container_width=True,
    )

    # Download
    csv = alerts_df.to_csv(index=False)
    st.download_button("⬇️ Download Alert CSV", data=csv,
                       file_name="high_risk_alerts.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: REAL-TIME SCORING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚡ Real-Time Scoring":
    st.title("⚡ Real-Time Customer Scoring")
    st.caption("Enter customer attributes below to instantly predict churn probability and get a retention recommendation.")
    st.markdown("---")

    with st.form("score_form"):
        st.subheader("🧾 Customer Profile")

        col1, col2, col3 = st.columns(3)
        with col1:
            tenure_months    = st.slider("Tenure (months)",         0, 72, 10)
            plan_type        = st.selectbox("Plan Type",            ["Basic","Standard","Premium"])
            contract_type    = st.selectbox("Contract Type",        ["Monthly","Yearly"])
            payment_method   = st.selectbox("Payment Method",       ["Card","UPI","Bank","Wallet"])
            auto_pay         = st.selectbox("Auto Pay",             [1, 0], format_func=lambda x: "Yes" if x==1 else "No")

        with col2:
            monthly_charges        = st.number_input("Monthly Charges ($)", 15.0, 200.0, 59.99, step=1.0)
            total_charges          = st.number_input("Total Charges ($)",   0.0, 10000.0, float(monthly_charges * tenure_months))
            satisfaction_score     = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
            num_support_tickets_30d= st.slider("Support Tickets (30d)",   0, 15, 1)
            avg_resolution_hrs     = st.number_input("Avg Resolution Time (hrs)", 0.5, 96.0, 12.0)

        with col3:
            login_count_30d         = st.slider("Logins (30d)",           0, 60, 12)
            session_minutes_30d     = st.number_input("Session Minutes (30d)", 0.0, 2000.0, 200.0)
            feature_usage_score     = st.slider("Feature Usage Score (0-100)", 0, 100, 40)
            days_since_last_login   = st.slider("Days Since Last Login",   0, 180, 10)
            discount_applied        = st.selectbox("Discount Applied",     [0,1], format_func=lambda x: "Yes" if x else "No")
            price_increase_last_3m  = st.selectbox("Price Increase (3m)",  [0,1], format_func=lambda x: "Yes" if x else "No")
            num_payment_failures_6m = st.slider("Payment Failures (6m)",   0, 5, 0)
            referral_count          = st.slider("Referrals",               0, 10, 0)

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

    if submitted:
        customer_data = {
            "tenure_months":             tenure_months,
            "plan_type":                 plan_type,
            "monthly_charges":           monthly_charges,
            "total_charges":             total_charges,
            "contract_type":             contract_type,
            "payment_method":            payment_method,
            "auto_pay":                  auto_pay,
            "num_support_tickets_30d":   num_support_tickets_30d,
            "avg_resolution_time_hours": avg_resolution_hrs,
            "satisfaction_score":        satisfaction_score,
            "login_count_30d":           login_count_30d,
            "session_minutes_30d":       session_minutes_30d,
            "feature_usage_score":       feature_usage_score,
            "days_since_last_login":     days_since_last_login,
            "discount_applied":          discount_applied,
            "price_increase_last_3m":    price_increase_last_3m,
            "num_payment_failures_6m":   num_payment_failures_6m,
            "referral_count":            referral_count,
        }
        df_single = pd.DataFrame([customer_data])
        df_fe     = add_features(df_single)
        X_t       = bundle["preprocessor"].transform(df_fe)
        churn_prob = float(bundle["model"].predict_proba(X_t)[0, 1])
        risk       = _classify_risk(churn_prob)

        reasons = []
        if price_increase_last_3m: reasons.append("price_increase_last_3m")
        if contract_type=="Monthly": reasons.append("monthly_contract")
        if satisfaction_score <= 2:  reasons.append("low_satisfaction")
        if num_support_tickets_30d >= 3: reasons.append("support_load")
        if login_count_30d <= 3:     reasons.append("low_engagement")
        if num_payment_failures_6m >= 2: reasons.append("payment_failures")
        if days_since_last_login >= 20:  reasons.append("low_recency")
        if not reasons: reasons = ["general_high_risk"]

        rec = recommend(churn_prob, reasons, customer_data, monthly_charges)

        st.markdown("---")
        risk_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader(f"{risk_colors.get(risk, '⚪')} Churn Prediction Result")
            prob_pct = churn_prob * 100
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                title={"text": f"Churn Probability – Risk: {risk.upper()}", "font": {"color":"white","size":14}},
                gauge={
                    "axis": {"range":[0,100],"tickcolor":"white"},
                    "bar":  {"color":{"high":"#ef4444","medium":"#f59e0b","low":"#22c55e"}.get(risk,"#a78bfa")},
                    "steps":[
                        {"range":[0,40],  "color":"rgba(34,197,94,0.15)"},
                        {"range":[40,65], "color":"rgba(245,158,11,0.15)"},
                        {"range":[65,100],"color":"rgba(239,68,68,0.15)"},
                    ],
                },
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color":"white"}, height=260)
            st.plotly_chart(fig_g, use_container_width=True)

            st.markdown("**Top Churn Drivers:**")
            for i, r in enumerate(reasons[:5], 1):
                st.markdown(f"  `{i}. {r.replace('_',' ').title()}`")

        with col_right:
            st.subheader("💡 Retention Recommendation")
            card_col = {"DISCOUNT":"#3b82f6","FREE_MONTH":"#8b5cf6",
                        "PLAN_UPGRADE":"#10b981","PRIORITY_SUPPORT":"#f59e0b",
                        "PERSONAL_CALL":"#ef4444","LOYALTY_REWARD":"#6366f1"}
            color = card_col.get(rec.offer_type, "#a78bfa")

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.07);border:1px solid {color};
                        border-radius:14px;padding:20px;margin-top:10px;">
                <h3 style="color:{color};margin:0 0 8px 0">{rec.offer_type.replace("_"," ").title()}</h3>
                <p style="color:#cbd5e1;margin:0 0 14px 0">{rec.description}</p>
                <table style="width:100%;color:#e2e8f0;font-size:0.88rem">
                    <tr><td>Retention Probability</td><td><b>{rec.estimated_retention_prob:.0%}</b></td></tr>
                    <tr><td>Offer Cost</td><td><b>${rec.offer_cost_usd:.2f}</b></td></tr>
                    <tr><td>Estimated Revenue Saved</td><td><b style="color:#34d399">${rec.estimated_revenue_saved:.2f}</b></td></tr>
                    <tr><td>Campaign ROI</td><td><b style="color:#a78bfa">{rec.roi_pct:.1f}%</b></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 Model Insights":
    st.title("🧠 Model Insights & Explainability")
    st.caption(f"Deep dive into model performance and feature importance for **{bundle['model_name']}**.")
    st.markdown("---")

    # ── CV scores ─────────────────────────────────────────────────────────────
    cv_path = ROOT / "models" / "cv_results.json"
    if cv_path.exists():
        cv_data = json.loads(cv_path.read_text())
        st.subheader("📊 Cross-Validation Comparison")
        cv_df = pd.DataFrame(cv_data).T.reset_index().rename(columns={"index":"Model"})
        fig_cv = px.bar(
            cv_df, x="Model", y=["roc_auc","recall"],
            barmode="group",
            color_discrete_sequence=["#a78bfa","#60a5fa"],
            template="plotly_dark",
            title="ROC-AUC & Recall by Model (CV)",
        )
        fig_cv.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig_cv, use_container_width=True)

    # ── Evaluation metrics ─────────────────────────────────────────────────────
    eval_path = ROOT / "models" / "eval_metrics.json"
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text())
        st.subheader("🎯 Hold-Out Test Metrics")
        cols = st.columns(6)
        for i, (k, v) in enumerate(eval_data.items()):
            cols[i].metric(k.upper().replace("_","-"), f"{v:.4f}")

    # ── SHAP plots ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🌐 SHAP Global Feature Importance")

    shap_bar = NOTEBOOKS / "shap_summary_bar.png"
    shap_bee = NOTEBOOKS / "shap_summary_beeswarm.png"

    tab1, tab2 = st.tabs(["Bar Chart", "Beeswarm"])
    with tab1:
        if shap_bar.exists():
            st.image(str(shap_bar), use_column_width=True)
        else:
            st.info("Run `python src/explain.py` to generate SHAP plots.")
    with tab2:
        if shap_bee.exists():
            st.image(str(shap_bee), use_column_width=True)
        else:
            st.info("Run `python src/explain.py` to generate SHAP plots.")

    # ── ROC / PR curves ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 ROC & Precision-Recall Curves")
    roc_path = NOTEBOOKS / "roc_curve.png"
    pr_path  = NOTEBOOKS / "pr_curve.png"

    col_roc, col_pr = st.columns(2)
    with col_roc:
        if roc_path.exists():
            st.image(str(roc_path), use_column_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate curves.")
    with col_pr:
        if pr_path.exists():
            st.image(str(pr_path), use_column_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate curves.")

    # ── Lift chart ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📉 Decile Lift Chart")
    lift_path = NOTEBOOKS / "lift_chart.png"
    if lift_path.exists():
        st.image(str(lift_path), use_column_width=True)
    else:
        st.info("Run `python src/evaluate.py` to generate lift chart.")

    # ── ROI Calculator ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💰 Retention Campaign ROI Calculator")
    st.caption("Estimate the revenue impact of your churn prevention campaign.")

    col_roi1, col_roi2 = st.columns(2)
    with col_roi1:
        n_targeted       = st.slider("Number of customers to target", 50, 5000, 500, 50)
        avg_monthly_rev  = st.number_input("Avg monthly revenue per customer ($)", 10.0, 200.0, 60.0)
        average_churn_risk = st.slider("Average predicted churn probability (%)", 10, 100, 65)
        retention_rate   = st.slider("Offer success among at-risk customers (%)", 10, 80, 45)
        avg_offer_cost   = st.number_input("Average offer cost ($)", 5.0, 500.0, 50.0)

    with col_roi2:
        customers_saved       = n_targeted * average_churn_risk / 100 * retention_rate / 100
        annual_rev_saved      = customers_saved * avg_monthly_rev * 12
        total_campaign_cost   = n_targeted * avg_offer_cost
        net_benefit           = annual_rev_saved - total_campaign_cost
        campaign_roi          = ((annual_rev_saved - total_campaign_cost) / max(total_campaign_cost,1)) * 100

        st.markdown(f"""
        <div style="background:rgba(167,139,250,0.1);border:1px solid #a78bfa;
                    border-radius:14px;padding:20px;margin-top:10px;">
            <h3 style="color:#a78bfa;">📊 ROI Summary</h3>
            <table style="width:100%;color:#e2e8f0;">
                <tr><td>Expected Customers Saved</td><td><b>{customers_saved:,.1f}</b></td></tr>
                <tr><td>Annual Revenue Saved</td><td><b style="color:#34d399">${annual_rev_saved:,.0f}</b></td></tr>
                <tr><td>Campaign Cost</td><td><b style="color:#f87171">${total_campaign_cost:,.0f}</b></td></tr>
                <tr><td>Net Benefit</td><td><b>${net_benefit:,.0f}</b></td></tr>
                <tr><td><b>Campaign ROI</b></td><td><b style="color:#a78bfa;font-size:1.2rem">{campaign_roi:.1f}%</b></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
