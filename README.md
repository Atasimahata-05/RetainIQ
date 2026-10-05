# RetainIQ: Customer Churn Prediction and Retention

RetainIQ is an early-warning system for subscription businesses. It estimates customer churn risk, explains influential signals, and recommends a retention offer so customer-success teams can prioritize outreach.

## Dataset

The project uses a reproducible synthetic dataset of 20,000 subscription customers with account, billing, contract, payment, support, satisfaction, and usage attributes. It includes missing values, outliers, and simulated billing-cycle seasonality. Churn prevalence is approximately 22%.

Dataset: [data/churn_data.csv](data/churn_data.csv). The notebook can regenerate it if missing.

## Technologies

Python 3.11, pandas, NumPy, scikit-learn, XGBoost, LightGBM, SHAP, imbalanced-learn, FastAPI, Uvicorn, Streamlit, Plotly, SQLite, matplotlib, and seaborn.

## Setup and Run

Create an environment and install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Open and run the submission notebook:

```powershell
jupyter lab notebooks/Atasi_Mahata_RetainIQ.ipynb
```

Run the full API and dashboard from the project root:

```powershell
python src/data_generation.py
python src/train.py
python src/evaluate.py
python src/explain.py
uvicorn src.api:app --reload --port 8000
streamlit run app/app.py
```

API documentation: `http://localhost:8000/docs`  
Dashboard: `http://localhost:8501`

## Model and Key Information

Logistic Regression, Random Forest, XGBoost, and LightGBM are compared using stratified five-fold cross-validation. In the current synthetic-data run, the selected Logistic Regression model achieved ROC-AUC 0.637, PR-AUC 0.313, and recall 0.800 at a 0.40 threshold. This threshold favors identifying potential churners and also flags many customers who do not churn.

SHAP explains model predictions. The recommendation engine suggests discounts, free months, plan upgrades, priority support, personal calls, or loyalty rewards. Revenue-saved and campaign-ROI figures are assumptions, not measured causal outcomes. Validate on real customer data and controlled retention experiments before production use.

## Submission Files

- Code: [notebooks/Atasi_Mahata_RetainIQ.ipynb](notebooks/Atasi_Mahata_RetainIQ.ipynb)
- Report: [Atasi_Mahata_ProjectReport.docx](Atasi_Mahata_ProjectReport.docx)
- Dependencies: [requirements.txt](requirements.txt)
