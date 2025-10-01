# 📈 Predictive-Analysis-for-sales-forecasting

A flexible **Streamlit** app for exploratory analysis and **time-series / ML forecasting** on your own sales dataset (CSV/Excel).  
It supports classical TS models (**ARIMA**, **Holt-Winters Exponential Smoothing**) and ML models (**Linear Regression**, **Random Forest**, **Gradient Boosting**, simple **Neural Network**) with built-in preprocessing, visualization, feature engineering, and multi-month horizon forecasts.

---

## ✨ Features

- 🔌 **Bring your own data** (CSV/XLS/XLSX) — choose file **encoding** and whether the **first row is header**
- 🗓️ Smart **date column detection/conversion** + sorting, duplicate date handling, and frequency inference (`MS` fallback)
- 🔍 **EDA**: data preview, descriptive stats, line plot, correlation heatmap
- 🧩 **Feature engineering** (optional):
  - Date features: `Year`, `Month`, `DayOfWeek`, `IsWeekend`
  - Lag features (1..N), Rolling mean windows
  - Categorical encoding: **Label Encoding** or **One-Hot Encoding**
- 🧠 **Models**:
  - Time-series: **ARIMA**, **Exponential Smoothing** (additive trend/seasonality, period=12)
  - Machine learning: **Linear Regression**, **Random Forest**, **Gradient Boosting**, **Dense Neural Net (Keras)**
- 🗺️ **Forecast horizons**: 3, 6, or 12 months (auto-scaled for daily data)
- 📊 **Interactive chart** (Plotly): Actual vs. Predicted, plus forecast table
- 📉 **Metrics** (for ML models): MSE, MAE
- 💾 **Export**: build a trimmed dataset of Date + Predictors + Target and **download as CSV**


---

## ✅ Requirements

Tested with Python **3.10–3.11**. Suggested `requirements.txt`:

```
streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
statsmodels==0.14.2
matplotlib==3.8.4
seaborn==0.13.2
plotly==5.24.1
tensorflow==2.15.0         # or: tensorflow-cpu==2.15.0
```

> If TensorFlow is heavy for your environment, you can **omit** it and simply avoid selecting the Neural Network model.

---

## ⚙️ Setup & Run

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Launch the app
streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`).

---

## 📥 Data Expectations

- A **date column** (daily or monthly preferred). If it’s not auto-detected, you can pick a column and the app will attempt to parse it to datetime.
- A **numeric target** (e.g., `Sales`).
- Optional **predictor** columns (numeric or categorical).

> The app handles duplicates by aggregating duplicate dates (mean), infers frequency via `pandas.infer_freq` and falls back to **monthly start (`MS`)** if unknown, then forward-fills gaps.

---

## 🧭 How to Use

1. **Upload** your dataset (CSV/Excel) in the sidebar.
2. Choose **encoding** and whether the first row is **header**.
3. If needed, select which column is the **Date** (the app tries to parse it).
4. Explore **Data Preview**, **Descriptive Stats**, and **Visualizations** (trend, heatmap).
5. Select your **Target** and any **Predictors**.
6. (Optional) Enable **Time Series Decomposition** (additive; period=12).
7. (Optional) Add **Date Features**, **Lag Features**, and **Rolling Means**.
8. (Optional) **Encode** categoricals (Label or One-Hot).
9. Pick a **Model** and **Forecast Horizon** → click **Run Forecast**.
10. Review the **Actual vs Predicted** chart, **Forecast Table**, and **Metrics** (ML models).
11. (Optional) Build a compact dataset and **Download CSV**.

---

## 🧠 Model Notes

- **ARIMA**: demo config `order=(5,1,0)`; consider grid/auto-ARIMA for production.
- **Exponential Smoothing**: additive trend/seasonality with `seasonal_periods=12` (monthly); adjust for your cadence.
- **ML models** use an iterative strategy:
  - We create `lag_k` features for the target (default 3 lags).
  - Forecasting rolls forward by feeding each prediction back into the lag window.
  - **Exogenous predictors** (your selected predictors) are kept at their **last observed values** unless you provide future values (not in this UI). For exogenous-aware forecasting, supply/engineer future predictor values.

---

## 📊 Metrics

- **ML models**: MSE & MAE (evaluated on a time-ordered 80/20 split; no shuffling)
- **TS models (ARIMA/ES)**: the demo focuses on forward forecasts rather than backtests. For rigorous evaluation, consider time-series CV (e.g., expanding windows).

---

## 🧪 Time Series Decomposition

- Additive decomposition into **Trend**, **Seasonality**, **Residual** (`period=12` by default).
- Requires a sufficiently long numeric series; the app will warn if not enough data.

---

## 🧯 Troubleshooting

- **No date column found** → select a candidate under “Select a Date Column” and let the app parse it.
- **Frequency not inferred** → app defaults to monthly start (`MS`).
- **Duplicate dates** → app aggregates duplicates (mean).
- **ARIMA fails to converge** → try different `order` or clean the series (stability, differencing).
- **Neural Net errors** → ensure TensorFlow is installed; otherwise use other models.
- **Correlation heatmap empty** → dataset may have no numeric columns (or all non-numeric after parsing).

---

## 🔒 Notes on Data Handling

- The app uses `st.cache_data` for faster file reloads during a session.
- Uploaded data is processed in memory; nothing is persisted unless you download the CSV.

---

## 🗺️ Roadmap / Ideas

- Configurable ARIMA `(p,d,q)` and seasonal params
- Advanced backtesting (TimeSeriesSplit)
- Prophet / LightGBM / XGBoost options
- Robust handling of **future exogenous** features
- Model persistence (joblib) and inference API (FastAPI)

---

## 👤 Author

**Vinay Mandora** · Manchester, UK
