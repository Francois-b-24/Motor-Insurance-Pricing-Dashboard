# 🏎️ Motor Insurance Pricing Dashboard

https://insuranalytics.streamlit.app/

A comprehensive pricing actuary toolkit built with Streamlit, demonstrating end-to-end insurance pricing capabilities from portfolio analysis to model monitoring.

## 📋 Features

| Tab | Description |
|-----|-------------|
| **📊 Portfolio Overview** | Exposure distribution, frequency analysis, risk heatmaps, segmentation by driver age, vehicle characteristics, bonus-malus |
| **🎯 GLM Pricing Model** | Poisson GLM for frequency with relativities, Gamma GLM for severity, coefficient analysis, premium simulator |
| **💰 Pure Premium** | Frequency × Severity decomposition, pure premium analysis by segment, model vs actual comparison |
| **🤖 GLM vs XGBoost** | Model benchmarking, lift curves, double lift analysis, SHAP interpretability |
| **📈 Model Monitoring** | A/E ratios, PSI drift detection, residual analysis, segment-level monitoring |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (recommended for compatibility with statsmodels and scipy)
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd pricing

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Deploy on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository → `app.py`
5. Click "Deploy!"

## 📊 Dataset

**freMTPL2freq + freMTPL2sev** — French Motor Third-Party Liability Insurance  
- ~670k policies  
- Standard actuarial benchmark datasets (available via OpenML)  
- **Variables:**
  - Driver: age, bonus-malus
  - Vehicle: power, age, fuel type, brand
  - Geographic: area (density), region
  - Exposure: policy exposure (years), claim count, claim amounts

The datasets are automatically downloaded from OpenML when you first run the application.

## 🏗️ Project Structure

```
pricing/
├── app.py                 # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Data loading and preprocessing
│   └── models.py          # GLM and XGBoost models
├── requirements.txt       # Python dependencies
├── README.md
└── .gitignore
```

## 🛠️ Tech Stack

- **Streamlit** — Interactive web dashboard
- **Statsmodels** — Poisson GLM (frequency) + Gamma GLM (severity)
- **XGBoost** — Gradient boosting with Poisson objective
- **SHAP** — Model interpretability and feature importance
- **Plotly** — Interactive visualizations
- **scikit-learn** — Data splitting and evaluation metrics
- **Pandas/NumPy** — Data manipulation

## 📈 Key Metrics

- **Gini Coefficient** — Model discrimination power
- **A/E Ratio** — Actual vs Expected (calibration)
- **PSI** — Population Stability Index (drift detection)
- **Pure Premium** — Frequency × Severity

## 🔍 Model Details

### Frequency Model
- **GLM**: Poisson regression with log link
- **XGBoost**: Gradient boosting with Poisson objective
- Features: vehicle power/age, driver age, bonus-malus, area density, fuel type, region

### Severity Model
- **GLM**: Gamma regression with log link (fitted on claims only)
- Features: same as frequency model

### Pure Premium
- **Formula**: Pure Premium = Frequency × Severity
- Both GLM and XGBoost frequency models are combined with GLM severity

## 📝 Notes

- Models are cached using Streamlit's caching mechanisms for faster reloads
- The application automatically handles missing severity data
- All visualizations are interactive (Plotly)

## 👤 Author

**François** — Actuarial Data Scientist  
Specializing in motor insurance pricing, GLM modeling, and ML applications in insurance.

## 📄 License

This project is open source and available for educational and research purposes.
