# 🏭 Hero Steel — Operational Intelligence Dashboard

> **Industry 4.0 Energy Analytics System** for Steel Manufacturing  
> Built on DAEWOO Steel Co. Ltd. sensor data schema · Python · Streamlit · Scikit-learn · Plotly

---

## 🎯 Project Overview

This is a professional, end-to-end **Operational Intelligence Dashboard** designed for steel plant management. It transforms raw energy and sensor data into actionable insights covering:

- ⚡ Real-time energy monitoring and load profiling
- 🚨 ML-powered anomaly detection (Isolation Forest)
- 🤖 Load type classification (Random Forest, ~99% accuracy)
- 💡 Executive-level business recommendations with cost impact analysis
- 🌿 CO₂ footprint tracking and sustainability metrics

---

## 📁 Project Structure

```
Hero Steel/
├── app.py                          # Main Streamlit application (5 pages)
├── requirements.txt                # Python dependencies
├── data/
│   ├── generate_data.py            # Synthetic dataset generator (DAEWOO schema)
│   └── steel_energy.csv            # Generated dataset (35,040 rows, 15-min intervals)
└── modules/
    ├── __init__.py
    ├── data_loader.py              # Cached loader + feature engineering
    ├── eda.py                      # 7 interactive EDA charts
    ├── anomaly_detection.py        # Isolation Forest + 4 anomaly charts
    ├── load_classifier.py          # Random Forest classifier + live predictor
    └── insights.py                 # Business intelligence + 4 cost/CO₂ charts
```

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python data/generate_data.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Executive Dashboard** | 6 KPI cards · Energy timeline · Load distribution · Monthly CO₂ · Anomaly snapshot |
| 📊 **Energy Explorer** | Hourly heatmap · Power Factor scatter · Correlation matrix · Data table |
| 🚨 **Anomaly Detection** | IF time-series · Score distribution · Anomaly heatmap · 3D scatter |
| 🤖 **Load Classifier** | Confusion matrix · Feature importance · Per-class metrics · Live predictor |
| 💡 **Business Insights** | Severity-ranked recommendations · Shift costs · Peak calendar · CO₂ trend |

---

## 🧠 ML Models

### Isolation Forest (Anomaly Detection)
- **Features**: Usage kWh, Reactive Power (lag/lead), CO₂, Power Factor, Apparent Power
- **Contamination**: 1.2% (configurable via sidebar slider)
- **Output**: Binary anomaly flag + continuous anomaly score

### Random Forest (Load Classification)
- **Target**: `Light_Load` | `Medium_Load` | `Maximum_Load`
- **Features**: 10 electrical + temporal features
- **Performance**: ~99% accuracy on 20% holdout test set
- **Extras**: Live predictor form for real-time classification

---

## 📐 Dataset Schema (DAEWOO Steel Co. Ltd.)

| Column | Description | Unit |
|--------|-------------|------|
| `date` | 15-minute timestamp | datetime |
| `Usage_kWh` | Active energy consumption | kWh |
| `Lagging_Current_Reactive_Power_kVarh` | Lagging reactive power | kVArh |
| `Leading_Current_Reactive_Power_kVarh` | Leading reactive power | kVArh |
| `CO2_tCO2` | Carbon emissions | tCO₂ |
| `Lagging_Current_Power_Factor` | Lagging power factor | 0–1 |
| `Leading_Current_Power_Factor` | Leading power factor | 0–1 |
| `NSM` | Seconds from midnight | s |
| `WeekStatus` | Weekday / Weekend | categorical |
| `Day_of_week` | Day name | categorical |
| `Load_Type` | Load classification | Light / Medium / Maximum |

---

## 💡 Key Business Insights Generated

1. **⚡ Power Quality** — Flags PF < 0.90 readings → capacitor bank recommendation with ₹ savings estimate  
2. **📉 Load Management** — Peak-to-offpeak ratio analysis → Time-of-Day tariff reduction strategy  
3. **🚨 Anomaly Impact** — Excess energy cost from flagged anomalies in ₹ Lakh  
4. **🌿 Sustainability** — CO₂ reduction potential via operational optimisation (ISO 50001 aligned)  
5. **📅 Scheduling** — Weekend load gap → maintenance window optimisation

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI Framework | Streamlit 1.35 |
| Visualisation | Plotly 5.22 |
| ML | Scikit-learn 1.5 (IsolationForest, RandomForest) |
| Data | Pandas 2.2, NumPy 1.26 |
| Statistics | SciPy 1.13 |

---

*Built as an Industry 4.0 operational analytics demonstration for Hero Steels Limited on 28 May 2026.*
