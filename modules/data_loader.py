"""
data_loader.py — Cached loader + feature engineering for steel_energy.csv
"""
import os
import streamlit as st
import pandas as pd
import numpy as np


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "steel_energy.csv")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── Temporal features ──────────────────────────────────────────────────────
    df["hour"]        = df["date"].dt.hour
    df["month"]       = df["date"].dt.month
    df["month_name"]  = df["date"].dt.strftime("%b")
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["date_only"]   = df["date"].dt.date
    df["shift"]       = pd.cut(
        df["hour"],
        bins=[-1, 5, 13, 21, 23],
        labels=["Night (00-05)", "Morning (06-13)", "Afternoon (14-21)", "Evening (22-23)"]
    )

    # ── Derived KPIs ───────────────────────────────────────────────────────────
    df["Apparent_Power_kVA"] = np.sqrt(
        df["Usage_kWh"] ** 2 +
        df["Lagging_Current_Reactive_Power_kVarh"] ** 2
    ).round(3)

    df["Power_Factor_Avg"] = (
        (df["Lagging_Current_Power_Factor"] + df["Leading_Current_Power_Factor"]) / 2
    ).round(3)

    df["Energy_Intensity"] = (df["Usage_kWh"] / df["Apparent_Power_kVA"]).round(3)

    # ── Rolling stats (1-hour = 4 intervals of 15-min) ────────────────────────
    df["Usage_RollingMean_1h"]  = df["Usage_kWh"].rolling(4, min_periods=1).mean().round(3)
    df["Usage_RollingStd_1h"]   = df["Usage_kWh"].rolling(4, min_periods=1).std().fillna(0).round(3)

    # ── Z-score for spike detection ───────────────────────────────────────────
    mu  = df["Usage_kWh"].mean()
    sig = df["Usage_kWh"].std()
    df["Usage_ZScore"] = ((df["Usage_kWh"] - mu) / sig).round(3)

    # ── Load type encoding ─────────────────────────────────────────────────────
    load_map = {"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2}
    df["Load_Code"] = df["Load_Type"].map(load_map)

    return df


def get_summary_kpis(df: pd.DataFrame) -> dict:
    """Return high-level plant KPIs."""
    total_kwh   = df["Usage_kWh"].sum()
    total_co2   = df["CO2_tCO2"].sum()
    avg_pf      = df["Power_Factor_Avg"].mean()
    peak_kwh    = df["Usage_kWh"].max()
    anomaly_pct = (df["Usage_ZScore"].abs() > 3.0).mean() * 100

    # Energy cost estimate (₹8/kWh industrial tariff)
    energy_cost_lakh = total_kwh * 8 / 1e5

    return {
        "total_kwh":        round(total_kwh, 1),
        "total_co2":        round(total_co2, 2),
        "avg_pf":           round(avg_pf, 3),
        "peak_kwh":         round(peak_kwh, 2),
        "anomaly_pct":      round(anomaly_pct, 2),
        "energy_cost_lakh": round(energy_cost_lakh, 1),
    }
