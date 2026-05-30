"""
Synthetic Steel Industry Energy Consumption Dataset Generator
Mirrors the DAEWOO Steel Co. Ltd. dataset schema used on Kaggle.
Columns: date, Usage_kWh, Lagging_Current_Reactive_Power_kVarh,
         Leading_Current_Reactive_Power_kVarh, CO2_tCO2,
         Lagging_Current_Power_Factor, Leading_Current_Power_Factor,
         NSM, WeekStatus, Day_of_week, Load_Type
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Date range: Jan 2018 – Dec 2018 at 15-min intervals ─────────────────────
date_range = pd.date_range(start="2018-01-01 00:00:00",
                           end="2018-12-31 23:45:00",
                           freq="15min")
n = len(date_range)

# ── Time features ─────────────────────────────────────────────────────────────
hour   = date_range.hour
dow    = date_range.dayofweek          # 0=Mon … 6=Sun
month  = date_range.month
nsm    = (hour * 3600 + date_range.minute * 60).astype(int)

weekstatus   = np.where(dow < 5, "Weekday", "Weekend")
day_of_week  = date_range.day_name()

# ── Shift / load profile ──────────────────────────────────────────────────────
# Blast-furnace / rolling-mill load pattern
is_night   = (hour >= 22) | (hour < 6)
is_peak    = (hour >= 8)  & (hour < 20)
is_weekend = dow >= 5

base_load = 3.5
peak_mult = np.where(is_peak & ~is_weekend, 3.2,
            np.where(is_night, 0.8, 1.8))

seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * (month - 1) / 12)  # slight winter peak

noise = np.random.normal(0, 0.25, n)

usage_kwh = np.clip(base_load * peak_mult * seasonal + noise, 0.5, 30.0)

# ── Power-quality features ────────────────────────────────────────────────────
lag_reactive  = usage_kwh * np.random.uniform(0.3, 0.55, n) + np.random.normal(0, 0.3, n)
lead_reactive = usage_kwh * np.random.uniform(0.02, 0.10, n) + np.random.normal(0, 0.1, n)

lag_reactive  = np.clip(lag_reactive, 0, None)
lead_reactive = np.clip(lead_reactive, 0, None)

lag_pf  = np.clip(0.82 + 0.10 * (1 - lag_reactive / (usage_kwh + 1)) +
                  np.random.normal(0, 0.02, n), 0.60, 1.00)
lead_pf = np.clip(0.95 + np.random.normal(0, 0.02, n), 0.75, 1.00)

co2 = usage_kwh * 0.0045 * (1 + np.random.normal(0, 0.05, n))
co2 = np.clip(co2, 0, None)

# Ensure mutable numpy arrays before anomaly injection
usage_kwh    = np.array(usage_kwh,    dtype=float)
lag_reactive = np.array(lag_reactive, dtype=float)
co2          = np.array(co2,          dtype=float)

# ── Load classification (business rule matching real dataset) ─────────────────
load_type = np.select(
    [usage_kwh < 4.5, usage_kwh < 11.0],
    ["Light_Load", "Medium_Load"],
    default="Maximum_Load"
)

# ── Inject realistic anomalies (≈1 % of rows) ────────────────────────────────
anom_idx = np.random.choice(n, size=int(n * 0.012), replace=False)
usage_kwh[anom_idx]   *= np.random.uniform(1.8, 3.5, len(anom_idx))
lag_reactive[anom_idx] *= np.random.uniform(1.5, 2.5, len(anom_idx))
co2[anom_idx]          *= np.random.uniform(1.8, 3.2, len(anom_idx))

# Re-clip after anomaly injection
usage_kwh    = np.clip(usage_kwh, 0.5, 90)
lag_reactive = np.clip(lag_reactive, 0, 50)
co2          = np.clip(co2, 0, 0.5)

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    "date":                                   date_range,
    "Usage_kWh":                              usage_kwh.round(3),
    "Lagging_Current_Reactive_Power_kVarh":   lag_reactive.round(3),
    "Leading_Current_Reactive_Power_kVarh":   lead_reactive.round(3),
    "CO2_tCO2":                               co2.round(5),
    "Lagging_Current_Power_Factor":           lag_pf.round(2),
    "Leading_Current_Power_Factor":           lead_pf.round(2),
    "NSM":                                    nsm,
    "WeekStatus":                             weekstatus,
    "Day_of_week":                            day_of_week,
    "Load_Type":                              load_type,
})

out_path = os.path.join(os.path.dirname(__file__), "steel_energy.csv")
df.to_csv(out_path, index=False)
print(f"[OK] Dataset saved: {out_path}  ({len(df):,} rows)")
