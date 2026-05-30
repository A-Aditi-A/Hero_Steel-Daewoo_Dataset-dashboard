"""
anomaly_detection.py — Isolation Forest anomaly detection (light theme, no emojis)
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import streamlit as st

BG      = "#f5f7fa"
CARD_BG = "#ffffff"
ACCENT1 = "#00897b"
ACCENT2 = "#d4650a"
ACCENT3 = "#c0392b"
ACCENT4 = "#1565c0"
TEXT    = "#1e2329"
GRID    = "#e0e4ec"

FEATURES = [
    "Usage_kWh",
    "Lagging_Current_Reactive_Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2_tCO2",
    "Lagging_Current_Power_Factor",
    "Apparent_Power_kVA",
]

LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD_BG,
    font=dict(family="Inter, sans-serif", color=TEXT, size=12),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=GRID),
    margin=dict(l=50, r=20, t=50, b=40),
)


@st.cache_data(show_spinner=False)
def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.012) -> pd.DataFrame:
    X = df[FEATURES].fillna(df[FEATURES].median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42, n_jobs=-1)
    preds  = model.fit_predict(Xs)
    scores = model.score_samples(Xs)
    result = df.copy()
    result["IF_Anomaly"] = preds == -1
    result["IF_Score"]   = scores.round(4)
    return result


def zscore_anomalies(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    result = df.copy()
    result["ZS_Anomaly"] = result["Usage_ZScore"].abs() > threshold
    return result


def plot_anomaly_timeseries(df: pd.DataFrame) -> go.Figure:
    normal  = df[~df["IF_Anomaly"]]
    anomaly = df[ df["IF_Anomaly"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["date"], y=normal["Usage_kWh"],
        mode="lines", name="Normal",
        line=dict(color=ACCENT1, width=1.2), opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=anomaly["date"], y=anomaly["Usage_kWh"],
        mode="markers", name="Anomaly",
        marker=dict(color=ACCENT3, size=6, symbol="x",
                    line=dict(width=1, color=ACCENT3)),
    ))
    fig.update_layout(**LAYOUT,
                      title="Anomaly Detection — Isolation Forest",
                      xaxis_title="Date", yaxis_title="Energy (kWh)",
                      hovermode="x unified")
    return fig


def plot_anomaly_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color, name in [(False, ACCENT1, "Normal"), (True, ACCENT3, "Anomaly")]:
        sub = df[df["IF_Anomaly"] == label]["IF_Score"]
        fig.add_trace(go.Histogram(
            x=sub, name=name, marker_color=color, opacity=0.75, nbinsx=60,
        ))
    fig.update_layout(**LAYOUT,
                      title="Isolation Forest Score Distribution",
                      xaxis_title="Anomaly Score (lower = more abnormal)",
                      yaxis_title="Count", barmode="overlay")
    return fig


def plot_anomaly_heatmap(df: pd.DataFrame) -> go.Figure:
    adf = df[df["IF_Anomaly"]].copy()
    adf["day"] = adf["date"].dt.day_name()
    pivot = adf.groupby(["day", "hour"]).size().unstack(fill_value=0)
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, "#f5f7fa"], [0.5, "#f5a07a"], [1.0, ACCENT3]],
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Anomalies: %{z}<extra></extra>",
        colorbar=dict(title="Count", tickfont=dict(color=TEXT)),
    ))
    fig.update_layout(**LAYOUT,
                      title="Anomaly Frequency — Hour x Day of Week",
                      xaxis_title="Hour", yaxis_title="")
    return fig


def plot_anomaly_scatter(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(5000, len(df)), random_state=42)
    colors = sample["IF_Anomaly"].map({False: ACCENT1, True: ACCENT3})
    fig = go.Figure(go.Scatter3d(
        x=sample["Usage_kWh"],
        y=sample["Lagging_Current_Reactive_Power_kVarh"],
        z=sample["CO2_tCO2"],
        mode="markers",
        marker=dict(size=3, color=colors, opacity=0.65),
        text=sample["IF_Anomaly"].map({False: "Normal", True: "Anomaly"}),
        hovertemplate=(
            "Usage: %{x:.2f} kWh<br>"
            "Reactive: %{y:.2f} kVArh<br>"
            "CO2: %{z:.4f} tCO2<br>"
            "Status: %{text}<extra></extra>"
        ),
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        scene=dict(
            bgcolor=CARD_BG,
            xaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID,
                       title="Usage kWh", color=TEXT),
            yaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID,
                       title="Reactive kVArh", color=TEXT),
            zaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID,
                       title="CO2 tCO2", color=TEXT),
        ),
        font=dict(family="Inter, sans-serif", color=TEXT),
        title="3D Anomaly Scatter — Energy x Reactive x CO2",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def get_anomaly_stats(df: pd.DataFrame) -> dict:
    total    = len(df)
    n_anom   = df["IF_Anomaly"].sum()
    pct      = round(n_anom / total * 100, 2)
    avg_anom = df.loc[df["IF_Anomaly"], "Usage_kWh"].mean()
    avg_norm = df.loc[~df["IF_Anomaly"], "Usage_kWh"].mean()
    return {
        "total": total,
        "n_anomaly": int(n_anom),
        "pct_anomaly": pct,
        "avg_anomaly_kwh": round(avg_anom, 2),
        "avg_normal_kwh":  round(avg_norm, 2),
        "excess_factor":   round(avg_anom / avg_norm, 2),
    }
