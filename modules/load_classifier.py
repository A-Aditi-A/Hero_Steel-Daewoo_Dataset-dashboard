"""
load_classifier.py — Random Forest load type classifier (light theme, no emojis)
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

BG      = "#f5f7fa"
CARD_BG = "#ffffff"
ACCENT1 = "#00897b"
ACCENT2 = "#d4650a"
ACCENT3 = "#c0392b"
ACCENT4 = "#1565c0"
TEXT    = "#1e2329"
GRID    = "#e0e4ec"

LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD_BG,
    font=dict(family="Inter, sans-serif", color=TEXT, size=12),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=GRID),
    margin=dict(l=50, r=20, t=50, b=40),
)

FEATURES = [
    "Lagging_Current_Reactive_Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2_tCO2",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "Apparent_Power_kVA",
    "Power_Factor_Avg",
    "NSM",
    "hour",
    "month",
]

LOAD_COLORS = {"Light_Load": ACCENT4, "Medium_Load": ACCENT2, "Maximum_Load": ACCENT3}
CLASS_ORDER = ["Light_Load", "Medium_Load", "Maximum_Load"]


@st.cache_data(show_spinner=False)
def train_classifier(df: pd.DataFrame):
    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["Load_Type"]
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y_enc = le.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return model, le, X_test, y_test, report, cm


def plot_confusion_matrix(cm: np.ndarray, labels: list) -> go.Figure:
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    text   = [[f"{cm[i,j]}<br>({cm_pct[i,j]:.1f}%)" for j in range(len(labels))]
              for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(
        z=cm_pct,
        x=[l.replace("_", " ") for l in labels],
        y=[l.replace("_", " ") for l in labels],
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#eaf4f2"], [0.5, "rgba(0,137,123,0.40)"], [1, ACCENT1]],
        showscale=True,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>%{z:.1f}%<extra></extra>",
        colorbar=dict(title="% of Actual", tickfont=dict(color=TEXT)),
    ))
    fig.update_layout(**LAYOUT, title="Confusion Matrix — Load Type Classifier",
                      xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def plot_feature_importance(model, feature_names: list) -> go.Figure:
    importances = model.feature_importances_
    idx = np.argsort(importances)
    short = {
        "Lagging_Current_Reactive_Power_kVarh": "Lag Reactive Power",
        "Leading_Current_Reactive_Power_kVarh": "Lead Reactive Power",
        "CO2_tCO2": "CO2 Emissions",
        "Lagging_Current_Power_Factor": "Lag Power Factor",
        "Leading_Current_Power_Factor": "Lead Power Factor",
        "Apparent_Power_kVA": "Apparent Power",
        "Power_Factor_Avg": "Avg Power Factor",
        "NSM": "Seconds from Midnight",
        "hour": "Hour of Day",
        "month": "Month",
    }
    labels = [short.get(feature_names[i], feature_names[i]) for i in idx]
    values = importances[idx]
    bar_colors = [
        ACCENT1 if v >= np.percentile(importances, 75) else
        ACCENT4 if v >= np.percentile(importances, 50) else "#b0b8c4"
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT, title="Feature Importance — Random Forest",
                      xaxis_title="Importance Score", yaxis_title="", height=400)
    return fig


def plot_class_metrics(report: dict) -> go.Figure:
    classes = ["Light_Load", "Medium_Load", "Maximum_Load"]
    metrics = ["precision", "recall", "f1-score"]
    colors  = [ACCENT1, ACCENT2, ACCENT4]
    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        vals = [report[c][metric] for c in classes]
        fig.add_trace(go.Bar(
            x=[c.replace("_", " ") for c in classes],
            y=vals, name=metric.capitalize(),
            marker_color=color, opacity=0.85,
        ))
    fig.update_layout(**LAYOUT, title="Per-Class Model Performance",
                      xaxis_title="Load Class", yaxis_title="Score", barmode="group")
    fig.update_yaxes(range=[0, 1.05])
    return fig


def predict_single(model, le, input_vals: dict) -> dict:
    row   = pd.DataFrame([input_vals])[FEATURES]
    probs = model.predict_proba(row)[0]
    pred  = le.inverse_transform([np.argmax(probs)])[0]
    return {"prediction": pred, "probabilities": dict(zip(le.classes_, probs.round(3)))}
