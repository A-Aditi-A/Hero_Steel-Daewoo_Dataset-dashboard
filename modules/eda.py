"""
eda.py — Exploratory Data Analysis charts (light theme, no emojis)
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

BG       = "#f5f7fa"
CARD_BG  = "#ffffff"
ACCENT1  = "#00897b"
ACCENT2  = "#d4650a"
ACCENT3  = "#c0392b"
ACCENT4  = "#1565c0"
TEXT     = "#1e2329"
GRID     = "#e0e4ec"

LOAD_COLORS = {
    "Light_Load":   "#1565c0",
    "Medium_Load":  "#d4650a",
    "Maximum_Load": "#c0392b",
}

LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="Inter, sans-serif", color=TEXT, size=12),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=GRID),
    margin=dict(l=50, r=20, t=50, b=40),
)


def _apply(fig: go.Figure) -> go.Figure:
    fig.update_layout(**LAYOUT)
    return fig


def plot_energy_timeseries(df: pd.DataFrame, resample: str = "D") -> go.Figure:
    ts = df.set_index("date")["Usage_kWh"].resample(resample).mean().reset_index()
    ts.columns = ["date", "Usage_kWh"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["date"], y=ts["Usage_kWh"],
        mode="lines", name="Avg kWh",
        line=dict(color=ACCENT1, width=2),
        fill="tozeroy",
        fillcolor="rgba(0,137,123,0.10)",
    ))
    fig.update_layout(title="Plant Energy Consumption Over Time",
                      xaxis_title="Date", yaxis_title="kWh (avg per interval)",
                      hovermode="x unified")
    return _apply(fig)


def plot_load_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["Load_Type"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.55,
        marker=dict(colors=[LOAD_COLORS.get(l, ACCENT4) for l in counts.index],
                    line=dict(color="#ffffff", width=3)),
        textfont=dict(size=13, color=TEXT),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(title="Load Type Distribution", showlegend=True,
                      legend=dict(orientation="h", y=-0.1))
    return _apply(fig)


def plot_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.groupby(["Day_of_week", "hour"])["Usage_kWh"].mean().unstack(fill_value=0)
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, "#eaf4f2"], [0.5, ACCENT1], [1.0, ACCENT3]],
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Avg kWh: %{z:.2f}<extra></extra>",
        colorbar=dict(title="kWh", tickfont=dict(color=TEXT)),
    ))
    fig.update_layout(title="Energy Heatmap — Hour x Day of Week",
                      xaxis_title="Hour", yaxis_title="")
    return _apply(fig)


def plot_monthly_summary(df: pd.DataFrame) -> go.Figure:
    mo = df.groupby("month").agg(
        total_kwh=("Usage_kWh", "sum"),
        total_co2=("CO2_tCO2", "sum"),
    ).reset_index()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    mo["month_name"] = mo["month"].apply(lambda x: month_labels[x-1])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=mo["month_name"], y=mo["total_kwh"],
        name="Total kWh", marker_color=ACCENT1, opacity=0.85,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=mo["month_name"], y=mo["total_co2"],
        name="CO2 (tCO2)", mode="lines+markers",
        line=dict(color=ACCENT3, width=2.5), marker=dict(size=7),
    ), secondary_y=True)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=CARD_BG,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        title="Monthly Energy Consumption and CO2 Footprint",
        hovermode="x unified", barmode="group",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=GRID),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    fig.update_yaxes(title_text="Total kWh", secondary_y=False,
                     gridcolor=GRID, linecolor=GRID, color=TEXT)
    fig.update_yaxes(title_text="CO2 (tCO2)", secondary_y=True,
                     gridcolor=GRID, color=ACCENT3)
    return fig


def plot_power_factor(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(3000, len(df)), random_state=7)
    fig = px.scatter(
        sample,
        x="Lagging_Current_Power_Factor",
        y="Usage_kWh",
        color="Load_Type",
        color_discrete_map=LOAD_COLORS,
        opacity=0.55,
        labels={"Lagging_Current_Power_Factor": "Lagging Power Factor",
                "Usage_kWh": "Energy (kWh)"},
        title="Power Factor vs Energy Draw by Load Type",
    )
    fig.add_vline(x=0.90, line_dash="dash", line_color=ACCENT2,
                  annotation_text="PF = 0.90 (target)", annotation_font_color=ACCENT2)
    return _apply(fig)


def plot_weekday_vs_weekend(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["hour", "WeekStatus"])["Usage_kWh"].mean().reset_index()
    fig = go.Figure()
    for status, color in [("Weekday", ACCENT1), ("Weekend", ACCENT2)]:
        sub = grp[grp["WeekStatus"] == status]
        fig.add_trace(go.Scatter(
            x=sub["hour"], y=sub["Usage_kWh"],
            name=status, mode="lines+markers",
            line=dict(color=color, width=2.5), marker=dict(size=5),
        ))
    fig.update_layout(
        title="Weekday vs Weekend — Hourly Load Profile",
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=2,
                   gridcolor=GRID, linecolor=GRID),
        yaxis_title="Avg kWh", hovermode="x unified",
    )
    return _apply(fig)


def plot_correlation(df: pd.DataFrame) -> go.Figure:
    cols = ["Usage_kWh", "Lagging_Current_Reactive_Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh", "CO2_tCO2",
            "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor",
            "Apparent_Power_kVA", "Power_Factor_Avg"]
    corr = df[cols].corr().round(2)
    short = ["Usage", "Lag React", "Lead React", "CO2",
             "Lag PF", "Lead PF", "App Power", "PF Avg"]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=short, y=short,
        colorscale=[[0, ACCENT3], [0.5, "#f5f7fa"], [1, ACCENT1]],
        zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Corr: %{z:.2f}<extra></extra>",
        colorbar=dict(title="r", tickfont=dict(color=TEXT)),
    ))
    fig.update_layout(title="Feature Correlation Matrix")
    return _apply(fig)
