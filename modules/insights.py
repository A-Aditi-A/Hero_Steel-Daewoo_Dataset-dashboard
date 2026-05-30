"""
insights.py — Business intelligence charts and insight cards (light theme, no emojis)
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def plot_shift_cost(df: pd.DataFrame, tariff_per_kwh: float = 8.0) -> go.Figure:
    grp = df.groupby("shift", observed=True).agg(
        total_kwh=("Usage_kWh", "sum"),
    ).reset_index()
    grp["cost_lakh"] = grp["total_kwh"] * tariff_per_kwh / 1e5
    colors = [ACCENT1, ACCENT4, ACCENT2, ACCENT3]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp["shift"], y=grp["cost_lakh"],
        marker_color=colors[:len(grp)],
        text=grp["cost_lakh"].apply(lambda x: f"Rs.{x:.1f}L"),
        textposition="outside",
        hovertemplate="%{x}<br>Cost: Rs.%{y:.2f} Lakh<extra></extra>",
    ))
    fig.update_layout(**LAYOUT,
                      title="Estimated Energy Cost by Production Shift",
                      xaxis_title="Shift", yaxis_title="Cost (Rs. Lakh)",
                      showlegend=False)
    return fig


def plot_pf_improvement(df: pd.DataFrame) -> go.Figure:
    pf_bins = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    pf_labs = ["<0.70","0.70-0.75","0.75-0.80","0.80-0.85",
               "0.85-0.90","0.90-0.95",">=0.95"]
    df2 = df.copy()
    df2["pf_bin"] = pd.cut(df2["Lagging_Current_Power_Factor"],
                           bins=pf_bins, labels=pf_labs, right=False)
    grp = df2.groupby("pf_bin", observed=True).agg(
        count=("Usage_kWh","count"),
        avg_kwh=("Usage_kWh","mean"),
    ).reset_index()
    bar_colors = [ACCENT3 if str(b) < "0.90" else ACCENT1 for b in grp["pf_bin"]]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=grp["pf_bin"], y=grp["count"],
        name="Reading Count", marker_color=bar_colors, opacity=0.85,
        hovertemplate="%{x}<br>Readings: %{y:,}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=grp["pf_bin"], y=grp["avg_kwh"],
        name="Avg kWh", mode="lines+markers",
        line=dict(color=ACCENT2, width=2.5), marker=dict(size=7),
    ), secondary_y=True)
    fig.add_vline(x=4.5, line_dash="dash", line_color=ACCENT1,
                  annotation_text="PF 0.90 target", annotation_font_color=ACCENT1)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=CARD_BG,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        title="Power Factor Distribution and Improvement Opportunity",
        barmode="group",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=GRID),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    fig.update_yaxes(title_text="Reading Count", secondary_y=False,
                     gridcolor=GRID, linecolor=GRID, color=TEXT)
    fig.update_yaxes(title_text="Avg Usage (kWh)", secondary_y=True, color=ACCENT2)
    return fig


def plot_peak_calendar(df: pd.DataFrame) -> go.Figure:
    daily = df.set_index("date")["Usage_kWh"].resample("D").max().reset_index()
    daily.columns = ["date", "peak_kwh"]
    daily["week"]     = daily["date"].dt.isocalendar().week.astype(int)
    daily["dow"]      = daily["date"].dt.dayofweek
    daily["dow_name"] = daily["date"].dt.day_name()
    day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    fig = go.Figure(go.Heatmap(
        x=daily["week"],
        y=daily["dow"],
        z=daily["peak_kwh"],
        colorscale=[[0,"#eaf4f2"],[0.5, "#f5c6a0"],[1.0, ACCENT3]],
        hovertemplate=(
            "Week: %{x}<br>Day: %{customdata}<br>"
            "Peak kWh: %{z:.2f}<extra></extra>"
        ),
        customdata=daily["dow_name"],
        colorbar=dict(title="Peak kWh", tickfont=dict(color=TEXT)),
    ))
    # Fix: apply base layout first, then update yaxis separately to avoid conflict
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=CARD_BG,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        title="Daily Peak Demand Risk Calendar (Full Year)",
        xaxis=dict(title="Week of Year", gridcolor=GRID, linecolor=GRID),
        legend=dict(bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    fig.update_yaxes(
        tickvals=list(range(7)),
        ticktext=day_labels,
        gridcolor=GRID,
        linecolor=GRID,
    )
    return fig


def plot_co2_trend(df: pd.DataFrame) -> go.Figure:
    weekly = df.set_index("date").resample("W").agg(
        total_co2=("CO2_tCO2","sum"),
        total_kwh=("Usage_kWh","sum"),
    ).reset_index()
    weekly["intensity"] = (weekly["total_co2"] / weekly["total_kwh"] * 1000).round(4)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=weekly["date"], y=weekly["total_co2"],
        name="Weekly CO2 (tCO2)", marker_color=ACCENT3, opacity=0.65,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=weekly["date"], y=weekly["intensity"],
        name="CO2 Intensity (g/kWh)", mode="lines",
        line=dict(color=ACCENT2, width=2.5),
    ), secondary_y=True)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=CARD_BG,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        title="Weekly CO2 Footprint and Intensity Trend",
        hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=GRID),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    fig.update_yaxes(title_text="CO2 (tCO2)",     secondary_y=False,
                     gridcolor=GRID, linecolor=GRID, color=TEXT)
    fig.update_yaxes(title_text="Intensity g/kWh", secondary_y=True, color=ACCENT2)
    return fig


def generate_insights(df: pd.DataFrame, anomaly_df: pd.DataFrame) -> list:
    insights = []

    low_pf_pct = (df["Lagging_Current_Power_Factor"] < 0.90).mean() * 100
    if low_pf_pct > 20:
        insights.append({
            "category": "Power Quality",
            "severity": "HIGH",
            "title": f"{low_pf_pct:.1f}% readings below target Power Factor (0.90)",
            "detail": (
                "Lagging power factor below 0.90 attracts utility penalty charges. "
                "Install capacitor banks at HT bus to correct reactive power draw. "
                "Estimated annual penalty savings: Rs. 12-18 Lakh at 35,000 kVAr correction."
            ),
            "color": ACCENT3,
        })

    peak_hrs = df[df["hour"].between(9, 11) | df["hour"].between(18, 21)]
    peak_avg = peak_hrs["Usage_kWh"].mean()
    off_avg  = df[df["hour"].between(0, 5)]["Usage_kWh"].mean()
    ratio    = peak_avg / max(off_avg, 0.01)
    if ratio > 2.5:
        insights.append({
            "category": "Load Management",
            "severity": "MEDIUM",
            "title": f"Peak-to-offpeak demand ratio: {ratio:.1f}x — Load shifting recommended",
            "detail": (
                f"Peak window (09:00-11:00, 18:00-21:00) draws {peak_avg:.1f} kWh avg vs "
                f"{off_avg:.1f} kWh in night hours. Shifting non-critical loads (grinding, "
                "water pumping, furnace charging) to off-peak reduces ToD tariff burden."
            ),
            "color": ACCENT2,
        })

    n_anom   = anomaly_df["IF_Anomaly"].sum()
    anom_kwh = anomaly_df.loc[anomaly_df["IF_Anomaly"], "Usage_kWh"].sum()
    norm_avg = anomaly_df.loc[~anomaly_df["IF_Anomaly"], "Usage_kWh"].mean()
    excess   = anom_kwh - (n_anom * norm_avg)
    cost_exc = excess * 8 / 1e5
    insights.append({
        "category": "Anomaly Impact",
        "severity": "HIGH" if cost_exc > 5 else "MEDIUM",
        "title": f"{n_anom:,} anomalous intervals detected — estimated excess cost: Rs. {cost_exc:.1f} Lakh",
        "detail": (
            f"Isolation Forest flagged {n_anom:,} readings ({n_anom/len(df)*100:.1f}% of total). "
            f"Excess energy consumed during these spikes: {excess:,.0f} kWh. "
            "Investigate faulty transformer taps, induction furnace arc instabilities, "
            "and rolling mill overloads as primary root causes."
        ),
        "color": ACCENT3 if cost_exc > 5 else ACCENT2,
    })

    total_co2 = df["CO2_tCO2"].sum()
    potential = total_co2 * 0.08
    insights.append({
        "category": "Sustainability",
        "severity": "LOW",
        "title": f"Total CO2: {total_co2:.1f} tCO2 — {potential:.1f} t reducible via optimisation",
        "detail": (
            "Power factor correction and peak load shifting can reduce plant energy consumption "
            f"by approx 8%, saving approximately {potential:.1f} tCO2 annually. "
            "This supports ISO 50001 compliance and ESG reporting targets."
        ),
        "color": ACCENT1,
    })

    wknd_avg = df[df["WeekStatus"]=="Weekend"]["Usage_kWh"].mean()
    wkdy_avg = df[df["WeekStatus"]=="Weekday"]["Usage_kWh"].mean()
    drop_pct = (1 - wknd_avg/wkdy_avg) * 100
    insights.append({
        "category": "Scheduling",
        "severity": "LOW",
        "title": f"Weekend load is {drop_pct:.1f}% lower — schedule maintenance windows optimally",
        "detail": (
            f"Weekday avg: {wkdy_avg:.2f} kWh vs Weekend avg: {wknd_avg:.2f} kWh. "
            "Planned maintenance (furnace relining, roll changes, scale pit cleaning) "
            "should be concentrated on weekends to maximise production uptime on weekdays."
        ),
        "color": ACCENT4,
    })
    return insights
