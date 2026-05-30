"""
app.py — Hero Steel Operational Intelligence Dashboard
Industry 4.0 Energy Analytics | Light Theme | No Emojis
"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Hero Steel | Operational Intelligence",
    page_icon="assets/favicon.ico" if False else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.data_loader      import load_data, get_summary_kpis
from modules.eda              import (plot_energy_timeseries, plot_load_distribution,
                                      plot_hourly_heatmap, plot_monthly_summary,
                                      plot_power_factor, plot_weekday_vs_weekend,
                                      plot_correlation)
from modules.anomaly_detection import (run_isolation_forest, plot_anomaly_timeseries,
                                        plot_anomaly_distribution, plot_anomaly_heatmap,
                                        plot_anomaly_scatter, get_anomaly_stats)
from modules.load_classifier   import (train_classifier, plot_confusion_matrix,
                                        plot_feature_importance, plot_class_metrics,
                                        predict_single, FEATURES, CLASS_ORDER)
from modules.insights          import (plot_shift_cost, plot_pf_improvement,
                                        plot_peak_calendar, plot_co2_trend,
                                        generate_insights)

# ── Global CSS (white / light theme) ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #f5f7fa; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e0e4ec;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p { color: #6b7280 !important; font-size: 12px; }

/* ── KPI Cards — fixed equal height ── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e0e4ec;
    border-top: 4px solid var(--accent, #00897b);
    border-radius: 10px;
    padding: 16px 12px;
    text-align: center;
    height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s, transform 0.2s;
    box-sizing: border-box;
}
.kpi-card:hover {
    box-shadow: 0 4px 16px rgba(0,137,123,0.12);
    transform: translateY(-2px);
}
.kpi-label {
    color: #6b7280; font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 6px; line-height: 1.3;
}
.kpi-value {
    color: #1e2329; font-size: 24px; font-weight: 700;
    line-height: 1.1; margin-bottom: 4px;
}
.kpi-sub { color: #9ca3af; font-size: 10px; }

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0 6px 0;
    border-bottom: 2px solid #e0e4ec;
    margin-bottom: 18px;
}
.section-header h2 { color: #1e2329; font-size: 16px; font-weight: 600; margin: 0; }
.section-badge {
    background: #e6f4f2; color: #00897b;
    padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 500;
}

/* ── Insight cards ── */
.insight-card {
    background: #ffffff;
    border: 1px solid #e0e4ec;
    border-left: 4px solid var(--sev-color, #00897b);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.insight-title { color: #1e2329; font-weight: 600; font-size: 14px; margin-bottom: 6px; }
.insight-detail { color: #6b7280; font-size: 13px; line-height: 1.6; }
.insight-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 10px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px;
}

/* ── Tabs ── */
[data-testid="stTabs"] button { color: #6b7280 !important; font-weight: 500; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00897b !important;
    border-bottom: 2px solid #00897b !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #e0e4ec; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading plant data..."):
    df_raw = load_data()
kpis = get_summary_kpis(df_raw)

with st.spinner("Running ML models..."):
    df_anom = run_isolation_forest(df_raw)
    model, le, X_test, y_test, report, cm = train_classifier(df_raw)

anom_stats = get_anomaly_stats(df_anom)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px 0;'>
        <div style='font-size:36px; color:#00897b; font-weight:900;'>HS</div>
        <div style='color:#1e2329; font-weight:700; font-size:16px; margin-top:4px;'>Hero Steel</div>
        <div style='color:#6b7280; font-size:11px;'>Operational Intelligence v1.0</div>
    </div>
    <hr style='border-color:#e0e4ec; margin:12px 0;'/>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")
    page = st.radio("", [
        "Executive Dashboard",
        "Energy Explorer",
        "Anomaly Detection",
        "Load Classifier",
        "Business Insights",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#e0e4ec; margin:12px 0;'/>", unsafe_allow_html=True)
    st.markdown("**Filters**")
    month_opts = ["All"] + list(df_raw["month_name"].unique())
    sel_month  = st.selectbox("Month", month_opts)
    sel_load   = st.selectbox("Load Type", ["All"] + CLASS_ORDER)
    sel_week   = st.selectbox("Week Status", ["All", "Weekday", "Weekend"])

    st.markdown("<hr style='border-color:#e0e4ec; margin:12px 0;'/>", unsafe_allow_html=True)
    st.markdown("**Anomaly Settings**")
    contamination = st.slider("IF Contamination %", 0.5, 5.0, 1.2, 0.1) / 100

    st.markdown("<hr style='border-color:#e0e4ec; margin:12px 0;'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#9ca3af; font-size:10px; line-height:2;'>
    Dataset: DAEWOO Steel Co. Ltd.<br>
    Period: Jan-Dec 2018<br>
    Records: 35,040 rows<br>
    Models: Isolation Forest + Random Forest
    </div>
    """, unsafe_allow_html=True)

# ── Apply filters ─────────────────────────────────────────────────────────────
df  = df_raw.copy()
df_a = df_anom.copy()
if sel_month != "All":
    df   = df[df["month_name"]    == sel_month]
    df_a = df_a[df_a["month_name"] == sel_month]
if sel_load != "All":
    df   = df[df["Load_Type"]    == sel_load]
    df_a = df_a[df_a["Load_Type"] == sel_load]
if sel_week != "All":
    df   = df[df["WeekStatus"]   == sel_week]
    df_a = df_a[df_a["WeekStatus"] == sel_week]

# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi_card(label, value, sub="", accent="#00897b"):
    st.markdown(f"""
    <div class='kpi-card' style='--accent:{accent};'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

def section(title, badge=""):
    badge_html = f"<span class='section-badge'>{badge}</span>" if badge else ""
    st.markdown(f"""
    <div class='section-header'>
        <h2>{title}</h2>{badge_html}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Dashboard":
    st.markdown("""
    <h1 style='color:#1e2329; font-size:24px; font-weight:700; margin:16px 0 4px 0;'>
        Hero Steel — Operational Intelligence Dashboard
    </h1>
    <p style='color:#6b7280; font-size:13px; margin-bottom:24px;'>
        Real-time plant energy analytics powered by ML &nbsp;|&nbsp; Industry 4.0 &nbsp;|&nbsp; DAEWOO Schema
    </p>""", unsafe_allow_html=True)

    section("Plant-Wide KPIs", "Jan-Dec 2018")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi_card("Total Energy",    f"{kpis['total_kwh']/1e3:.1f}k kWh", "Full year",        "#00897b")
    with c2: kpi_card("Energy Cost",     f"Rs.{kpis['energy_cost_lakh']:.0f}L","@ Rs.8/kWh",      "#d4650a")
    with c3: kpi_card("Avg Power Factor",f"{kpis['avg_pf']:.3f}",              "Target >= 0.90",   "#1565c0")
    with c4: kpi_card("Peak Demand",     f"{kpis['peak_kwh']:.1f} kWh",        "Max 15-min",       "#c0392b")
    with c5: kpi_card("Total CO2",       f"{kpis['total_co2']:.0f} t",         "tCO2 equivalent",  "#00897b")
    with c6: kpi_card("Anomaly Rate",    f"{kpis['anomaly_pct']:.1f}%",        f"{anom_stats['n_anomaly']:,} flags","#c0392b")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])
    with col_l: st.plotly_chart(plot_energy_timeseries(df), use_container_width=True)
    with col_r: st.plotly_chart(plot_load_distribution(df), use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2: st.plotly_chart(plot_weekday_vs_weekend(df), use_container_width=True)
    with col_r2: st.plotly_chart(plot_monthly_summary(df),    use_container_width=True)

    section("Anomaly Snapshot", f"{anom_stats['n_anomaly']:,} flags detected")
    ca1,ca2,ca3,ca4 = st.columns(4)
    with ca1: kpi_card("Anomalous Intervals", f"{anom_stats['n_anomaly']:,}",         f"{anom_stats['pct_anomaly']}% of total","#c0392b")
    with ca2: kpi_card("Avg Anomaly Load",    f"{anom_stats['avg_anomaly_kwh']} kWh", "Per interval",                          "#d4650a")
    with ca3: kpi_card("Avg Normal Load",     f"{anom_stats['avg_normal_kwh']} kWh",  "Per interval",                          "#00897b")
    with ca4: kpi_card("Excess Factor",       f"{anom_stats['excess_factor']}x",      "Anomaly vs Normal",                     "#c0392b")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ENERGY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Energy Explorer":
    st.markdown("<h1 style='color:#1e2329;'>Energy Explorer</h1>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Power Factor", "Correlation", "Monthly"])
    with tab1: st.plotly_chart(plot_hourly_heatmap(df),  use_container_width=True)
    with tab2: st.plotly_chart(plot_power_factor(df),    use_container_width=True)
    with tab3: st.plotly_chart(plot_correlation(df),     use_container_width=True)
    with tab4: st.plotly_chart(plot_monthly_summary(df), use_container_width=True)

    section("Filtered Data Table", f"{len(df):,} rows")
    disp_cols = ["date","Usage_kWh","Lagging_Current_Power_Factor",
                 "CO2_tCO2","Load_Type","WeekStatus","shift","Usage_ZScore"]
    st.dataframe(df[disp_cols].sort_values("date", ascending=False).head(500),
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Detection":
    st.markdown("<h1 style='color:#1e2329;'>Anomaly Detection Engine</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#6b7280;'>Isolation Forest · contamination={contamination*100:.1f}% · {anom_stats['n_anomaly']:,} anomalies flagged</p>",
                unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("Anomalous Readings", f"{anom_stats['n_anomaly']:,}",         f"{anom_stats['pct_anomaly']}% of total","#c0392b")
    with c2: kpi_card("Avg Anomaly kWh",    f"{anom_stats['avg_anomaly_kwh']}",     "Per interval",                          "#d4650a")
    with c3: kpi_card("Avg Normal kWh",     f"{anom_stats['avg_normal_kwh']}",      "Per interval",                          "#00897b")
    with c4: kpi_card("Excess Multiplier",  f"{anom_stats['excess_factor']}x",      "Anomaly / Normal",                      "#c0392b")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1,tab2,tab3,tab4 = st.tabs(["Time Series", "Score Distribution", "Heatmap", "3D Scatter"])
    with tab1: st.plotly_chart(plot_anomaly_timeseries(df_a),   use_container_width=True)
    with tab2: st.plotly_chart(plot_anomaly_distribution(df_a), use_container_width=True)
    with tab3: st.plotly_chart(plot_anomaly_heatmap(df_a),      use_container_width=True)
    with tab4: st.plotly_chart(plot_anomaly_scatter(df_a),      use_container_width=True)

    section("Top Anomalous Events")
    top_anom = (df_a[df_a["IF_Anomaly"]]
                .nlargest(20, "Usage_kWh")
                [["date","Usage_kWh","CO2_tCO2","Lagging_Current_Power_Factor",
                  "Load_Type","shift","IF_Score"]]
                .reset_index(drop=True))
    st.dataframe(top_anom, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LOAD CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Load Classifier":
    st.markdown("<h1 style='color:#1e2329;'>ML Load Type Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280;'>Random Forest · 200 trees · 80/20 train-test split</p>",
                unsafe_allow_html=True)

    overall_acc = report["accuracy"]
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("Overall Accuracy", f"{overall_acc*100:.1f}%",                  "Test set",          "#00897b")
    with c2: kpi_card("Light Load F1",    f"{report['Light_Load']['f1-score']:.3f}",  "Precision/Recall",  "#1565c0")
    with c3: kpi_card("Medium Load F1",   f"{report['Medium_Load']['f1-score']:.3f}", "Precision/Recall",  "#d4650a")
    with c4: kpi_card("Max Load F1",      f"{report['Maximum_Load']['f1-score']:.3f}","Precision/Recall",  "#c0392b")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l: st.plotly_chart(plot_confusion_matrix(cm, le.classes_), use_container_width=True)
    with col_r: st.plotly_chart(plot_feature_importance(model, FEATURES), use_container_width=True)
    st.plotly_chart(plot_class_metrics(report), use_container_width=True)

    section("Live Load Predictor", "Enter sensor readings below")
    with st.form("predictor_form"):
        p1,p2,p3 = st.columns(3)
        with p1:
            lag_react  = st.number_input("Lag Reactive Power (kVArh)", 0.0, 50.0, 8.5, 0.1)
            lead_react = st.number_input("Lead Reactive Power (kVArh)", 0.0, 10.0, 1.2, 0.1)
            co2_val    = st.number_input("CO2 (tCO2)", 0.0, 0.5, 0.05, 0.001, format="%.4f")
        with p2:
            lag_pf    = st.number_input("Lagging PF", 0.60, 1.0, 0.87, 0.01)
            lead_pf   = st.number_input("Leading PF", 0.75, 1.0, 0.95, 0.01)
            app_power = st.number_input("Apparent Power (kVA)", 0.0, 90.0, 10.5, 0.1)
        with p3:
            pf_avg    = st.number_input("Avg Power Factor", 0.60, 1.0, 0.91, 0.01)
            nsm_val   = st.number_input("Seconds from Midnight", 0, 86400, 32400, 900)
            hour_val  = st.number_input("Hour of Day", 0, 23, 9)
            month_val = st.number_input("Month (1-12)", 1, 12, 6)
        submitted = st.form_submit_button("Predict Load Type", use_container_width=True)
        if submitted:
            inp = {
                "Lagging_Current_Reactive_Power_kVarh": lag_react,
                "Leading_Current_Reactive_Power_kVarh": lead_react,
                "CO2_tCO2": co2_val,
                "Lagging_Current_Power_Factor": lag_pf,
                "Leading_Current_Power_Factor": lead_pf,
                "Apparent_Power_kVA": app_power,
                "Power_Factor_Avg": pf_avg,
                "NSM": nsm_val,
                "hour": hour_val,
                "month": month_val,
            }
            result = predict_single(model, le, inp)
            pred   = result["prediction"]
            probs  = result["probabilities"]
            color_map = {"Light_Load":"#1565c0","Medium_Load":"#d4650a","Maximum_Load":"#c0392b"}
            color = color_map.get(pred, "#00897b")
            st.markdown(f"""
            <div class='kpi-card' style='--accent:{color}; max-width:360px; margin:16px auto; height:auto; padding:20px;'>
                <div class='kpi-label'>Predicted Load Type</div>
                <div class='kpi-value' style='color:{color};'>{pred.replace("_"," ")}</div>
                <div class='kpi-sub'>
                    Light: {probs.get('Light_Load',0):.1%} &nbsp;|&nbsp;
                    Medium: {probs.get('Medium_Load',0):.1%} &nbsp;|&nbsp;
                    Max: {probs.get('Maximum_Load',0):.1%}
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Business Insights":
    st.markdown("<h1 style='color:#1e2329;'>Business Insights and Recommendations</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280;'>AI-generated operational recommendations for plant management</p>",
                unsafe_allow_html=True)

    section("Actionable Insights", "Auto-generated")
    insights = generate_insights(df, df_a)
    sev_colors = {"HIGH": "#c0392b", "MEDIUM": "#d4650a", "LOW": "#00897b"}
    sev_bg     = {"HIGH": "#fdf0ef", "MEDIUM": "#fdf6ee", "LOW": "#eaf4f2"}

    for ins in insights:
        sc = sev_colors.get(ins["severity"], "#00897b")
        sb = sev_bg.get(ins["severity"], "#eaf4f2")
        st.markdown(f"""
        <div class='insight-card' style='--sev-color:{ins["color"]};'>
            <span class='insight-badge' style='background:{sb}; color:{sc};'>
                {ins["severity"]} &nbsp;·&nbsp; {ins["category"]}
            </span>
            <div class='insight-title'>{ins["title"]}</div>
            <div class='insight-detail'>{ins["detail"]}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Supporting Analysis")
    tab1,tab2,tab3,tab4 = st.tabs(["Shift Costs","Power Factor","Peak Calendar","CO2 Trend"])
    with tab1: st.plotly_chart(plot_shift_cost(df),     use_container_width=True)
    with tab2: st.plotly_chart(plot_pf_improvement(df), use_container_width=True)
    with tab3: st.plotly_chart(plot_peak_calendar(df),  use_container_width=True)
    with tab4: st.plotly_chart(plot_co2_trend(df),      use_container_width=True)
