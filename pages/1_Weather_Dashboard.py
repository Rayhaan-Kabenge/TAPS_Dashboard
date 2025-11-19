# weather_dashboard_complete.py
# ---------------------------------------------------
# KSUTAPS • Complete Weather Dashboard + AI (Original + Enhancements)
# Preserves ALL original features + adds decision support, alerts, caching
# ---------------------------------------------------
import os
import json
import datetime as dt
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

from src.data_loader import load_climate, load_climate_with_et
from src.plotting import (
    et_reference_figure,
    gdd_cumulative_line,
    gdd_daily_bar,
    forecast_daily_temp_band,
    rainfall_bar_chart,
    temperature_calendar_heatmap,
    water_balance_figure,
    weather_trend_figure,
)
from src.utils import download_button_for_figure, json_default
from src.weather_calcs import (
    agg_df,
    cached_request,
    calculate_irrigation_decision,
    compute_et,
    compute_gdd_columns,
    compute_water_balance,
    generate_smart_alerts,
    summarize_weather_period,
    temperature_calendar,
)
from src import weather_tools


# =========================
# API KEYS (Replace before deployment!)
# =========================

# secrets
API_KEY = st.secrets["open_weather_map"]
OPENAI_API_KEY = st.secrets["open_ai_key"]
OPENAI_MODELS_TRY = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

# =========================
# PAGE SETUP & ENHANCED CSS
# =========================
st.set_page_config(page_title="KSUTAPS • Complete Weather Dashboard", layout="wide")
st.markdown("""
<style>
:root{
  --card-r:14px; --shadow:0 1px 6px rgba(0,0,0,.06);
  --shadow-lg:0 6px 24px rgba(0,0,0,.12);
}
.block {border:1px solid #e6e6e6; border-radius:12px; padding:1rem; background:#fafafa; margin-bottom:0.75rem;}
.small-metric > div[data-testid="stMetricValue"] { font-size:1.3rem; }
details[data-testid="stExpander"] summary { font-weight:600; }
.card { border:1px solid #e7e7e7; border-radius:var(--card-r); padding:1rem; background:#fff; box-shadow:var(--shadow); }
.card:hover { box-shadow:var(--shadow-lg); }
.hero { border-radius:18px; padding:1rem; color:#0b0b0b; position:relative; overflow:hidden; }
.hero-ai{ border-radius:18px; padding:1rem 1.25rem; background:linear-gradient(135deg,#eef5ff,#f7fbff);
  border:1px solid #e4eeff; box-shadow:var(--shadow); }
.decision-card { border: 3px solid #28a745; border-radius: 16px; padding: 1.5rem;
  background: linear-gradient(135deg, #e8f5e9, #f1f8f4); margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.alert-card { border-left: 6px solid #dc3545; border-radius: 8px; padding: 1rem;
  background: #fff3cd; margin: 0.5rem 0; }
.info-alert { border-left-color: #17a2b8; background: #d1ecf1; }
.warn-alert { border-left-color: #ffc107; background: #fff3cd; }
.success-alert { border-left-color: #28a745; background: #d4edda; }
.chip { display:inline-block; padding:.25rem .6rem; border-radius:999px; font-size:.85rem; font-weight:600; margin-right:.35rem; }
.chip.danger { color:#721c24; background:#f8d7da; }
.chip.warn   { color:#856404; background:#fff3cd; }
.chip.info   { color:#0c5460; background:#d1ecf1; }
.chip.ok     { color:#155724; background:#d4edda; }
.chip.rain   { color:#004085; background:#cce5ff; }
.daycard { border:1px solid #eee; border-radius:12px; padding:.75rem; background:#fafafa; text-align:center; }
.daycard:hover { background:#f2f7ff; border-color:#d7e6ff; }
.wind-arrow { width:36px; height:36px; display:inline-block; }
.help-text { font-size:0.85rem; color:#666; font-style:italic; margin-top:0.25rem; }
.section-desc { font-size:0.9rem; color:#555; background:#f8f9fa; padding:0.5rem; border-radius:6px; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# =========================
# CONFIGURATION
# =========================
#
DEFAULT_CLIMATE_PATH = "Files/ET_analysis_Data1.xlsx"
DEFAULT_SHEET = "Climate_Data"
DEFAULT_STATION = "Colby, KS"
ELEVATION_M = 965.0
LATITUDE_RAD = 0.688

BASE_URL_CURRENT = "http://api.openweathermap.org/data/2.5/weather"
BASE_URL_FORECAST = "http://api.openweathermap.org/data/2.5/forecast"

COL_DATE = "Date"
COL_TMIN = "Tmin(°c)"
COL_TMAX = "Tmax(°c)"
COL_RHMIN = "Rhmin(%)"
COL_RHMAX = "Rhmax(%)"
COL_U2 = "U2(m s-1)"
COL_RS = "Rs(MJ m-2)"
COL_PCP = "Precip.(mm)"
REQUIRED = [COL_DATE, COL_TMIN, COL_TMAX, COL_RHMIN, COL_RHMAX, COL_U2, COL_RS, COL_PCP]
CLIMATE_NUMERIC_COLS = [COL_TMIN, COL_TMAX, COL_RHMIN, COL_RHMAX, COL_U2, COL_RS, COL_PCP]

COL_GDD_STD_DAILY = "GDD (°C·d)"
COL_GDD_FP_DAILY = "FromPlanting_GDD (°C·d)"
COL_GDD_FP_CUM = "ΣGDD (from planting)"


# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.title("⚙️ Weather Controls")
st.sidebar.markdown('<p class="help-text">Choose between bundled sample data or upload your own workbook.</p>', unsafe_allow_html=True)

with st.sidebar.expander("📂 Data Source", expanded=True):
    data_mode = st.radio("Data input", ["Sample dataset", "Upload workbook"], index=0)
    sheet_name = st.text_input("Sheet name", value=DEFAULT_SHEET)
    station_label = st.text_input("Station label", value=DEFAULT_STATION)
    if data_mode == "Upload workbook":
        climate_file = st.file_uploader("Climate workbook", type=["xlsx", "xls"])
    else:
        climate_file = DEFAULT_CLIMATE_PATH

if data_mode == "Upload workbook" and climate_file is None:
    st.sidebar.info("Upload a workbook or switch to the bundled sample dataset.")
    st.stop()

st.sidebar.markdown(f"**Station:** {station_label}")

with st.sidebar.expander("📍 Station Settings", expanded=False):
    st.markdown('<p class="help-text">Adjust for your specific location</p>', unsafe_allow_html=True)
    elevation_m = st.number_input("Elevation (m)", value=float(ELEVATION_M), step=1.0,
                                  help="Site elevation above mean sea level - affects ET calculations")
    latitude_rad = st.number_input("Latitude (radians)", value=float(LATITUDE_RAD), step=0.001,
                                   help="Site latitude in radians (39.43° ≈ 0.688)")

with st.sidebar.expander("🌱 Crop Settings", expanded=True):
    st.markdown('<p class="help-text">Growth stage affects irrigation thresholds</p>', unsafe_allow_html=True)
    growth_stage = st.selectbox("Growth Stage",
                                ["vegetative", "reproductive", "grain_fill", "maturity"],
                                index=0,
                                help="Current crop development stage")
    gdd_base = st.number_input("GDD Base (°C)", value=10.0, step=0.5,
                               help="Base temperature for GDD calculation")
    gdd_cap = st.number_input("GDD Cap (°C)", value=30.0, step=0.5,
                              help="Upper temperature threshold for GDD")
    planting_default = dt.date(2024, 5, 8)
    planting_date = st.date_input("Planting Date", value=planting_default,
                                  help="Date when crop was planted")
    use_harvest = st.checkbox("Set Harvest Date?", value=False)
    harvest_date = st.date_input("Harvest Date", value=dt.date(2024, 10, 31)) if use_harvest else None
    gdd_target = st.number_input("Target cumulative GDD", value=0.0, step=50.0,
                                 help="Set to compare current accumulation vs. seasonal target (optional)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Date Range")
st.sidebar.markdown('<p class="help-text">Select time period for analysis</p>', unsafe_allow_html=True)
date_preset = st.sidebar.radio("Quick select:",
                               ["Last 7 days", "Last 30 days", "Season to date", "All"],
                               index=2)
custom = st.sidebar.checkbox("Custom range")
start_date = end_date = None
if custom:
    start_date = st.sidebar.date_input("Start date")
    end_date = st.sidebar.date_input("End date")

st.sidebar.markdown("---")
with st.sidebar.expander("💧 What-If Scenarios", expanded=True):
    st.markdown('<p class="help-text">Simulate irrigation planning scenarios</p>', unsafe_allow_html=True)
    simulated_irrigation_in = st.slider("Planned Irrigation (inches)", 0.0, 3.0, 0.0, 0.1,
                                       help="Amount of irrigation to simulate")
    irrigation_cost_per_inch = st.slider("Cost ($/acre-inch)", 10.0, 50.0, 25.0, 5.0,
                                        help="Cost per acre-inch of water")
    st.markdown("**Weather Adjustments**")
    sim_temp = st.slider("Δ Temperature (°C)", -10, 10, 0,
                        help="Adjust forecast temperature up/down")
    sim_rh = st.slider("Δ Humidity (%)", -50, 50, 0,
                      help="Adjust forecast humidity up/down")

st.sidebar.markdown("---")
view_mode = st.sidebar.radio("📊 Display Mode", ["Graphs", "Cards"], index=0,
                             help="Choose between graphical or card-based view")

plot_vars = st.sidebar.multiselect(
    "Variables to Plot",
    [COL_TMAX, COL_TMIN, COL_PCP, COL_U2, COL_RHMAX, COL_RHMIN, COL_RS,
     "ETo (mm)", "ETr (mm)", COL_GDD_STD_DAILY, COL_GDD_FP_CUM],
    default=[COL_TMAX, COL_TMIN, COL_PCP],
    help="Select multiple variables to overlay on charts"
)

roll_toggle = st.sidebar.checkbox("Show rolling average",
                                  help="Smooth data with moving average")
roll_window = st.sidebar.select_slider("Rolling window (days)", options=[7, 14, 30], value=7,
                                      disabled=not roll_toggle)
cum_toggle = st.sidebar.checkbox("Show cumulative (Precip/ET)",
                                 help="Display cumulative totals over time")
agg_choice = st.sidebar.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"], index=0,
                                  help="Time period for data summarization")
allow_download = st.sidebar.checkbox("Show download buttons", value=True)

# =========================
# LOAD AND PROCESS DATA
# =========================
try:
    df = load_climate(
        climate_file,
        sheet_name,
        required_columns=REQUIRED,
        date_column=COL_DATE,
        numeric_columns=CLIMATE_NUMERIC_COLS,
    )
except Exception as e:
    st.error(f"❌ Could not read data: {e}")
    st.stop()

df_et = compute_et(
    df,
    elevation_m,
    latitude_rad,
    rs_col=COL_RS,
    tmax_col=COL_TMAX,
    tmin_col=COL_TMIN,
    rhmax_col=COL_RHMAX,
    rhmin_col=COL_RHMIN,
    wind_col=COL_U2,
    date_col=COL_DATE,
)
pdate = pd.Timestamp(planting_date)
hdate = pd.Timestamp(harvest_date) if harvest_date else None
df_full = compute_gdd_columns(
    df_et,
    date_col=COL_DATE,
    tmin_col=COL_TMIN,
    tmax_col=COL_TMAX,
    tbase=gdd_base,
    tcap=gdd_cap,
    planting_date=pdate,
    harvest_date=hdate,
    gdd_daily_col=COL_GDD_STD_DAILY,
    gdd_fp_daily_col=COL_GDD_FP_DAILY,
    gdd_fp_cum_col=COL_GDD_FP_CUM,
)

# Date filtering
dmin, dmax = df_full[COL_DATE].min().date(), df_full[COL_DATE].max().date()
if date_preset == "Last 7 days":
    start, end = dmax - dt.timedelta(days=6), dmax
elif date_preset == "Last 30 days":
    start, end = dmax - dt.timedelta(days=29), dmax
elif date_preset == "Season to date":
    season_start = dt.date(dmax.year, 4, 1)
    start, end = max(season_start, dmin), dmax
else:
    start, end = dmin, dmax
if custom and start_date and end_date:
    start, end = start_date, end_date

m = (df_full[COL_DATE].dt.date >= start) & (df_full[COL_DATE].dt.date <= end)
dfx = df_full.loc[m].copy()

# Aggregation
if agg_choice == "Weekly":
    dfa = agg_df(
        dfx,
        COL_DATE,
        "W",
        cols_sum=[COL_PCP, "ETo (mm)", "ETr (mm)", COL_GDD_STD_DAILY, COL_GDD_FP_DAILY],
        cols_mean=[COL_TMAX, COL_TMIN, COL_U2, COL_RHMAX, COL_RHMIN, COL_RS],
    )
elif agg_choice == "Monthly":
    dfa = agg_df(
        dfx,
        COL_DATE,
        "MS",
        cols_sum=[COL_PCP, "ETo (mm)", "ETr (mm)", COL_GDD_STD_DAILY, COL_GDD_FP_DAILY],
        cols_mean=[COL_TMAX, COL_TMIN, COL_U2, COL_RHMAX, COL_RHMIN, COL_RS],
    )
else:
    dfa = dfx.copy()

if COL_GDD_FP_DAILY in dfa.columns:
    dfa[COL_GDD_FP_CUM] = dfa[COL_GDD_FP_DAILY].cumsum()

# Precompute irrigation decision inputs for rain tab
recent_precip = dfx[COL_PCP].tail(7).sum() if COL_PCP in dfx.columns else 0.0
recent_eto = dfx["ETo (mm)"].tail(7).sum() if "ETo (mm)" in dfx.columns else 0.0
decision = calculate_irrigation_decision(
    recent_precip_mm=float(recent_precip),
    eto_mm=float(recent_eto),
    growth_stage=growth_stage,
    cost_per_inch=irrigation_cost_per_inch,
)

# Apply rolling average if enabled
if roll_toggle:
    for col in [COL_TMAX, COL_TMIN, COL_U2, COL_RHMAX, COL_RHMIN, COL_RS, "ETo (mm)", "ETr (mm)", COL_GDD_STD_DAILY, COL_GDD_FP_DAILY]:
        if col in plot_vars and col in dfa.columns:
            dfa[col] = dfa[col].rolling(roll_window, min_periods=1).mean()

# Apply cumulative if enabled
if cum_toggle:
    for col in [COL_PCP, "ETo (mm)", "ETr (mm)"]:
        if col in plot_vars and col in dfa.columns:
            dfa[col] = dfa[col].cumsum()

# =========================
# FETCH FORECAST DATA (with caching)
# =========================
city_used = st.session_state.get("owm_last_city", "Colby")
fc_data, fc_source = cached_request(BASE_URL_FORECAST,
                                     {"appid": API_KEY, "q": city_used, "units": "metric"})

dff = pd.DataFrame()
if fc_data and fc_data.get("cod") == "200":
    rows = []
    for it in fc_data.get("list", []):
        m = it.get("main", {})
        w = it.get("weather", [{}])[0]
        wind = it.get("wind", {}) or {}
        rows.append({
            "Datetime": pd.to_datetime(it.get("dt_txt")),
            "Temperature": (m.get("temp") or 0) + sim_temp,
            "Temp_Min": (m.get("temp_min") or 0) + sim_temp,
            "Temp_Max": (m.get("temp_max") or 0) + sim_temp,
            "Humidity": min(100, max(0, (m.get("humidity") or 0) + sim_rh)),
            "POP": (it.get("pop", 0.0) or 0.0) * 100.0,
            "Rain_3h": it.get("rain", {}).get("3h", 0.0) or 0.0,
            "Wind": wind.get("speed", 0.0) or 0.0,
            "WindDeg": wind.get("deg", 0) or 0,
            "Clouds": it.get("clouds", {}).get("all", 0),
            "Icon": w.get("icon", ""),
            "Desc": w.get("description", "").title(),
        })
    dff = pd.DataFrame(rows).sort_values("Datetime").reset_index(drop=True)

# =========================
# MAIN DASHBOARD - TABS
# =========================
tab_data, tab_et, tab_rain, tab_owm, tab_ai = st.tabs([
    "📊 Weather Overview",
    "💧 Reference ET",
    "🌧️ Rain & Irrigation",
    "🌍 Current & Forecast",
    "🤖 AI Insights",
])

# ===================================================
# TAB 1: DATA & CHARTS (ORIGINAL + ENHANCEMENTS)
# ===================================================
with tab_data:
    st.title("📈 Weather Data & Analysis")
    st.markdown('<p class="section-desc">Historical weather data with ET calculations, GDD tracking, and customizable visualizations. Use sidebar controls to filter dates and select variables.</p>', unsafe_allow_html=True)
    
    # =========================
    # NEW! IRRIGATION DECISION CARD
    # =========================
    # st.markdown("### 💧 Irrigation Decision Support")
    # st.markdown('<p class="help-text">AI-powered recommendation based on recent weather, ET demand, and crop stage</p>', unsafe_allow_html=True)
    
    # recent_precip = dfx[COL_PCP].tail(7).sum() if len(dfx) >= 7 else 0
    # recent_eto = dfx["ETo (mm)"].tail(7).sum() if len(dfx) >= 7 and "ETo (mm)" in dfx.columns else 0
    
    # decision = calculate_irrigation_decision(
    #     recent_precip_mm=recent_precip,
    #     eto_mm=recent_eto,
    #     growth_stage=growth_stage,
    #     cost_per_inch=irrigation_cost_per_inch
    # )
    
    # # Add what-if simulation
    # simulated_deficit = decision["water_deficit_mm"] - (simulated_irrigation_in * 25.4)
    # simulated_cost = simulated_irrigation_in * irrigation_cost_per_inch
    
    # st.markdown('<div class="decision-card">', unsafe_allow_html=True)
    # col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    # with col1:
    #     if decision["irrigate"]:
    #         st.markdown("### ✅ IRRIGATE RECOMMENDED")
    #         st.markdown(f"**Apply: {decision['inches_needed']} inches**")
    #     else:
    #         st.markdown("### ⏸️ NO IRRIGATION NEEDED")
    #         st.markdown("**Sufficient moisture**")
    #     st.caption("Based on 7-day water balance")
    # with col2:
    #     st.metric("Water Deficit", f"{decision['water_deficit_mm']:.1f} mm",
    #               delta=f"ETo {decision['eto_mm']:.1f} - Rain {decision['effective_rain_mm']:.1f}")
    # with col3:
    #     st.metric("Cost Estimate", f"${decision['cost_usd']:.2f}/acre",
    #               delta=f"{decision['inches_needed']} in × ${irrigation_cost_per_inch}")
    # with col4:
    #     st.metric("Growth Stage", growth_stage.replace("_", " ").title())
    
    # st.markdown("**Reasoning:**")
    # for reason in decision["reasoning"]:
    #     st.markdown(f"• {reason}")
    
    # =========================
    # NEW! SMART ALERTS
    # =========================
    st.markdown("### 🚨 Smart Alerts (Next 48 Hours)")
    st.markdown('<p class="help-text">Automated alerts for frost, heat, wind, rain, and ideal spray windows</p>', unsafe_allow_html=True)
    
    alerts = generate_smart_alerts(dff)
    
    if alerts:
        alert_cols = st.columns(min(len(alerts), 3))
        for idx, alert in enumerate(alerts[:3]):
            with alert_cols[idx % 3]:
                alert_class = f"{alert['type']}-alert"
                st.markdown(f'<div class="alert-card {alert_class}">', unsafe_allow_html=True)
                st.markdown(f"**{alert['icon']} {alert['title']}**")
                st.markdown(f"{alert['message']}")
                st.caption(f"→ {alert['action']}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        if len(alerts) > 3:
            with st.expander(f"View {len(alerts) - 3} more alerts"):
                for alert in alerts[3:]:
                    st.markdown(f"**{alert['icon']} {alert['title']}:** {alert['message']}")
                    st.caption(f"→ {alert['action']}")
    else:
        st.info("✅ No critical alerts. Weather conditions are favorable for field operations.")
    
    st.markdown("---")
    
    # =========================
    # PERIOD SUMMARY METRICS
    # =========================
    st.markdown("### 📊 Period Summary")
    st.markdown('<p class="help-text">Key statistics for the selected date range</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="block">', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Date Range", f"{start} → {end}")
        with c2:
            st.metric("Records", int(len(dfx)))
        with c3:
            expected_days = (end - start).days + 1
            present_days = dfx[COL_DATE].dt.date.nunique()
            completeness = (present_days / max(expected_days, 1)) * 100
            st.metric("Completeness", f"{completeness:.0f}%")
        with c4:
            st.metric("Total Precip", f"{dfx[COL_PCP].sum():.1f} mm")
        with c5:
            if "ETo (mm)" in dfx.columns:
                total_eto = dfx["ETo (mm)"].sum()
                deficit = total_eto - dfx[COL_PCP].sum()
                st.metric("Water Deficit", f"{deficit:.1f} mm",
                          delta=f"ETo: {total_eto:.1f} mm")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # =========================
    # DISPLAY MODE: CARDS OR GRAPHS
    # =========================
    if view_mode == "Cards":
        st.markdown("### 📋 Summary Cards")
        st.markdown('<p class="help-text">Average values for all weather variables in the selected period</p>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: st.metric("Mean Tmax", f"{dfx[COL_TMAX].mean():.1f} °C")
        with c2: st.metric("Mean Tmin", f"{dfx[COL_TMIN].mean():.1f} °C")
        with c3: st.metric("Mean Wind", f"{dfx[COL_U2].mean():.1f} m/s")
        with c4: st.metric("Mean RHmax", f"{dfx[COL_RHMAX].mean():.0f} %")
        with c5: st.metric("Mean RHmin", f"{dfx[COL_RHMIN].mean():.0f} %")
        with c6: st.metric("Mean Solar", f"{dfx[COL_RS].mean():.1f} MJ/m²")
        st.caption("💡 Tip: Switch to **Graphs** mode in the sidebar for detailed time-series visualization.")
        trend_fig = None
    else:
        st.markdown("### 📈 Weather Trends")
        st.markdown('<p class="help-text">Multi-variable overlay chart with dual axes for temperature/variables (left) and precipitation (right)</p>', unsafe_allow_html=True)
        trend_cols = [v for v in plot_vars if v != COL_PCP and v in dfa.columns]
        trend_fig = weather_trend_figure(dfa, COL_DATE, COL_PCP, trend_cols, cumulative=cum_toggle)
        if len(dfa):
            trend_fig.update_xaxes(range=[dfa[COL_DATE].min(), dfa[COL_DATE].max()])
        st.plotly_chart(trend_fig, width="stretch")
        st.caption(f"📊 Data source: {getattr(climate_file, 'name', 'upload')} | Aggregation: {agg_choice} | Points: {len(dfa)}")
        if allow_download and trend_fig is not None:
            download_button_for_figure(trend_fig, filename="weather_trends.png")

    st.markdown("---")
    st.markdown("### 🗓️ Historical Temperature Calendar")
    st.markdown('<p class="help-text">Heatmap showing average daily temperature by day-of-year across all years in the dataset</p>', unsafe_allow_html=True)

    temp_var_to_plot = st.radio("Temperature Variable:", (COL_TMAX, COL_TMIN), index=0, horizontal=True, key="temp_calendar_selector")
    pivot_table = temperature_calendar(df_et, COL_DATE, temp_var_to_plot)
    if not pivot_table.empty:
        fig_heatmap = temperature_calendar_heatmap(pivot_table, f"Historical Average Daily {temp_var_to_plot}")
        st.plotly_chart(fig_heatmap, width="stretch")
        if allow_download:
            download_button_for_figure(fig_heatmap, filename="avg_daily_temp_heatmap.png")

    # =========================
    # GDD ANALYSIS (ORIGINAL)
    # =========================
    st.markdown("---")
    st.markdown("### 🌡️ Growing Degree Days (GDD)")
    st.markdown('<p class="help-text">GDD accumulation from planting date using base and cap temperatures set in the sidebar</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if COL_GDD_FP_DAILY in dfa.columns:
            st.subheader("Daily GDD (from planting)")
            fig_gdd_daily = gdd_daily_bar(dfa, COL_DATE, COL_GDD_FP_DAILY)
            st.plotly_chart(fig_gdd_daily, width="stretch")
    with c2:
        if COL_GDD_FP_CUM in dfa.columns:
            st.subheader("Cumulative GDD (from planting)")
            fig_gdd_cum = gdd_cumulative_line(dfa, COL_DATE, COL_GDD_FP_CUM)
            st.plotly_chart(fig_gdd_cum, width="stretch")

    # Current GDD status
    if COL_GDD_FP_CUM in dfx.columns and len(dfx) > 0:
        current_gdd = dfx[COL_GDD_FP_CUM].iloc[-1]
        days_since_planting = (dfx[COL_DATE].iloc[-1] - pdate).days
        st.info(f"📊 **Current GDD Status:** {current_gdd:.1f}°C·day accumulated over {days_since_planting} days since planting ({planting_date})")

# ===================================================
# TAB 2: REFERENCE ET (ORIGINAL)
# ===================================================
with tab_et:
    st.title("💧 Reference Evapotranspiration (ETo & ETr)")
    st.markdown('<p class="section-desc">ASCE Standardized Penman-Monteith calculations for short (grass/ETo) and tall (alfalfa/ETr) reference crops. Use these values to estimate crop water use when multiplied by crop coefficients (Kc).</p>', unsafe_allow_html=True)
    
    try:
        df_et = load_climate_with_et(
            climate_file,
            sheet_name,
            required_columns=REQUIRED,
            date_column=COL_DATE,
            numeric_columns=CLIMATE_NUMERIC_COLS,
            elevation_m=elevation_m,
            latitude_rad=latitude_rad,
            rs_col=COL_RS,
            tmax_col=COL_TMAX,
            tmin_col=COL_TMIN,
            rhmax_col=COL_RHMAX,
            rhmin_col=COL_RHMIN,
            wind_col=COL_U2,
        )
    except Exception as e:
        st.error(f"ET calculation error: {e}")
        st.stop()
    
    et_view = st.radio("Display Mode:", ["Daily", "Rolling (7d)", "Monthly totals", "Cumulative since start"],
                      index=0, horizontal=True,
                      help="Choose how to aggregate/smooth ET data")
    dfe = df_et.copy()
    if et_view == "Rolling (7d)":
        dfe["ETo (mm)"] = dfe["ETo (mm)"].rolling(7, min_periods=1).mean()
        dfe["ETr (mm)"] = dfe["ETr (mm)"].rolling(7, min_periods=1).mean()
    elif et_view == "Monthly totals":
        dfe = agg_df(dfe, COL_DATE, "MS", cols_sum=["ETo (mm)", "ETr (mm)"])
    elif et_view == "Cumulative since start":
        dfe["ETo (mm)"] = dfe["ETo (mm)"].cumsum()
        dfe["ETr (mm)"] = dfe["ETr (mm)"].cumsum()
    
    fig_et = et_reference_figure(dfe, COL_DATE)
    st.plotly_chart(fig_et, width="stretch")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Avg ETo (mm/day)", f"{df_et['ETo (mm)'].mean():.2f}")
    with c2: st.metric("Total ETo (mm)", f"{df_et['ETo (mm)'].sum():.1f}")
    with c3: st.metric("Avg ETr (mm/day)", f"{df_et['ETr (mm)'].mean():.2f}")
    with c4: st.metric("Total ETr (mm)", f"{df_et['ETr (mm)'].sum():.1f}")
    
    with st.expander("🔧 Calculation Details & Assumptions"):
        st.markdown("""
        **Method:** ASCE Standardized Penman-Monteith (2005)
        
        **Reference Surfaces:**
        - **ETo (Short):** Grass reference, 0.12 m height (Cn=900, Cd=0.34)
        - **ETr (Tall):** Alfalfa reference, 0.50 m height (Cn=1600, Cd=0.38)
        
        **Required Inputs:**
        - Solar radiation (Rs) in MJ m⁻² d⁻¹
        - Wind speed at 2m height (m/s)
        - Min/Max air temperature (°C)
        - Min/Max relative humidity (%)
        - Site elevation and latitude
        
        **Assumptions:**
        - Albedo = 0.23
        - Soil heat flux (G) = 0 for daily calculations
        - Wind height = 2.0 m
        
        **Usage:** Multiply ET by crop coefficient (Kc) to estimate actual crop water use:
        ```
        ETc = Kc × ETo
        ```
        """)


with tab_rain:
    st.title("🌧️ Rainfall & Irrigation Planner")
    st.markdown('<p class="section-desc">Assess rainfall patterns, water balance, and irrigation recommendations for the filtered period.</p>', unsafe_allow_html=True)

    total_rain = float(dfx[COL_PCP].sum()) if COL_PCP in dfx.columns else 0.0
    avg_daily_rain = float(dfx[COL_PCP].mean()) if COL_PCP in dfx.columns else 0.0
    heaviest_row = None
    if COL_PCP in dfx.columns and not dfx.empty:
        heaviest_row = dfx.loc[dfx[COL_PCP].idxmax()]
    deficit_total = float(dfx["ETo (mm)"].sum() - total_rain) if "ETo (mm)" in dfx.columns else None

    rain_cols = st.columns(4)
    with rain_cols[0]:
        st.metric("Total Rain", f"{total_rain:.1f} mm")
    with rain_cols[1]:
        st.metric("Avg Daily", f"{avg_daily_rain:.1f} mm" if avg_daily_rain else "n/a")
    with rain_cols[2]:
        if heaviest_row is not None:
            st.metric(
                "Heaviest Day",
                heaviest_row[COL_DATE].date().isoformat(),
                delta=f"{heaviest_row[COL_PCP]:.1f} mm",
            )
        else:
            st.metric("Heaviest Day", "n/a")
    with rain_cols[3]:
        if deficit_total is not None:
            st.metric("Water Balance", f"{deficit_total:.1f} mm", delta="ETo - Rain")
        else:
            st.metric("Water Balance", "n/a")

    last_wet = None
    if COL_PCP in dfx.columns:
        wet_days = dfx[dfx[COL_PCP] >= 2][COL_DATE]
        if not wet_days.empty:
            last_wet = wet_days.max()
    if last_wet is not None:
        days_since = (dfx[COL_DATE].max() - last_wet).days
        st.caption(f"🌦️ Days since ≥2 mm rain: {days_since} days (last on {last_wet.date()})")

    st.markdown("### 📊 Daily Rainfall Hyetograph")
    if COL_PCP in dfx.columns and len(dfx):
        rain_fig = rainfall_bar_chart(dfx, COL_DATE, COL_PCP)
        st.plotly_chart(rain_fig, width="stretch")
        if allow_download:
            download_button_for_figure(rain_fig, filename="daily_rainfall.png")
    else:
        st.info("Rainfall data unavailable for this period.")

    st.markdown("---")
    st.markdown("### 💧 Cumulative Water Balance")
    st.markdown('<p class="help-text">Compare cumulative rainfall against ET demand to highlight deficits or surpluses.</p>', unsafe_allow_html=True)
    balance_df = compute_water_balance(dfx, COL_DATE, COL_PCP, "ETo (mm)")
    if not balance_df.empty:
        fig_balance = water_balance_figure(balance_df, COL_DATE)
        st.plotly_chart(fig_balance, width="stretch")
        final_balance = balance_df["Balance"].iloc[-1]
        if final_balance < 0:
            st.warning(f"⚠️ Water deficit of {abs(final_balance):.1f} mm detected. Consider irrigation.")
        else:
            st.success(f"✅ Water surplus of {final_balance:.1f} mm. Adequate moisture available.")
        if allow_download:
            download_button_for_figure(fig_balance, filename="water_balance.png")
    else:
        st.caption("Water balance chart unavailable (missing ETo/precip columns).")

    st.markdown("---")
    st.markdown("### 💦 Irrigation Decision Support")
    st.markdown('<div class="decision-card">', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        st.metric("Water Deficit", f"{decision['water_deficit_mm']:.1f} mm",
                  delta=f"ETo {decision['eto_mm']:.1f} − Rain {decision['effective_rain_mm']:.1f}")
    with cols[1]:
        if decision["irrigate"]:
            st.metric("Recommended", f"{decision['inches_needed']} in", f"${decision['cost_usd']:.2f}/acre")
        else:
            st.metric("Recommended", "No irrigation")
    with cols[2]:
        st.metric("Cost / inch", f"${irrigation_cost_per_inch:.0f}")
    st.markdown("<ul>" + "".join(f"<li>{msg}</li>" for msg in decision["reasoning"]) + "</ul>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if simulated_irrigation_in > 0:
        sim_deficit = decision["water_deficit_mm"] - (simulated_irrigation_in * 25.4)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Simulated Deficit", f"{sim_deficit:.1f} mm",
                      delta=f"{sim_deficit - decision['water_deficit_mm']:.1f} mm",
                      delta_color="inverse")
        with sc2:
            st.metric("Added Water", f"{simulated_irrigation_in:.2f} in")
        with sc3:
            st.metric("Added Cost", f"${simulated_irrigation_in * irrigation_cost_per_inch:.2f}/acre")

    st.markdown("---")
    st.markdown("### 🌦️ Rainfall Events & Weekly Totals")
    top_events = dfx[[COL_DATE, COL_PCP]].sort_values(COL_PCP, ascending=False).head(10)
    if not top_events.empty:
        st.dataframe(top_events.rename(columns={COL_DATE: "Date", COL_PCP: "Rain (mm)"}), hide_index=True, use_container_width=True)
    weekly = agg_df(dfx, COL_DATE, "W", cols_sum=[COL_PCP])
    if not weekly.empty:
        weekly = weekly.rename(columns={COL_DATE: "Week", COL_PCP: "Rain (mm)"})
        st.dataframe(weekly, hide_index=True, use_container_width=True)


# ===================================================
# TAB 3: CURRENT & FORECAST (ORIGINAL + SATELLITE MAP)
# ===================================================
with tab_owm:
    import pydeck as pdk
    
    st.title("🌦️ Current Weather & 5-Day Forecast")
    st.markdown('<p class="section-desc">Live weather data from OpenWeatherMap API with current conditions, hourly/daily forecasts, comfort index, and spray window detection. Includes what-if temperature/humidity adjustments.</p>', unsafe_allow_html=True)
    
    def temp_gradient_bg(temp, units):
        t = temp if units == "metric" else (temp - 32) * 5/9
        if t <= 0:    colors = ["#b3d9ff", "#e6f2ff"]
        elif t <= 15: colors = ["#c8f7c5", "#e9ffe9"]
        elif t <= 25: colors = ["#ffe6b3", "#fff3d6"]
        elif t <= 35: colors = ["#ffc7a0", "#ffe4d6"]
        else:         colors = ["#ffb3b3", "#ffe0e0"]
        return f"background: linear-gradient(135deg, {colors[0]}, {colors[1]});"
    
    def wind_svg(deg):
        return f"""
        <div style="transform: rotate({deg}deg);">
          <svg class="wind-arrow" viewBox="0 0 24 24">
            <path fill="#333" d="M12 2l4 8h-3v12h-2V10H8l4-8z"/>
          </svg>
        </div>"""
    
    def chip(text, cls="info"):
        st.markdown(f"<span class='chip {cls}'>{text}</span>", unsafe_allow_html=True)
    
    def find_spray_windows(dfh: pd.DataFrame, hours_ahead=48):
        """Detect ideal spray windows based on temp, humidity, wind, and rain"""
        now = pd.Timestamp.now()
        x = dfh[dfh["Datetime"].between(now, now + pd.Timedelta(hours=hours_ahead))].copy()
        if x.empty: return []
        ok = (
            (x["Temperature"].between(10, 28, inclusive="both")) &
            (x["Humidity"].between(40, 80, inclusive="both")) &
            (x["Wind"].le(6.0)) &
            (x["POP"].lt(20)) &
            (x["Rain_3h"].fillna(0.0).eq(0.0))
        )
        windows = []
        if ok.any():
            grp = (ok != ok.shift()).cumsum()
            for _, gdf in x[ok].groupby(grp[ok]):
                windows.append((gdf["Datetime"].min(), gdf["Datetime"].max() + pd.Timedelta(hours=3)))
        return windows
    
    def comfort_index_c(dff24):
        """Calculate apparent temperature comfort index"""
        if dff24.empty: return "—"
        T = dff24["Temperature"].clip(lower=-20, upper=50)
        RH = dff24["Humidity"].clip(lower=1, upper=100)
        AT = T + 0.33*RH/100*6 - 0.7
        m = AT.mean()
        if m < 5:   return "Cold"
        if m < 20:  return "Cool"
        if m < 27:  return "Comfortable"
        if m < 32:  return "Warm"
        return "Hot"
    
    # Location and settings
    colA, colB, colC, colD = st.columns([1.1, 1, 1, 1])
    with colA:
        st.session_state.setdefault("owm_favs", [])
        city = st.text_input("City", value=st.session_state.get("owm_last_city", "Colby"),
                            help="Enter city name for weather lookup")
        add_fav = st.button("⭐ Add to favorites")
        if add_fav and city and city not in st.session_state["owm_favs"]:
            st.session_state["owm_favs"].append(city)
        if st.session_state["owm_favs"]:
            pick = st.selectbox("Favorites", st.session_state["owm_favs"],
                                index=0 if city not in st.session_state["owm_favs"] else st.session_state["owm_favs"].index(city))
            if pick and pick != city:
                city = pick
        st.session_state["owm_last_city"] = city
    with colB:
        unit_option = st.selectbox("Units", ["Metric (°C, m/s)", "Imperial (°F, mph)"], index=0)
    with colC:
        simulate = st.checkbox("What-if (+/- Temp & RH)", value=False,
                              help="Adjust forecast values to test scenarios")
    with colD:
        compare = st.checkbox("Compare another city", value=False)
    
    if unit_option.startswith("Metric"):
        units, temp_unit, speed_unit = "metric", "°C", "m/s"
    else:
        units, temp_unit, speed_unit = "imperial", "°F", "mph"
    
    if simulate:
        sim_temp = st.slider("Δ Temperature", -10, 10, 0)
        sim_rh = st.slider("Δ Humidity (%)", -50, 50, 0)
    else:
        sim_temp = sim_rh = 0
    
    # Fetch current weather
    def fetch_current(cname, units):
        try:
            r = requests.get(BASE_URL_CURRENT, params={"appid": API_KEY, "q": cname, "units": units}, timeout=15)
            j = r.json()
            return j if j.get("cod") == 200 else {}
        except Exception:
            return {}
    
    def fetch_forecast(cname, units):
        try:
            r = requests.get(BASE_URL_FORECAST, params={"appid": API_KEY, "q": cname, "units": units}, timeout=15)
            j = r.json()
            return j if j.get("cod") == "200" else {}
        except Exception:
            return {}
    
    curr = fetch_current(city, units)
    fc = fetch_forecast(city, units)
    
    # Additional data (AQI, UV)
    aqi = None; uv = None; lat = lon = None
    if curr.get("coord"):
        lat = curr["coord"].get("lat")
        lon = curr["coord"].get("lon")
        try:
            aq = requests.get("http://api.openweathermap.org/data/2.5/air_pollution",
                              params={"lat": lat, "lon": lon, "appid": API_KEY}, timeout=10).json()
            if aq.get("list"):
                aqi = aq["list"][0]["main"]["aqi"]
        except Exception:
            pass
        try:
            oc = requests.get("https://api.openweathermap.org/data/2.5/onecall",
                              params={"lat": lat, "lon": lon, "exclude":"minutely,hourly,daily,alerts", "units": units, "appid": API_KEY},
                              timeout=10).json()
            uv = oc.get("current", {}).get("uvi")
        except Exception:
            pass
    
    # Process forecast
    dff_owm = pd.DataFrame()
    if fc and fc.get("list"):
        rows = []
        for it in fc["list"]:
            m = it.get("main", {})
            w = it.get("weather", [{}])[0]
            wind = it.get("wind", {}) or {}
            rows.append({
                "Datetime": pd.to_datetime(it.get("dt_txt")),
                "Temperature": (m.get("temp") or 0) + sim_temp,
                "Temp_Min": (m.get("temp_min") or 0) + sim_temp,
                "Temp_Max": (m.get("temp_max") or 0) + sim_temp,
                "Humidity": min(100, max(0, (m.get("humidity") or 0) + sim_rh)),
                "POP": (it.get("pop", 0.0) or 0.0) * 100.0,
                "Rain_3h": it.get("rain", {}).get("3h", 0.0) or 0.0,
                "Snow_3h": it.get("snow", {}).get("3h", 0.0) or 0.0,
                "Clouds": it.get("clouds", {}).get("all", 0),
                "Wind": wind.get("speed", 0.0) or 0.0,
                "WindDeg": wind.get("deg", 0) or 0,
                "Icon": w.get("icon", ""),
                "Desc": w.get("description", "").title(),
                "Date": pd.to_datetime(it.get("dt_txt")).date(),
                "Day": pd.to_datetime(it.get("dt_txt")).strftime("%a"),
            })
        dff_owm = pd.DataFrame(rows).sort_values("Datetime").reset_index(drop=True)
    
    # Current weather display
    if curr:
        main = curr.get("main", {})
        weather = curr.get("weather", [{}])[0]
        wind = curr.get("wind", {})
        temp = (main.get("temp") or 0) + sim_temp
        feels = (main.get("feels_like") or 0) + sim_temp
        rh = min(100, max(0, (main.get("humidity") or 0) + sim_rh))
        stamp = curr.get("dt")
        
        hero_bg = temp_gradient_bg(temp, units)
        st.markdown(f"<div class='hero' style='{hero_bg}'>", unsafe_allow_html=True)
        hc1, hc2, hc3, hc4 = st.columns([1.6, 1, 1, 1], gap="small")
        with hc1:
            st.subheader(f"Now in {city}")
            c1, c2 = st.columns([1, 3])
            with c1:
                icon = weather.get("icon", "")
                st.image(f"http://openweathermap.org/img/wn/{icon}@2x.png", width=84)
            with c2:
                st.metric("Temp", f"{round(temp)} {temp_unit}")
                st.caption(weather.get("description", "").title())
                st.write(f"Feels like **{round(feels)} {temp_unit}** | RH **{int(rh)}%**")
                if not dff_owm.empty:
                    soon = dff_owm[dff_owm["Datetime"] <= (pd.Timestamp.now() + pd.Timedelta(hours=24))]
                    if (soon["Temp_Min"] <= (0 if units=="metric" else 32)).any(): chip("❄️ Frost risk", "danger")
                    if (soon["Temp_Max"] >= (35 if units=="metric" else 95)).any(): chip("🥵 Heat stress", "warn")
                    if (soon["Wind"] >= 10).any(): chip("💨 Windy", "info")
                    if (soon["POP"] >= 60).any() or (soon["Rain_3h"] >= 2).any(): chip("🌧️ Rain likely", "rain")
        with hc2:
            st.subheader("Hi/Lo")
            st.metric("High", f"{round(main.get('temp_max', 0)+sim_temp)} {temp_unit}")
            st.metric("Low",  f"{round(main.get('temp_min', 0)+sim_temp)} {temp_unit}")
        with hc3:
            st.subheader("Wind")
            st.markdown(wind_svg(wind.get("deg", 0) or 0), unsafe_allow_html=True)
            st.metric("Speed", f"{wind.get('speed', 0)} {speed_unit}")
        with hc4:
            st.subheader("More")
            if aqi is not None:
                label = ["Good","Fair","Moderate","Poor","Very Poor"][int(aqi)-1]
                st.metric("AQI", f"{aqi} ({label})")
            else:
                st.caption("AQI n/a")
            st.metric("UV", f"{uv:.1f}" if uv is not None else "n/a")
            if stamp:
                st.caption("Updated: " + pd.to_datetime(stamp, unit="s").strftime("%Y-%m-%d %H:%M"))
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Forecast tabs
    if not dff_owm.empty:
        ft_hourly, ft_daily, ft_charts = st.tabs(["🔮 Hourly", "📅 Daily", "📈 Charts"])
        
        with ft_hourly:
            st.markdown("#### Next 24 hours")
            st.markdown('<p class="help-text">Temperature and precipitation probability for the next 24 hours</p>', unsafe_allow_html=True)
            now = pd.Timestamp.now()
            d24 = dff_owm[dff_owm["Datetime"].between(now, now + pd.Timedelta(hours=24))].copy()
            if d24.empty: d24 = dff_owm.head(8).copy()
            fig24 = make_subplots(specs=[[{"secondary_y": True}]])
            fig24.add_trace(go.Scatter(x=d24["Datetime"], y=d24["Temperature"], mode="lines+markers",
                                       name=f"Temp ({temp_unit})"), secondary_y=False)
            fig24.add_bar(x=d24["Datetime"], y=d24["POP"], name="POP (%)", opacity=0.45, secondary_y=True)
            fig24.update_yaxes(title_text=f"Temp ({temp_unit})", secondary_y=False)
            fig24.update_yaxes(title_text="POP (%)", range=[0, 100], secondary_y=True)
            fig24.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                                template="plotly_white", legend=dict(orientation="h"))
            st.plotly_chart(fig24, width="stretch")
            
            cA, cB = st.columns([1,1])
            with cA:
                st.subheader("Comfort Index")
                st.metric("Next 24h", comfort_index_c(d24))
                st.caption("Based on apparent temperature")
            with cB:
                st.subheader("🚜 Spray Windows (48h)")
                wins = find_spray_windows(dff_owm, hours_ahead=48)
                if wins:
                    for i, (a, b) in enumerate(wins[:3], start=1):
                        st.write(f"**Window {i}:** {a.strftime('%a %H:%M')} → {b.strftime('%a %H:%M')}")
                    st.caption("Ideal: 10-28°C, 40-80% RH, <6 m/s wind, <20% POP")
                else:
                    st.caption("No ideal window in the next 48h.")
            
            with st.expander("Hourly table (next 48h)"):
                d48 = dff_owm[dff_owm["Datetime"].between(now, now + pd.Timedelta(hours=48))].copy()
                st.dataframe(d48, use_container_width=True, height=320)
        
        with ft_daily:
            st.markdown("#### Next 5 days — daily summary")
            st.markdown('<p class="help-text">Daily min/max temperatures, total rainfall, and average precipitation probability</p>', unsafe_allow_html=True)
            daily = (dff_owm.groupby(["Date","Day"], as_index=False)
                        .agg({"Temp_Min":"min","Temp_Max":"max","Rain_3h":"sum","POP":"mean",
                              "Icon":lambda x: x.value_counts().index[0]}))
            cols = st.columns(len(daily))
            for i, (_, row) in enumerate(daily.iterrows()):
                with cols[i]:
                    st.markdown(f"<div class='daycard'>", unsafe_allow_html=True)
                    st.write(f"**{row['Day']}** {row['Date'].strftime('%m/%d')}")
                    st.image(f"http://openweathermap.org/img/wn/{row['Icon']}@2x.png", width=48)
                    st.write(f"{round(row['Temp_Min'])}/{round(row['Temp_Max'])}{temp_unit}")
                    st.caption(f"Rain: {row['Rain_3h']:.1f} mm • POP: {row['POP']:.0f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            pick = st.radio("Drill-down day", options=list(daily["Date"].astype(str)), index=0, horizontal=True)
            dd = dff_owm[dff_owm["Date"] == pd.to_datetime(pick).date()]
            if not dd.empty:
                fig_dd = forecast_daily_temp_band(dd, temp_unit)
                st.plotly_chart(fig_dd, width="stretch")
        
        with ft_charts:
            st.markdown("#### Wind Rose (next 5 days)")
            st.markdown('<p class="help-text">Average wind speed by direction across the forecast period</p>', unsafe_allow_html=True)
            if "WindDeg" in dff_owm.columns and dff_owm["WindDeg"].notna().any():
                dirs = np.array(["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"])
                deg = pd.to_numeric(dff_owm["WindDeg"], errors="coerce").mod(360).fillna(0)
                idx = np.floor(((deg + 11.25) % 360) / 22.5).astype(int) % 16
                cats = pd.Categorical.from_codes(idx, dirs, ordered=True)
                rose = (dff_owm.assign(Direction=cats).groupby("Direction")["Wind"].mean().reindex(dirs).fillna(0))
                fig_rose = go.Figure(go.Barpolar(r=rose.values, theta=rose.index.tolist(),
                                                marker_line_color="white", marker_line_width=1, opacity=0.85, name="Mean wind"))
                fig_rose.update_layout(height=340, template="plotly_white",
                                       margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig_rose, width="stretch")
            else:
                st.info("No wind direction data.")
            
            st.markdown("#### Combined Trends")
            st.markdown('<p class="help-text">Temperature, humidity, and rainfall over the forecast period</p>', unsafe_allow_html=True)
            fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
            fig_combo.add_trace(go.Scatter(x=dff_owm["Datetime"], y=dff_owm["Temperature"], name=f"Temp ({temp_unit})", mode="lines"),
                                secondary_y=False)
            fig_combo.add_trace(go.Scatter(x=dff_owm["Datetime"], y=dff_owm["Humidity"], name="Humidity (%)", mode="lines"),
                                secondary_y=False)
            fig_combo.add_trace(go.Bar(x=dff_owm["Datetime"], y=dff_owm["Rain_3h"], name="Rain (3h) mm", opacity=0.4),
                                secondary_y=True)
            fig_combo.update_yaxes(title_text=f"Temp/Humidity", secondary_y=False)
            fig_combo.update_yaxes(title_text="Rain (mm)", secondary_y=True)
            fig_combo.update_layout(height=360, template="plotly_white",
                                    margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig_combo, width="stretch")
    
    # NEW! Satellite map
    if curr and curr.get("coord"):
        st.markdown("---")
        st.markdown("### 📍 Location Map")
        st.markdown('<p class="help-text">Satellite view showing the weather station location</p>', unsafe_allow_html=True)
        lat = curr["coord"]["lat"]; lon = curr["coord"]["lon"]
        point_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({"lat":[lat],"lon":[lon]}),
            get_position="[lon, lat]", get_radius=500,
            get_fill_color=[200, 30, 0, 160], pickable=False
        )
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=9)
        mapbox_key = os.environ.get("MAPBOX_API_KEY") or getattr(pdk.settings, "mapbox_key", None)
        if mapbox_key:
            deck = pdk.Deck(layers=[point_layer], initial_view_state=view_state,
                            map_style="mapbox://styles/mapbox/satellite-v9")
            st.pydeck_chart(deck); st.caption("Basemap: Mapbox Satellite.")
        else:
            satellite_layer = pdk.Layer(
                "TileLayer",
                data="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                min_zoom=0, max_zoom=19, tile_size=256
            )
            deck = pdk.Deck(layers=[satellite_layer, point_layer], initial_view_state=view_state, map_style=None)
            st.pydeck_chart(deck); st.caption("Basemap: Esri World Imagery.")
    
    # Export forecast
    if not dff_owm.empty:
        with st.expander("🔎 Export Forecast Data"):
            st.dataframe(dff_owm, use_container_width=True, height=320)
            csv = dff_owm.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download forecast CSV", csv, file_name=f"{city}_forecast.csv", mime="text/csv")

# ===================================================
# TAB 4: AI INSIGHTS (COMPLETE IMPLEMENTATION)
# ===================================================
with tab_ai:
    st.header("🤖 AI Insights & Decision Support")
    st.markdown('<p class="section-desc">Dual-mode AI assistant: data-grounded analysis using your weather records and forecast, or general agricultural Q&A. Includes JSON export for external analysis.</p>', unsafe_allow_html=True)
    
    # OpenAI client setup
    try:
        from openai import OpenAI
        _openai_ok = True
    except Exception:
        _openai_ok = False
    
    if not _openai_ok:
        st.info("📦 Install the OpenAI SDK: `pip install openai>=1.40`")
        st.stop()
    elif not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        st.info("🔑 Configure OPENAI_API_KEY to enable AI features")
        st.stop()
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
    
    forecast_context = dff if not dff.empty else dff_owm
    weather_tool_context = {
        "date_col": COL_DATE,
        "precip_col": COL_PCP,
        "eto_col": "ETo (mm)",
        "gdd_col": COL_GDD_FP_CUM,
        "planting_date": planting_date,
        "growth_stage": growth_stage,
        "cost_per_inch": irrigation_cost_per_inch,
    }
    tool_definitions = weather_tools.WEATHER_TOOL_DEFINITIONS

    def execute_tool_call(tool_name, tool_args):
        try:
            return weather_tools.execute_weather_tool(tool_name, tool_args, dfx, weather_tool_context, forecast_context)
        except Exception as exc:
            return {"error": str(exc)}

    def _call_openai(messages, max_tokens=700, temperature=0.2):
        """Call OpenAI with fallback models"""
        for model in OPENAI_MODELS_TRY:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"All models failed. Last error: {last_err}")

    def _sanitize_history(msgs):
        safe = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content is None:
                continue
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, default=json_default)
                except Exception:
                    content = str(content)
            if content.strip() == "":
                continue
            safe.append({"role": role, "content": content})
        return safe


    def build_weather_ai_payload() -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "view": {
                "station": station_label,
                "date_range": {"start": str(start), "end": str(end)},
                "records": int(len(dfx)),
                "growth_stage": growth_stage,
                "gdd_target": gdd_target if gdd_target > 0 else None,
                "selected_variables": plot_vars,
            }
        }
        weather_summary = summarize_weather_period(dfx, COL_DATE, COL_PCP, COL_TMAX, COL_TMIN, "ETo (mm)")
        if weather_summary:
            payload["weather_summary"] = weather_summary

        if COL_GDD_FP_CUM in dfa.columns and not dfa.empty:
            payload["gdd_status"] = weather_tools.get_gdd_status(
                dfa,
                date_col=COL_DATE,
                gdd_col=COL_GDD_FP_CUM,
                planting_date=planting_date,
                target=gdd_target if gdd_target > 0 else None,
            )

        if "ETo (mm)" in dfx.columns:
            payload["eto_summary"] = weather_tools.get_eto_for_period(
                dfx,
                COL_DATE,
                "ETo (mm)",
                start=str(start),
                end=str(end),
            )

        if COL_PCP in dfx.columns:
            rain_events = (
                dfx[[COL_DATE, COL_PCP]]
                .sort_values(COL_PCP, ascending=False)
                .head(5)
                .assign(date=lambda d: d[COL_DATE].dt.strftime("%Y-%m-%d"))
            )
            payload["rainfall_summary"] = {
                "total_mm": float(dfx[COL_PCP].sum()),
                "avg_daily_mm": float(dfx[COL_PCP].mean()),
                "heaviest_events": [
                    {"date": row["date"], "rain_mm": float(row[COL_PCP])}
                    for _, row in rain_events.iterrows()
                ],
            }

        irrigation_summary = dict(decision)
        irrigation_summary["recent_window_days"] = 7
        irrigation_summary["cost_per_inch"] = irrigation_cost_per_inch
        payload["irrigation_recommendation"] = irrigation_summary

        balance_df = compute_water_balance(dfx, COL_DATE, COL_PCP, "ETo (mm)")
        if not balance_df.empty:
            payload["water_balance"] = {
                "latest_date": str(balance_df[COL_DATE].iloc[-1].date()),
                "balance_mm": float(balance_df["Balance"].iloc[-1]),
            }

        payload["forecast"] = {
            "city": city_used,
            "summary": weather_tools.get_forecast_summary(forecast_context),
        }
        payload["alerts"] = weather_tools.get_smart_alerts(forecast_context).get("alerts", [])
        payload["simulation"] = {
            "planned_irrigation_in": simulated_irrigation_in,
            "weather_adjustments": {"temp_shift": sim_temp, "humidity_shift": sim_rh},
        }
        payload["tools_available"] = [tool["function"]["name"] for tool in weather_tools.WEATHER_TOOL_DEFINITIONS]
        return payload

    json_payload = build_weather_ai_payload()
    
    # Display JSON
    st.markdown("### 📄 Data Snapshot")
    st.markdown('<p class="help-text">Structured summary covering GDD, ET, rainfall, irrigation guidance, and forecast context.</p>', unsafe_allow_html=True)
    
    with st.expander("View JSON Snapshot"):
        st.json(json_payload)
    
    json_str = json.dumps(json_payload, indent=2, default=json_default)
    st.download_button("⬇️ Download JSON", json_str.encode("utf-8"),
                      file_name=f"weather_snapshot_{start}_{end}.json",
                      mime="application/json")
    
    # One-click analysis
    st.markdown("---")
    st.markdown("### 🧠 One-Click Analysis")
    st.markdown('<p class="help-text">Get instant AI-powered insights and recommendations based on your data</p>', unsafe_allow_html=True)
    
    user_goal = st.text_area("Optional: Customize analysis focus",
                             placeholder="e.g., Focus on irrigation timing and cost optimization",
                             help="Leave blank for general analysis")
    
    if st.button("🚀 Generate Analysis"):
        with st.spinner("Analyzing weather data..."):
            try:
                messages = [
                    {"role": "system", "content": 
                     "You are an agricultural weather analyst. Provide concise, actionable insights. "
                     "Format: **Conclusions** (5-7 bullets with numbers), **Recommendations** (prioritized actions), "
                     "**Questions** (2-3 follow-ups). Each bullet ≤20 words."},
                    {"role": "user", "content": 
                     f"{user_goal or 'Analyze weather, irrigation needs, and forecast hazards.'}\n\nData:\n{json_str}"}
                ]
                analysis = _call_openai(messages, max_tokens=900, temperature=0.1)
                st.markdown(analysis)
            except Exception as e:
                st.error(f"Analysis error: {e}")
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Chat Assistant")
    st.markdown('<p class="help-text">Ask questions about your data or general agricultural topics. Auto-detects data vs. general questions.</p>', unsafe_allow_html=True)
    
    mode = st.radio("Mode:", ["Auto", "Data-only", "General-only"], horizontal=True,
                    help="Auto: detects question type | Data-only: uses snapshot | General-only: no data context")
    
    if "ai_chat" not in st.session_state:
        st.session_state.ai_chat = []
    
    for msg in st.session_state.ai_chat:
        st.chat_message(msg["role"]).write(msg["content"])
    
    def is_data_question(text):
        keywords = ["precip", "rain", "eto", "temperature", "gdd", "irrigation", "forecast", "deficit", "weather"]
        return any(k in text.lower() for k in keywords)
    
    if prompt := st.chat_input("Ask anything about weather or agriculture..."):
        st.session_state.ai_chat.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        try:
            use_data = (mode == "Data-only") or (mode == "Auto" and is_data_question(prompt))
            history = _sanitize_history(st.session_state.ai_chat[:-1])
            messages = []
            if use_data:
                system_prompt = (
                    "You are an agricultural weather analyst. Use the provided data snapshot and tools when needed. "
                    "Cite key numbers with units."
                )
                messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": f"Weather data snapshot:\n{json_str}"})
            else:
                messages.append({"role": "system", "content": "You are a helpful agricultural advisor."})
            messages.extend(history)
            messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Generating answer..."):
                if use_data:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        tools=tool_definitions,
                        tool_choice="auto",
                        max_tokens=700,
                        temperature=0.2,
                    )
                    iteration = 0
                    while response.choices[0].finish_reason == "tool_calls" and iteration < 5:
                        iteration += 1
                        assistant_msg = response.choices[0].message
                        messages.append(
                            {
                                "role": "assistant",
                                "content": assistant_msg.content or "",
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in (assistant_msg.tool_calls or [])
                                ],
                            }
                        )
                        for tool_call in assistant_msg.tool_calls or []:
                            try:
                                args = json.loads(tool_call.function.arguments)
                            except Exception:
                                args = {}
                            tool_response = execute_tool_call(tool_call.function.name, args)
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": json.dumps(tool_response, default=json_default),
                                }
                            )
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            tools=tool_definitions,
                            tool_choice="auto",
                            max_tokens=700,
                            temperature=0.2,
                        )
                else:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=600,
                        temperature=0.4,
                    )
            answer = response.choices[0].message.content or "(no response)"
            st.session_state.ai_chat.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
        except Exception as e:
            st.error(f"Chat error: {e}")

# Footer
st.markdown("---")
st.markdown(f'<p class="help-text" style="text-align:center">KSUTAPS Complete Weather Dashboard | Generated: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Data: {fc_source.upper()}</p>', unsafe_allow_html=True)

# streamlit run weather_dashboard_complete.py
