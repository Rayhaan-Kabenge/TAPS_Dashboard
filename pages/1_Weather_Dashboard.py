# weather_dashboard_complete.py
# ---------------------------------------------------
# KSUTAPS • Complete Weather Dashboard + AI (Original + Enhancements)
# Preserves ALL original features + adds decision support, alerts, caching
# ---------------------------------------------------
import os, io, json, math, base64, textwrap
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests


def _json_default(o):
    import numpy as np
    import pandas as pd
    import datetime as dt

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    # Last resort: string
    return str(o)


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
FILE_PATH = "/Users/rayhaankabenge/Desktop/KSUTAPS/2025/ET_analysis/Data/ET_analysis_Data1.xlsx"
SHEET_NAME = "Climate_Data"
STATION_ID = "Colby, KS"
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

COL_GDD_STD_DAILY = "GDD (°C·d)"
COL_GDD_FP_DAILY = "FromPlanting_GDD (°C·d)"
COL_GDD_FP_CUM = "ΣGDD (from planting)"

# =========================
# NEW! CACHING LAYER
# =========================
if "weather_cache" not in st.session_state:
    st.session_state.weather_cache = {}

def cached_request(url, params, timeout=15, cache_minutes=30):
    """Cache API requests to improve speed and reduce quota usage"""
    cache_key = f"{url}_{json.dumps(params, sort_keys=True)}"
    now = dt.datetime.now()
    
    if cache_key in st.session_state.weather_cache:
        cached_data, timestamp = st.session_state.weather_cache[cache_key]
        if (now - timestamp).total_seconds() < cache_minutes * 60:
            return cached_data, "cached"
    
    try:
        r = requests.get(url, params=params, timeout=timeout)
        data = r.json()
        st.session_state.weather_cache[cache_key] = (data, now)
        return data, "live"
    except Exception as e:
        if cache_key in st.session_state.weather_cache:
            return st.session_state.weather_cache[cache_key][0], "cached_fallback"
        return {}, "error"

# =========================
# ET CALCULATIONS (ASCE Daily)
# =========================
def ascedaily(rfcrp, z, lat, doy, israd, tmax, tmin,
              vapr=float('NaN'), tdew=float('NaN'),
              rhmax=float('NaN'), rhmin=float('NaN'),
              wndsp=float('NaN'), wndht=2.0):
    """ASCE Standardized Penman-Monteith ET calculation"""
    tavg = (tmax + tmin) / 2.0
    patm = 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26
    psycon = 0.000665 * patm
    Udelta = 2503.0 * math.exp(17.27 * tavg / (tavg + 237.3))
    Udelta = Udelta / ((tavg + 237.3) ** 2.0)
    emax = 0.6108 * math.exp((17.27 * tmax) / (tmax + 237.3))
    emin = 0.6108 * math.exp((17.27 * tmin) / (tmin + 237.3))
    es = (emax + emin) / 2.0
    if not math.isnan(vapr):
        ea = vapr
    elif not math.isnan(tdew):
        ea = 0.6108 * math.exp((17.27 * tdew) / (tdew + 237.3))
    elif not math.isnan(rhmax) and not math.isnan(rhmin):
        ea = (emin * rhmax / 100. + emax * rhmin / 100.) / 2.0
    elif not math.isnan(rhmax):
        ea = emin * rhmax / 100.
    elif not math.isnan(rhmin):
        ea = emax * rhmin / 100.
    else:
        tdew = tmin - 2.0
        ea = 0.6108 * math.exp((17.27 * tdew) / (tdew + 237.3))
    albedo = 0.23
    rns = (1.0 - albedo) * israd
    latrad = lat * math.pi / 180.0
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi / 365.0 * doy)
    ldelta = 0.409 * math.sin(2.0 * math.pi / 365.0 * doy - 1.39)
    ws = math.acos(-1.0 * math.tan(latrad) * math.tan(ldelta))
    ra1 = ws * math.sin(latrad) * math.sin(ldelta)
    ra2 = math.cos(latrad) * math.cos(ldelta) * math.sin(ws)
    ra = 24.0 / math.pi * 4.92 * dr * (ra1 + ra2)
    rso = (0.75 + 2e-5 * z) * ra
    ratio = sorted([0.3, (israd / rso if rso != 0 else 0), 1.0])[1]
    fcd = sorted([0.05, 1.35 * ratio - 0.35, 1.0])[1]
    tk4 = ((tmax + 273.16) ** 4.0 + (tmin + 273.16) ** 4.0) / 2.0
    rnl = 4.901e-9 * fcd * (0.34 - 0.14 * math.sqrt(ea)) * tk4
    rn = rns - rnl
    g = 0.0
    if math.isnan(wndsp): wndsp = 2.0
    u2 = wndsp * (4.87 / math.log(67.8 * wndht - 5.42))
    if rfcrp == 'S':
        Cn, Cd = 900.0, 0.34
    else:
        Cn, Cd = 1600.0, 0.38
    etsz = 0.408 * Udelta * (rn - g) + psycon * (Cn / (tavg + 273.0)) * u2 * (es - ea)
    etsz = etsz / (Udelta + psycon * (1.0 + Cd * u2))
    return etsz

def calculate_effective_rainfall(precip_mm, method="usda"):
    """Calculate effective rainfall using USDA method"""
    if method == "usda":
        if precip_mm < 25:
            return precip_mm * 0.95
        else:
            return 25 * 0.95 + (precip_mm - 25) * 0.75
    return precip_mm * 0.8

# =========================
# GDD HELPERS
# =========================
def gdd_excel_style(tmin, tmax, tbase, tmax_cap):
    """Calculate Growing Degree Days using standard method"""
    tavg = (tmin + tmax) / 2.0
    if (tmax <= tmax_cap) and (tmin >= tbase):
        return tavg - tbase
    elif (tmax <= tmax_cap) and (tmax >= tbase) and (tmin < tbase):
        return ((tmax + tbase) / 2.0) - tbase
    elif (tmax > tmax_cap) and (tmin >= tbase):
        return ((tmax_cap + tmin) / 2.0) - tbase
    elif (tmax > tmax_cap) and (tmin < tbase):
        return ((tmax_cap + tbase) / 2.0) - tbase
    else:
        return 0.0

def compute_gdd_columns(df: pd.DataFrame, tbase: float, tcap: float,
                        planting_date: pd.Timestamp, harvest_date: pd.Timestamp | None) -> pd.DataFrame:
    """Compute GDD columns for the entire dataset"""
    out = df.copy()
    std_daily, fp_daily = [], []
    for _, r in out.iterrows():
        g = gdd_excel_style(float(r[COL_TMIN]), float(r[COL_TMAX]), tbase, tcap)
        std_daily.append(round(g, 3))
        if r[COL_DATE] >= planting_date and (harvest_date is None or r[COL_DATE] <= harvest_date):
            fp_daily.append(round(g, 3))
        else:
            fp_daily.append(0.0)
    out[COL_GDD_STD_DAILY] = std_daily
    out[COL_GDD_FP_DAILY] = fp_daily
    out[COL_GDD_FP_CUM] = out[COL_GDD_FP_DAILY].cumsum()
    return out

# =========================
# NEW! IRRIGATION DECISION ENGINE
# =========================
def calculate_irrigation_decision(recent_precip_mm, eto_mm, soil_moisture_percent=None,
                                  growth_stage="vegetative", cost_per_inch=25.0):
    """Decision engine for irrigation recommendations based on water balance"""
    effective_rain = calculate_effective_rainfall(recent_precip_mm)
    water_deficit = eto_mm - effective_rain
    
    stage_thresholds = {
        "vegetative": 15,
        "reproductive": 10,
        "grain_fill": 12,
        "maturity": 20
    }
    threshold = stage_thresholds.get(growth_stage, 15)
    
    irrigate = bool(water_deficit > threshold)

    inches_needed = max(0, water_deficit / 25.4) if irrigate else 0
    cost_estimate = inches_needed * cost_per_inch
    
    reasons = []
    if water_deficit > threshold:
        reasons.append(f"Water deficit ({water_deficit:.1f}mm) exceeds threshold ({threshold}mm)")
    if effective_rain < eto_mm * 0.5:
        reasons.append(f"Effective rainfall ({effective_rain:.1f}mm) insufficient")
    if growth_stage in ["reproductive", "grain_fill"]:
        reasons.append(f"Critical {growth_stage} stage")
    
    return {
        "irrigate": irrigate,
        "inches_needed": round(inches_needed, 2),
        "cost_usd": round(cost_estimate, 2),
        "water_deficit_mm": round(water_deficit, 1),
        "effective_rain_mm": round(effective_rain, 1),
        "eto_mm": round(eto_mm, 1),
        "reasoning": reasons or ["Sufficient moisture from recent rainfall"]
    }

# =========================
# NEW! SMART ALERT SYSTEM
# =========================
def generate_smart_alerts(forecast_df: pd.DataFrame, current_temp: float = None) -> list:
    """Generate actionable alerts from forecast data"""
    alerts = []
    if forecast_df.empty:
        return alerts
    
    now = pd.Timestamp.now()
    next_24h = forecast_df[forecast_df["Datetime"].between(now, now + pd.Timedelta(hours=24))]
    next_48h = forecast_df[forecast_df["Datetime"].between(now, now + pd.Timedelta(hours=48))]
    
    if next_24h.empty:
        return alerts
    
    # Frost risk
    if (next_24h["Temp_Min"] <= 0).any():
        frost_time = next_24h[next_24h["Temp_Min"] <= 0]["Datetime"].min()
        alerts.append({
            "type": "danger",
            "icon": "❄️",
            "title": "FROST RISK",
            "message": f"Freezing temps expected at {frost_time.strftime('%I%p %a')}",
            "action": "Delay irrigation. Protect sensitive crops."
        })
    
    # Heat stress
    if (next_24h["Temp_Max"] >= 35).any():
        alerts.append({
            "type": "warn",
            "icon": "🔥",
            "title": "HEAT STRESS",
            "message": f"High temps up to {next_24h['Temp_Max'].max():.0f}°C expected",
            "action": "Increase irrigation frequency. Monitor crop stress."
        })
    
    # High wind
    if (next_48h["Wind"] >= 6.0).any():
        alerts.append({
            "type": "warn",
            "icon": "💨",
            "title": "HIGH WIND",
            "message": f"Wind speeds up to {next_48h['Wind'].max():.1f} m/s",
            "action": "Avoid spraying. Risk of drift/evaporation."
        })
    
    # Heavy rain
    if (next_24h["Rain_3h"] >= 10).any() or (next_24h["POP"] >= 70).any():
        total_rain = next_24h["Rain_3h"].sum()
        alerts.append({
            "type": "info",
            "icon": "🌧️",
            "title": "HEAVY RAIN EXPECTED",
            "message": f"Up to {total_rain:.1f}mm in next 24h",
            "action": "Delay irrigation and field operations."
        })
    
    # Ideal spray window
    ideal_conditions = (
        (next_48h["Temperature"].between(10, 28)) &
        (next_48h["Humidity"].between(40, 80)) &
        (next_48h["Wind"] < 6.0) &
        (next_48h["POP"] < 20)
    )
    if ideal_conditions.any():
        ideal_df = next_48h[ideal_conditions]
        window_start = ideal_df["Datetime"].min()
        window_end = ideal_df["Datetime"].max()
        alerts.append({
            "type": "success",
            "icon": "✅",
            "title": "SPRAY WINDOW",
            "message": f"Ideal conditions from {window_start.strftime('%I%p')} to {window_end.strftime('%I%p %a')}",
            "action": "Schedule spray applications during this window."
        })
    
    return alerts

# =========================
# DATA LOADING
# =========================
@st.cache_data(show_spinner=False)
def load_climate(path: str, sheet: str) -> pd.DataFrame:
    """Load and validate climate data from Excel"""
    df = pd.read_excel(path, sheet_name=sheet)
    df.rename(columns={c: c.strip().replace("\u00A0", " ") for c in df.columns}, inplace=True)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in `{sheet}`: {missing}")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE])
    for col in [COL_TMIN, COL_TMAX, COL_RHMIN, COL_RHMAX, COL_U2, COL_RS, COL_PCP]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[COL_TMIN, COL_TMAX, COL_RHMIN, COL_RHMAX, COL_U2, COL_RS, COL_PCP])
    df["DOY"] = df[COL_DATE].dt.dayofyear
    return df.sort_values(COL_DATE).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def compute_et(df: pd.DataFrame, z_m: float, latitude_rad: float) -> pd.DataFrame:
    """Calculate reference ET (ETo and ETr) for all records"""
    lat_deg = float(latitude_rad) * 180.0 / math.pi
    eto_vals, etr_vals = [], []
    for _, r in df.iterrows():
        eto = ascedaily("S", z_m, lat_deg, int(r["DOY"]),
                        r[COL_RS], r[COL_TMAX], r[COL_TMIN],
                        rhmax=r[COL_RHMAX], rhmin=r[COL_RHMIN],
                        wndsp=r[COL_U2], wndht=2.0)
        etr = ascedaily("T", z_m, lat_deg, int(r["DOY"]),
                        r[COL_RS], r[COL_TMAX], r[COL_TMIN],
                        rhmax=r[COL_RHMAX], rhmin=r[COL_RHMIN],
                        wndsp=r[COL_U2], wndht=2.0)
        eto_vals.append(eto)
        etr_vals.append(etr)
    out = df.copy()
    out["ETo (mm)"] = pd.to_numeric(eto_vals, errors="coerce")
    out["ETr (mm)"] = pd.to_numeric(etr_vals, errors="coerce")
    return out

def agg_df(df: pd.DataFrame, freq: str, cols_sum=None, cols_mean=None):
    """Aggregate data by time period (Weekly/Monthly)"""
    cols_sum = cols_sum or []
    cols_mean = cols_mean or []
    g = df.set_index(COL_DATE).groupby(pd.Grouper(freq=freq))
    parts = []
    if cols_sum: parts.append(g[cols_sum].sum())
    if cols_mean: parts.append(g[cols_mean].mean())
    if parts:
        agg = pd.concat(parts, axis=1).reset_index()
        return agg
    return df.copy()

def _bytes_from_figure(fig):
    """Convert plotly figure to PNG bytes"""
    buf = io.BytesIO()
    try:
        fig.write_image(buf, format="png", scale=2)
        return buf.getvalue()
    except Exception:
        return None

def download_button_for_figure(fig, filename="chart.png", label="⬇️ Download PNG"):
    """Create download button for plotly figure"""
    b = _bytes_from_figure(fig)
    if b is None:
        st.caption("PNG export needs `kaleido` (`pip install -U kaleido`).")
        return
    st.download_button(label, b, file_name=filename, mime="image/png")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.title("⚙️ Weather Controls")
st.sidebar.markdown(f"**Station:** {STATION_ID}")
st.sidebar.markdown('<p class="help-text">Configure all weather analysis parameters below</p>', unsafe_allow_html=True)

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
    df = load_climate(FILE_PATH, SHEET_NAME)
except Exception as e:
    st.error(f"❌ Could not read data: {e}")
    st.stop()

df_et = compute_et(df, elevation_m, latitude_rad)
pdate = pd.Timestamp(planting_date)
hdate = pd.Timestamp(harvest_date) if harvest_date else None
df_full = compute_gdd_columns(df_et, gdd_base, gdd_cap, pdate, hdate)

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
    dfa = agg_df(dfx, "W",
                 cols_sum=[COL_PCP, "ETo (mm)", "ETr (mm)", COL_GDD_STD_DAILY, COL_GDD_FP_DAILY],
                 cols_mean=[COL_TMAX, COL_TMIN, COL_U2, COL_RHMAX, COL_RHMIN, COL_RS])
elif agg_choice == "Monthly":
    dfa = agg_df(dfx, "MS",
                 cols_sum=[COL_PCP, "ETo (mm)", "ETr (mm)", COL_GDD_STD_DAILY, COL_GDD_FP_DAILY],
                 cols_mean=[COL_TMAX, COL_TMIN, COL_U2, COL_RHMAX, COL_RHMIN, COL_RS])
else:
    dfa = dfx.copy()

if COL_GDD_FP_DAILY in dfa.columns:
    dfa[COL_GDD_FP_CUM] = dfa[COL_GDD_FP_DAILY].cumsum()

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
tab_data, tab_et, tab_owm, tab_ai = st.tabs([
    "📈 Data & Charts", "💧 Reference ET", "🌦️ Current & Forecast", "🤖 AI Insights"
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
    
    # What-if simulation
    #if simulated_irrigation_in > 0:
   #     st.markdown("---")
    #    st.markdown(f"**💡 What-If Scenario:** Adding {simulated_irrigation_in} inches")
    #    sim_col1, sim_col2, sim_col3 = st.columns(3)
 #       with sim_col1:
  #          st.metric("New Deficit", f"{simulated_deficit:.1f} mm",
   #                   delta=f"{simulated_deficit - decision['water_deficit_mm']:.1f} mm",
    #                  delta_color="inverse")
  #      with sim_col2:
 #           st.metric("Simulated Cost", f"${simulated_cost:.2f}/acre")
  #      with sim_col3:
   #         water_use_efficiency = (simulated_irrigation_in / decision['eto_mm'] * 25.4 * 100) if decision['eto_mm'] > 0 else 0
    #        st.metric("Water Use Efficiency", f"{water_use_efficiency:.1f}%")
    
  #  st.markdown('</div>', unsafe_allow_html=True)
    
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
    else:
        # =========================
        # MAIN OVERLAY CHART
        # =========================
        st.markdown("### 📈 Weather Trends")
        st.markdown('<p class="help-text">Multi-variable overlay chart with dual axes for temperature/variables (left) and precipitation (right)</p>', unsafe_allow_html=True)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        x = dfa[COL_DATE]
        primary_cols = [v for v in plot_vars if v != COL_PCP and v in dfa.columns]
        secondary_cols = [COL_PCP] if (COL_PCP in plot_vars and COL_PCP in dfa.columns) else []
        
        if secondary_cols and dfa[COL_PCP].notna().any():
            fig.add_bar(name="Precip (mm)", x=x, y=dfa[COL_PCP], opacity=0.55, secondary_y=True)
        
        primary_count = 0
        for col in primary_cols:
            s = dfa[col]
            if s.notna().any():
                fig.add_trace(go.Scatter(name=col, x=x, y=s, mode="lines+markers",
                                         line=dict(width=2), marker=dict(size=5)), secondary_y=False)
                primary_count += 1
        
        if primary_count == 0 and secondary_cols and dfa[COL_PCP].notna().any():
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(name="Precip (mm)", x=x, y=dfa[COL_PCP], opacity=0.55, secondary_y=False)
            primary_cols = []
        
        def finite_minmax(series_list):
            vals = pd.concat(series_list, axis=0) if series_list else pd.Series(dtype=float)
            vals = pd.to_numeric(vals, errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                return None, None
            return float(vals.min()), float(vals.max())
        
        prim_series = [dfa[c] for c in primary_cols if c in dfa.columns]
        ymin, ymax = finite_minmax(prim_series)
        if ymin is not None and ymax is not None and ymin != ymax:
            pad = 0.05 * (ymax - ymin)
            fig.update_yaxes(range=[ymin - pad, ymax + pad], secondary_y=False)
        
        if COL_PCP in secondary_cols and dfa[COL_PCP].notna().any():
            y2min, y2max = finite_minmax([dfa[COL_PCP]])
            if y2min is not None and y2max is not None:
                pad2 = 0.1 * max(1.0, (y2max - y2min))
                fig.update_yaxes(range=[0, y2max + pad2], secondary_y=True)
        
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                          template="plotly_white")
        if len(dfa):
            fig.update_xaxes(title_text="Date", range=[dfa[COL_DATE].min(), dfa[COL_DATE].max()])
        else:
            fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Value", secondary_y=False)
        fig.update_yaxes(title_text=("Precip / Cumulative (mm)" if cum_toggle else "Precip (mm)"), secondary_y=True)
        
        # Add data source badge
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"📊 Data source: Local file | Aggregation: {agg_choice} | Points: {len(dfa)}")
        
        # =========================
        # NEW! CUMULATIVE WATER BALANCE
        # =========================
        st.markdown("---")
        st.markdown("### 💧 Cumulative Water Balance")
        st.markdown('<p class="help-text">Shows cumulative precipitation vs. ET demand over time, with the balance (surplus/deficit) highlighted</p>', unsafe_allow_html=True)
        
        if "ETo (mm)" in dfx.columns and COL_PCP in dfx.columns:
            water_balance = dfx.copy()
            water_balance["Cumulative_Precip"] = water_balance[COL_PCP].cumsum()
            water_balance["Cumulative_ETo"] = water_balance["ETo (mm)"].cumsum()
            water_balance["Balance"] = water_balance["Cumulative_Precip"] - water_balance["Cumulative_ETo"]
            
            fig_balance = go.Figure()
            fig_balance.add_trace(go.Scatter(
                x=water_balance[COL_DATE], y=water_balance["Cumulative_Precip"],
                mode="lines", name="Cumulative Precip", line=dict(color="blue", width=2)
            ))
            fig_balance.add_trace(go.Scatter(
                x=water_balance[COL_DATE], y=water_balance["Cumulative_ETo"],
                mode="lines", name="Cumulative ETo", line=dict(color="orange", width=2)
            ))
            fig_balance.add_trace(go.Scatter(
                x=water_balance[COL_DATE], y=water_balance["Balance"],
                mode="lines", name="Balance (Precip - ETo)",
                line=dict(color="green", width=2, dash="dash"),
                fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'
            ))
            
            fig_balance.update_layout(
                height=350,
                template="plotly_white",
                legend=dict(orientation="h"),
                yaxis_title="Cumulative (mm)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_balance, use_container_width=True)
            
            # Balance summary
            final_balance = water_balance["Balance"].iloc[-1]
            if final_balance < 0:
                st.warning(f"⚠️ Water deficit of {abs(final_balance):.1f} mm detected. Consider irrigation.")
            else:
                st.success(f"✅ Water surplus of {final_balance:.1f} mm. Adequate moisture available.")
        
        # =========================
        # HISTORICAL TEMPERATURE CALENDAR (ORIGINAL)
        # =========================
        st.markdown("---")
        st.markdown("### 🗓️ Historical Temperature Calendar")
        st.markdown('<p class="help-text">Heatmap showing average daily temperature by day-of-year across all years in the dataset</p>', unsafe_allow_html=True)
        
        temp_var_to_plot = st.radio("Temperature Variable:", (COL_TMAX, COL_TMIN), index=0,
                                    horizontal=True, key="temp_calendar_selector")
        df_heatmap = df_et.copy()
        df_heatmap['Month'] = df_heatmap[COL_DATE].dt.month
        df_heatmap['DayOfMonth'] = df_heatmap[COL_DATE].dt.day
        avg_daily_temp = df_heatmap.groupby(['Month', 'DayOfMonth'])[temp_var_to_plot].mean().reset_index()
        pivot_table = avg_daily_temp.pivot_table(index='DayOfMonth', columns='Month', values=temp_var_to_plot)
        pivot_table = pivot_table.sort_index(axis=1)
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        x_axis_labels = [month_names[i-1] for i in pivot_table.columns]
        fig_heatmap = px.imshow(pivot_table, x=x_axis_labels, y=pivot_table.index,
                                color_continuous_scale=px.colors.sequential.thermal,
                                aspect="auto",
                                title=f"Historical Average Daily {temp_var_to_plot}",
                                labels=dict(x="Month", y="Day of Month", color="Avg Temp (°C)"))
        fig_heatmap.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10), title_x=0.5)
        fig_heatmap.update_traces(hovertemplate="<b>Month:</b> %{x}<br><b>Day:</b> %{y}<br><b>Avg Temp:</b> %{z:.1f}°C<extra></extra>")
        fig_heatmap.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_heatmap, use_container_width=True)
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
                fig_gdd_daily = px.bar(dfa, x=COL_DATE, y=COL_GDD_FP_DAILY,
                                       title="Daily GDD (°C·day)",
                                       template="plotly_white")
                fig_gdd_daily.update_traces(marker_color="lightgreen")
                st.plotly_chart(fig_gdd_daily, use_container_width=True)
        with c2:
            if COL_GDD_FP_CUM in dfa.columns:
                st.subheader("Cumulative GDD (from planting)")
                fig_gdd_cum = px.line(dfa, x=COL_DATE, y=COL_GDD_FP_CUM,
                                      title="ΣGDD from planting (°C·day)",
                                      markers=True, template="plotly_white")
                fig_gdd_cum.update_traces(line_color="darkgreen")
                st.plotly_chart(fig_gdd_cum, use_container_width=True)
        
        # Current GDD status
        if COL_GDD_FP_CUM in dfx.columns and len(dfx) > 0:
            current_gdd = dfx[COL_GDD_FP_CUM].iloc[-1]
            days_since_planting = (dfx[COL_DATE].iloc[-1] - pdate).days
            st.info(f"📊 **Current GDD Status:** {current_gdd:.1f}°C·day accumulated over {days_since_planting} days since planting ({planting_date})")
        
        # =========================
        # DATA QUALITY CHECK (ORIGINAL)
        # =========================
        with st.expander("🔎 Data Quality Check"):
            st.markdown("View first 5 rows, data types, and non-null counts to verify data integrity")
            cols_show = [c for c in plot_vars if c in dfa.columns]
            extra = [COL_GDD_STD_DAILY, COL_GDD_FP_DAILY, COL_GDD_FP_CUM]
            show = [COL_DATE] + sorted(list(set(cols_show + extra)))
            show = [c for c in show if c in dfa.columns]
            st.dataframe(dfa[show].head(), use_container_width=True)
            st.write("**Data Types:**", dfa[show].dtypes.to_frame("dtype"))
            st.write("**Non-null Counts:**")
            st.write(dfa[show].notna().sum())
        
        # =========================
        # DOWNLOADS (ORIGINAL)
        # =========================
        if allow_download:
            st.markdown("---")
            st.markdown("### 📥 Export Data")
            st.markdown('<p class="help-text">Download filtered dataset and charts for external analysis</p>', unsafe_allow_html=True)
            
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                csv = dfa.to_csv(index=False).encode("utf-8")
                st.download_button("📄 Download Data (CSV)", csv,
                                  file_name=f"weather_data_{start}_{end}.csv",
                                  mime="text/csv")
            with dl_col2:
                download_button_for_figure(fig, filename="weather_trends.png")

# ===================================================
# TAB 2: REFERENCE ET (ORIGINAL)
# ===================================================
with tab_et:
    st.title("💧 Reference Evapotranspiration (ETo & ETr)")
    st.markdown('<p class="section-desc">ASCE Standardized Penman-Monteith calculations for short (grass/ETo) and tall (alfalfa/ETr) reference crops. Use these values to estimate crop water use when multiplied by crop coefficients (Kc).</p>', unsafe_allow_html=True)
    
    try:
        df = load_climate(FILE_PATH, SHEET_NAME)
        df_et = compute_et(df, elevation_m, latitude_rad)
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
        dfe = agg_df(dfe, "MS", cols_sum=["ETo (mm)", "ETr (mm)"])
    elif et_view == "Cumulative since start":
        dfe["ETo (mm)"] = dfe["ETo (mm)"].cumsum()
        dfe["ETr (mm)"] = dfe["ETr (mm)"].cumsum()
    
    fig_et = go.Figure()
    fig_et.add_trace(go.Scatter(x=dfe[COL_DATE], y=dfe["ETo (mm)"],
                                mode="lines+markers", name="ETo (short reference)"))
    fig_et.add_trace(go.Scatter(x=dfe[COL_DATE], y=dfe["ETr (mm)"],
                                mode="lines+markers", name="ETr (tall reference)"))
    fig_et.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10),
                        legend=dict(orientation="h"), template="plotly_white",
                        yaxis_title="ET (mm)", hovermode='x unified')
    st.plotly_chart(fig_et, use_container_width=True)
    
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
            st.plotly_chart(fig24, use_container_width=True)
            
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
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=dd["Datetime"], y=dd["Temp_Min"], mode="lines", line=dict(width=0), name="Min"))
                fig_dd.add_trace(go.Scatter(x=dd["Datetime"], y=dd["Temp_Max"], mode="lines",
                                            fill='tonexty', fillcolor="rgba(255,165,0,0.15)",
                                            line=dict(width=0), name="Max"))
                fig_dd.add_trace(go.Scatter(x=dd["Datetime"], y=dd["Temperature"], mode="lines+markers",
                                            name=f"Temp ({temp_unit})", line=dict(width=2)))
                fig_dd.update_layout(height=340, template="plotly_white",
                                     margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig_dd, use_container_width=True)
        
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
                st.plotly_chart(fig_rose, use_container_width=True)
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
            st.plotly_chart(fig_combo, use_container_width=True)
    
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
    
    # Helper functions for AI
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
    
    # Build JSON snapshot
    def build_json_snapshot():
        """Build comprehensive JSON for AI analysis"""
        summary_data = {
            "meta": {
                "station": STATION_ID,
                "date_range": {"start": str(start), "end": str(end)},
                "growth_stage": growth_stage,
                "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            },
            "summary": {
                "total_precip_mm": float(dfx[COL_PCP].sum()),
                "total_eto_mm": float(dfx["ETo (mm)"].sum()) if "ETo (mm)" in dfx.columns else 0,
                "avg_temp_max": float(dfx[COL_TMAX].mean()),
                "avg_temp_min": float(dfx[COL_TMIN].mean()),
                "gdd_accumulated": float(dfx[COL_GDD_FP_CUM].iloc[-1]) if COL_GDD_FP_CUM in dfx.columns and len(dfx) > 0 else 0
            },
            # "irrigation_decision": decision,
            # "alerts": [{"type": a["type"], "title": a["title"], "message": a["message"], "action": a["action"]} 
            #           for a in alerts[:5]],
            "forecast_summary": {
                "city": city_used,
                "next_24h_temp_range": [float(dff["Temp_Min"].head(8).min()), float(dff["Temp_Max"].head(8).max())] if not dff.empty else [],
                "next_24h_rain_mm": float(dff["Rain_3h"].head(8).sum()) if not dff.empty else 0
            }
        }
        return summary_data
    
    json_snapshot = build_json_snapshot()
    
    # Display JSON
    st.markdown("### 📄 Data Snapshot")
    st.markdown('<p class="help-text">Compact JSON summary of weather data, ET, GDD, irrigation decision, and forecast for AI analysis</p>', unsafe_allow_html=True)
    
    with st.expander("View JSON Snapshot"):
        st.json(json_snapshot)
    
    json_str = json.dumps(json_snapshot, indent=2, default=_json_default)
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
            
            if use_data:
                messages = [
                    {"role": "system", "content": "Answer using the provided weather data JSON. Be concise and include units."},
                    {"role": "user", "content": f"Data:\n{json_str}\n\nQuestion: {prompt}"}
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful agricultural advisor."},
                    {"role": "user", "content": prompt}
                ]
            
            answer = _call_openai(messages, max_tokens=500, temperature=0.3)
            st.session_state.ai_chat.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
        except Exception as e:
            st.error(f"Chat error: {e}")

# Footer
st.markdown("---")
st.markdown(f'<p class="help-text" style="text-align:center">KSUTAPS Complete Weather Dashboard | Generated: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Data: {fc_source.upper()}</p>', unsafe_allow_html=True)

# streamlit run weather_dashboard_complete.py