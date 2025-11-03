# crop_dashboard_enhanced.py
# ===================================================
# KSUTAPS • Enhanced Crop Health Dashboard
# NEW FEATURES: Kc water need, Cost analysis, QA/QC, Scenario simulator
# ===================================================
import os
from glob import glob
from datetime import datetime
import random
import re
from io import BytesIO
import json

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterstats import zonal_stats

import folium
from folium.features import GeoJson, GeoJsonTooltip, Popup
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG & ENHANCED THEME
# ============================================================
st.set_page_config(page_title="Enhanced Crop Dashboard", layout="wide")

st.markdown("""
<style>
.small-muted {color:#6b7280; font-size:0.9rem;}
.metric-card {border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#ffffff}
.legend-gradient {height:14px; width:100%; border:1px solid #ccc; border-radius:6px;}
.qa-banner {
    background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    border-left: 5px solid #ffc107;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
.cost-card {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border: 2px solid #28a745;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.simulator-panel {
    background: #f8f9fa;
    border: 2px solid #dee2e6;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_IMAGE_DIR = "/Files/Index"
DEFAULT_SHP_PATH = "/Files/2024_Colby_TAPS_Harvest_Area.shp"
DEFAULT_MGMT_XLSX = "/Files/Nitrogen_24.xlsx"
DEFAULT_MGMT_N_SHEET = "Sheet1"
GHG_FILE_PATH = "/Files/GHG_2024.xlsx"

# OpenAI API Key
OPENAI_API_KEY = st.secrets["open_ai_key"]
# Index ranges
NDVI_MIN, NDVI_MAX = -0.2, 1.0
MCARI_MIN, MCARI_MAX = 0.0, 1.0

# Stress thresholds
DEFAULT_NDVI_STRESS = 0.5
DEFAULT_MCARI_STRESS = 0.3

# ============================================================
# NEW! KC CALCULATION FUNCTIONS
# ============================================================
def calculate_kc_from_ndvi(ndvi):
    """Calculate crop coefficient from NDVI using FAO method"""
    if not np.isfinite(ndvi) or ndvi < 0:
        return np.nan
    fc = 1.26 * ndvi - 0.18
    fc = np.clip(fc, 0, 1)
    kc = 1.13 * fc + 0.14
    return np.clip(kc, 0.1, 1.2)

def calculate_water_need(kc, eto_mm, effective_rainfall_mm):
    """Calculate water need in mm"""
    if not np.isfinite(kc) or not np.isfinite(eto_mm):
        return np.nan
    water_need = kc * eto_mm - effective_rainfall_mm
    return max(0, water_need)

def calculate_effective_rainfall(precip_mm):
    """USDA method for effective rainfall"""
    if precip_mm < 25:
        return precip_mm * 0.95
    else:
        return 25 * 0.95 + (precip_mm - 25) * 0.75

# ============================================================
# HELPERS (cached)
# ============================================================
@st.cache_data(show_spinner=False)
def list_tifs(directory: str):
    if not os.path.isdir(directory):
        return []
    return sorted(glob(os.path.join(directory, "*.tif")))

FNAME_RE = re.compile(r"^(?P<index>[A-Za-z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$", re.IGNORECASE)

@st.cache_data(show_spinner=False)
def build_image_catalog(directory: str):
    """Build {index: {date: path}} from TIFFs"""
    catalog = {}
    for p in list_tifs(directory):
        fn = os.path.basename(p)
        m = FNAME_RE.match(fn)
        if m:
            ix = m.group("index").upper()
            date_str = m.group("date")
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue
        else:
            try:
                ix, date_str = fn.split("_", 1)
                ix = ix.upper()
                dt = datetime.strptime(os.path.splitext(date_str)[0], "%Y-%m-%d").date()
            except Exception:
                continue
        catalog.setdefault(ix, {})[dt] = p
    for k in list(catalog.keys()):
        catalog[k] = dict(sorted(catalog[k].items(), key=lambda kv: kv[0]))
    return catalog

@st.cache_data(show_spinner=False)
def load_management_n(xlsx_path: str, sheet: str) -> pd.DataFrame:
    """Parse nitrogen schedule"""
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
    except Exception:
        return pd.DataFrame(columns=["TRT_ID", "Date", "Amount"])
    
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "TRT_ID" not in df.columns:
        return pd.DataFrame(columns=["TRT_ID", "Date", "Amount"])
    
    planting_date_col = None
    planting_amt_col = None
    date_like_cols = []
    date_hdr_re = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$")
    
    for c in df.columns:
        lc = c.lower()
        if lc == "trt_id":
            continue
        if "planting date amount" in lc:
            planting_amt_col = c
        elif "planting date" in lc:
            planting_date_col = c
        elif "lbs" in lc:
            continue
        elif date_hdr_re.match(c):
            date_like_cols.append(c)
    
    def _clean_num(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = re.sub(r"[^0-9.\-]", "", str(x))
        try:
            return float(s)
        except Exception:
            return np.nan
    
    if date_like_cols:
        df_dates = df.melt(id_vars=["TRT_ID"], value_vars=date_like_cols,
                          var_name="Date", value_name="Amount")
        df_dates["Amount"] = df_dates["Amount"].map(_clean_num)
        df_dates["Date"] = pd.to_datetime(df_dates["Date"], format="%m/%d/%y", errors="coerce")
    else:
        df_dates = pd.DataFrame(columns=["TRT_ID", "Date", "Amount"])
    
    if planting_date_col and planting_amt_col:
        df_plant = df[["TRT_ID", planting_date_col, planting_amt_col]].rename(
            columns={planting_date_col: "Date", planting_amt_col: "Amount"}
        )
        df_plant["Date"] = pd.to_datetime(df_plant["Date"], errors="coerce")
        df_plant["Amount"] = df_plant["Amount"].map(_clean_num)
    else:
        df_plant = pd.DataFrame(columns=["TRT_ID", "Date", "Amount"])
    
    out = pd.concat([df_dates, df_plant], ignore_index=True)
    out = out.dropna(subset=["Date", "Amount"])
    out["TRT_ID"] = out["TRT_ID"].astype(str)
    out["Date"] = out["Date"].dt.date
    out = out.groupby(["TRT_ID", "Date"], as_index=False)["Amount"].sum()
    return out

@st.cache_data(show_spinner=False)
def load_ghg_data(file_path: str):
    """Load GHG data"""
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_excel(file_path)
        df['TRT_ID'] = df['TRT_ID'].ffill()
        id_vars = ['TRT_ID', 'Plot']
        date_cols = [col for col in df.columns if col not in id_vars]
        df_long = pd.melt(df, id_vars=id_vars, value_vars=date_cols, 
                         var_name='Date', value_name='N2O_Flux')
        df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce', dayfirst=True)
        df_long.dropna(subset=['Date', 'N2O_Flux'], inplace=True)
        df_long['TRT_ID'] = df_long['TRT_ID'].astype(str).str.replace('T', '', regex=False)
        df_long['Date'] = df_long['Date'].dt.date
        return df_long
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def read_plots(shp_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    if "Plot_ID" not in gdf.columns:
        cand = [c for c in gdf.columns if c.lower() in ("plot_id", "plotid", "plot", "name", "id")]
        gdf["Plot_ID"] = gdf[cand[0]] if cand else np.arange(len(gdf)).astype(str)
    if "TRT_ID" not in gdf.columns:
        gdf["TRT_ID"] = "N/A"
    gdf["Plot_ID"] = gdf["Plot_ID"].astype(str)
    gdf["TRT_ID"] = gdf["TRT_ID"].astype(str)
    return gdf

def colorize_ryg(a, vmin, vmax, alpha=220):
    a = a.astype("float32", copy=False)
    mask = ~np.isfinite(a)
    t = (a - vmin) / (vmax - vmin + 1e-9)
    t = np.clip(t, 0.0, 1.0)
    r = np.where(t < 0.5, 255.0, (1.0 - 2.0 * (t - 0.5)) * 255.0)
    g = np.where(t < 0.5, (2.0 * t) * 255.0, 255.0)
    b = np.zeros_like(r)
    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = b.astype(np.uint8)
    a8 = np.where(mask, 0, alpha).astype(np.uint8)
    return np.dstack([r, g, b, a8])

def value_to_css(v, vmin, vmax):
    if not np.isfinite(v):
        return "rgba(0,0,0,0)"
    t = (v - vmin) / (vmax - vmin + 1e-9)
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        r = 255
        g = int(2.0 * t * 255)
    else:
        r = int((1.0 - 2.0 * (t - 0.5)) * 255)
        g = 255
    return f"rgba({r},{g},0,0.9)"

@st.cache_data(show_spinner="📊 Calculating zonal statistics...")
def calculate_all_zonal_stats(catalog: dict, _plots_gdf: gpd.GeoDataFrame):
    """Pre-calculate zonal statistics for ALL indices × ALL dates × ALL plots"""
    all_stats = []
    
    for index_name, date_dict in catalog.items():
        for date_val, img_path in date_dict.items():
            try:
                with rasterio.open(img_path) as src:
                    arr = src.read(1).astype("float32")
                    arr[~np.isfinite(arr)] = np.nan
                    
                    plots_native = _plots_gdf.to_crs(src.crs)
                    zs = zonal_stats(
                        plots_native,
                        arr,
                        affine=src.transform,
                        stats=["mean"],
                        nodata=np.nan,
                        all_touched=False,
                    )
                    
                    for idx, stat_dict in enumerate(zs):
                        all_stats.append({
                            'Plot_ID': _plots_gdf.iloc[idx]['Plot_ID'],
                            'TRT_ID': _plots_gdf.iloc[idx]['TRT_ID'],
                            'Index': index_name,
                            'Date': date_val,
                            'Mean': stat_dict.get('mean', np.nan)
                        })
            except Exception as e:
                st.warning(f"Error processing {index_name} on {date_val}: {e}")
                continue
    
    if not all_stats:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_stats)
    df['Plot_ID'] = df['Plot_ID'].astype(str)
    df['TRT_ID'] = df['TRT_ID'].astype(str)
    return df

# ============================================================
# NEW! QA/QC VALIDATION
# ============================================================
def run_qa_checks(master_stats: pd.DataFrame, catalog: dict) -> list:
    """Run quality assurance checks and return issues"""
    issues = []
    
    if master_stats.empty:
        issues.append({"type": "error", "message": "No zonal statistics available"})
        return issues
    
    # Check for date gaps
    for index_name in master_stats['Index'].unique():
        index_data = master_stats[master_stats['Index'] == index_name]
        dates = sorted(index_data['Date'].unique())
        if len(dates) > 1:
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            max_gap = max(gaps) if gaps else 0
            if max_gap > 10:
                issues.append({
                    "type": "warn",
                    "message": f"{index_name}: {max_gap}-day gap detected between images"
                })
    
    # Check for suspicious values
    for index_name in ['NDVI', 'MCARI2']:
        if index_name in master_stats['Index'].values:
            idx_data = master_stats[master_stats['Index'] == index_name]['Mean']
            if index_name == 'NDVI':
                suspicious = ((idx_data < -0.5) | (idx_data > 1.1)).sum()
            else:
                suspicious = (idx_data < 0).sum()
            
            if suspicious > 0:
                issues.append({
                    "type": "warn",
                    "message": f"{index_name}: {suspicious} suspicious values detected"
                })
    
    # Check data completeness
    expected_plots = master_stats['Plot_ID'].nunique()
    for index_name in master_stats['Index'].unique():
        for date in master_stats['Date'].unique():
            actual = len(master_stats[
                (master_stats['Index'] == index_name) & 
                (master_stats['Date'] == date)
            ])
            if actual < expected_plots * 0.8:
                issues.append({
                    "type": "info",
                    "message": f"{index_name} on {date}: Only {actual}/{expected_plots} plots processed"
                })
                break  # Only report first instance
    
    return issues

# ============================================================
# DATA PATHS
# ============================================================
image_dir = DEFAULT_IMAGE_DIR
shp_path = DEFAULT_SHP_PATH
mgmt_xlsx = DEFAULT_MGMT_XLSX
mgmt_sheet = DEFAULT_MGMT_N_SHEET

catalog = build_image_catalog(image_dir)
if not catalog:
    st.error("❌ No .tif files found. Expected format: NDVI_YYYY-MM-DD.tif")
    st.stop()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Dashboard Controls")

# NEW! Scenario Simulator in Sidebar
with st.sidebar.expander("🎯 Scenario Simulator", expanded=True):
    st.markdown("**Irrigation Planning**")
    sim_irrigation_in = st.slider("Planned Irrigation (inches)", 0.0, 3.0, 1.0, 0.1)
    sim_cost_per_inch = st.slider("Water Cost ($/acre-inch)", 10.0, 60.0, 25.0, 5.0)
    
    st.markdown("**Weather Inputs**")
    sim_eto_mm = st.slider("Expected ETo (mm/week)", 20.0, 60.0, 35.0, 5.0)
    sim_rainfall_mm = st.slider("Expected Rainfall (mm/week)", 0.0, 50.0, 10.0, 5.0)
    
    st.markdown("**Energy**")
    sim_energy_kwh_per_inch = st.number_input("kWh per acre-inch", value=100.0, step=10.0)
    sim_energy_cost_per_kwh = st.number_input("Cost per kWh ($)", value=0.12, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### Map Options")
index_name = st.sidebar.radio("Index:", sorted(catalog.keys()), horizontal=True)
dates = sorted(catalog[index_name].keys())
sel_date = st.sidebar.selectbox("Date:", dates, index=len(dates)-1)
opacity = st.sidebar.slider("Raster opacity", 0.1, 1.0, 0.95, 0.05)

st.sidebar.markdown("### Stress Thresholds")
ndvi_stress_threshold = st.sidebar.slider("NDVI threshold", 0.0, 1.0, DEFAULT_NDVI_STRESS, 0.01)
mcari_stress_threshold = st.sidebar.slider("MCARI2 threshold", 0.0, 1.0, DEFAULT_MCARI_STRESS, 0.01)

if index_name.upper() == "NDVI":
    stress_threshold = ndvi_stress_threshold
else:
    stress_threshold = mcari_stress_threshold

basemap = st.sidebar.selectbox("Basemap", 
                                ["Esri.WorldImagery (Satellite)", "CartoDB Positron", "Stamen Terrain"], 
                                index=0)

# Value ranges
if index_name.upper() == "NDVI":
    vmin, vmax = NDVI_MIN, NDVI_MAX
else:
    vmin, vmax = MCARI_MIN, MCARI_MAX

# ============================================================
# LOAD DATA
# ============================================================
plots = read_plots(shp_path)
plot_to_trt = dict(zip(plots["Plot_ID"], plots["TRT_ID"]))

# Pre-calculate all zonal stats
master_zonal_stats = calculate_all_zonal_stats(catalog, plots)

# Deterministic TRT colors
rng = random.Random(42)
trt_colors = {trt: f"#{rng.randrange(0x100000, 0xFFFFFF):06x}" 
              for trt in plots["TRT_ID"].unique()}

# Load management data
N_df = load_management_n(mgmt_xlsx, mgmt_sheet)
if not N_df.empty:
    N_df = N_df.groupby(["TRT_ID", "Date"], as_index=False)["Amount"].sum()

ghg_df = load_ghg_data(GHG_FILE_PATH)

# ============================================================
# CURRENT DATE PROCESSING
# ============================================================
img_path = catalog[index_name][sel_date]
with rasterio.open(img_path) as src:
    arr = src.read(1).astype("float32")
    arr[~np.isfinite(arr)] = np.nan

    # Reproject to WGS84
    dst_crs = "EPSG:4326"
    h, w = arr.shape
    left, bottom, right, top = array_bounds(h, w, src.transform)
    transform84, w84, h84 = calculate_default_transform(
        src.crs, dst_crs, w, h, left, bottom, right, top
    )
    arr84 = np.empty((h84, w84), dtype="float32")
    reproject(
        source=arr,
        destination=arr84,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform84,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    L2, B2, R2, T2 = array_bounds(h84, w84, transform84)

    # Zonal stats
    plots_native = plots.to_crs(src.crs)
    zs = zonal_stats(
        plots_native,
        arr,
        affine=src.transform,
        stats=["min", "max", "mean", "median", "std"],
        nodata=np.nan,
        all_touched=False,
    )

plots_w84 = plots.to_crs("EPSG:4326").copy()
plots_w84 = plots_w84.assign(
    Index_Min=[s.get("min", np.nan) for s in zs],
    Index_Max=[s.get("max", np.nan) for s in zs],
    Index_Mean=[s.get("mean", np.nan) for s in zs],
    Index_Median=[s.get("median", np.nan) for s in zs],
    Index_Std=[s.get("std", np.nan) for s in zs],
)
plots_w84["color"] = plots_w84["Index_Mean"].apply(lambda v: value_to_css(v, vmin, vmax))

plot_stats_df = plots_w84[[
    "Plot_ID", "TRT_ID", "Index_Min", "Index_Max", "Index_Mean", 
    "Index_Median", "Index_Std", "geometry", "color"
]].copy()

# ============================================================
# MAIN PAGE
# ============================================================
st.title(f"🌾 Enhanced Crop Health Dashboard")
st.caption(f"Index: **{index_name}** | Date: **{sel_date}** | Plots: **{len(plots_w84)}**")

# ============================================================
# NEW! QA/QC BANNER
# ============================================================
# qa_issues = run_qa_checks(master_zonal_stats, catalog)
# if qa_issues:
#     critical_issues = [i for i in qa_issues if i['type'] == 'error']
#     warnings = [i for i in qa_issues if i['type'] == 'warn']
    
#     if critical_issues or warnings:
#         st.markdown('<div class="qa-banner">', unsafe_allow_html=True)
#         st.markdown("### ⚠️ Data Quality Alerts")
#         for issue in critical_issues[:3]:
#             st.error(f"❌ {issue['message']}")
#         for issue in warnings[:3]:
#             st.warning(f"⚠️ {issue['message']}")
#         if len(qa_issues) > 6:
#             st.info(f"ℹ️ {len(qa_issues) - 6} additional QA checks flagged")
#         st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# NEW! KC-BASED WATER NEED ANALYSIS
# ============================================================
if index_name.upper() == "NDVI":
    st.markdown("---")
    st.markdown("### 💧 Water Need Analysis (Kc-Based)")
    
    current_stats = master_zonal_stats[
        (master_zonal_stats['Index'] == 'NDVI') & 
        (master_zonal_stats['Date'] == sel_date)
    ].copy()
    
    if not current_stats.empty:
        # Calculate Kc and water need
        current_stats['Kc'] = current_stats['Mean'].apply(calculate_kc_from_ndvi)
        effective_rain = calculate_effective_rainfall(sim_rainfall_mm)
        current_stats['Water_Need_mm'] = current_stats['Kc'].apply(
            lambda kc: calculate_water_need(kc, sim_eto_mm, effective_rain)
        )
        current_stats['Water_Need_in'] = current_stats['Water_Need_mm'] / 25.4
        current_stats['Cost_Estimate'] = current_stats['Water_Need_in'] * sim_cost_per_inch
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_kc = current_stats['Kc'].mean()
            st.metric("Average Kc", f"{avg_kc:.3f}")
        with col2:
            avg_need = current_stats['Water_Need_in'].mean()
            st.metric("Avg Water Need", f"{avg_need:.2f} in/week")
        with col3:
            total_cost = current_stats['Cost_Estimate'].sum() / len(plots)
            st.metric("Est. Cost", f"${total_cost:.2f}/acre")
        with col4:
            high_need_plots = (current_stats['Water_Need_in'] > 1.5).sum()
            st.metric("High Need Plots", f"{high_need_plots}/{len(current_stats)}")
        
        # Per-TRT water need table
        st.markdown("#### Water Need by Treatment")
        trt_water_summary = current_stats.groupby('TRT_ID').agg({
            'Kc': 'mean',
            'Water_Need_in': 'mean',
            'Cost_Estimate': 'mean',
            'Plot_ID': 'count'
        }).round(3)
        trt_water_summary.columns = ['Avg_Kc', 'Water_Need_in', 'Cost_$/acre', 'Plot_Count']
        trt_water_summary = trt_water_summary.sort_values('Water_Need_in', ascending=False)
        st.dataframe(trt_water_summary, use_container_width=True)
        
        # Bar chart
        fig_water = px.bar(
            trt_water_summary.reset_index(),
            x='TRT_ID',
            y='Water_Need_in',
            title="Water Need by Treatment (inches/week)",
            labels={'Water_Need_in': 'Inches', 'TRT_ID': 'Treatment'},
            color='Water_Need_in',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_water, use_container_width=True)
        
        # Export
        csv_water = current_stats[['Plot_ID', 'TRT_ID', 'Kc', 'Water_Need_in', 'Cost_Estimate']].to_csv(index=False)
        st.download_button(
            "📥 Download Water Need Data (CSV)",
            csv_water.encode('utf-8'),
            file_name=f"water_need_{sel_date}.csv",
            mime="text/csv"
        )

# ============================================================
# NEW! COST ANALYSIS DASHBOARD
# ============================================================
# st.markdown("---")
# st.markdown("### 💰 Cost Analysis")

# st.markdown('<div class="cost-card">', unsafe_allow_html=True)

# cost_col1, cost_col2, cost_col3 = st.columns(3)

# with cost_col1:
#     st.markdown("**💧 Irrigation Costs**")
#     total_acres = len(plots)  # Simplified: 1 plot = 1 acre
#     irrigation_cost = sim_irrigation_in * sim_cost_per_inch * total_acres
#     st.metric("Total Irrigation", f"${irrigation_cost:,.2f}")
#     st.caption(f"{sim_irrigation_in} in × ${sim_cost_per_inch}/in × {total_acres} acres")

# with cost_col2:
#     st.markdown("**⚡ Energy Costs**")
#     energy_kwh = sim_irrigation_in * sim_energy_kwh_per_inch * total_acres
#     energy_cost = energy_kwh * sim_energy_cost_per_kwh
#     st.metric("Energy Cost", f"${energy_cost:,.2f}")
#     st.caption(f"{energy_kwh:,.0f} kWh × ${sim_energy_cost_per_kwh}/kWh")

# with cost_col3:
#     st.markdown("**🌱 Nitrogen Applied**")
#     if not N_df.empty:
#         total_n = N_df['Amount'].sum()
#         st.metric("Total N", f"{total_n:.1f} lbs/acre")
#         # Assuming $0.50/lb N cost
#         n_cost = total_n * 0.50
#         st.metric("N Cost Est.", f"${n_cost:,.2f}")
#     else:
#         st.info("No N data")

# # Per-TRT cost breakdown
# if not N_df.empty:
#     st.markdown("#### Cost Breakdown by Treatment")
#     trt_n_totals = N_df.groupby('TRT_ID')['Amount'].sum().reset_index()
#     trt_n_totals.columns = ['TRT_ID', 'Total_N_lbs']
#     trt_n_totals['N_Cost'] = trt_n_totals['Total_N_lbs'] * 0.50
#     trt_n_totals['Irrigation_Cost'] = sim_irrigation_in * sim_cost_per_inch
#     trt_n_totals['Energy_Cost'] = sim_irrigation_in * sim_energy_kwh_per_inch * sim_energy_cost_per_kwh
#     trt_n_totals['Total_Cost'] = trt_n_totals[['N_Cost', 'Irrigation_Cost', 'Energy_Cost']].sum(axis=1)
    
#     st.dataframe(trt_n_totals.round(2), use_container_width=True)
    
#     # Cost comparison chart
#     fig_cost = px.bar(
#         trt_n_totals,
#         x='TRT_ID',
#         y=['N_Cost', 'Irrigation_Cost', 'Energy_Cost'],
#         title="Cost Components by Treatment",
#         labels={'value': 'Cost ($)', 'TRT_ID': 'Treatment'},
#         barmode='stack'
#     )
#     st.plotly_chart(fig_cost, use_container_width=True)

# st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# MAP VISUALIZATION
# ============================================================
st.markdown("---")
st.markdown("### 🗺️ Spatial View")

if basemap.startswith("Esri"):
    tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    attr = "Esri"
elif "Positron" in basemap:
    tiles = "CartoDB positron"
    attr = None
else:
    tiles = "Stamen Terrain"
    attr = None

center_lat = (T2 + B2) / 2.0
center_lon = (L2 + R2) / 2.0
m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles=tiles, attr=attr)

# Raster overlay
rgba84 = colorize_ryg(arr84, vmin=vmin, vmax=vmax, alpha=int(opacity * 255))
folium.raster_layers.ImageOverlay(
    image=rgba84,
    bounds=[[B2, L2], [T2, R2]],
    opacity=1.0
).add_to(m)

# NEW! Enhanced polygon layer with click popup
def style_fn(feature):
    val = feature["properties"].get("Index_Mean", np.nan)
    outline = "#C62828" if (np.isfinite(val) and val < stress_threshold) else "#222222"
    return {
        "fillColor": feature["properties"]["color"],
        "color": outline,
        "weight": 2,
        "fillOpacity": 0.6
    }

# Add interactive popups
for _, row in plots_w84.iterrows():
    plot_id = row['Plot_ID']
    trt_id = row['TRT_ID']
    mean_val = row['Index_Mean']
    
    # Get mini time series for this plot
    plot_history = master_zonal_stats[
        (master_zonal_stats['Plot_ID'] == plot_id) &
        (master_zonal_stats['Index'] == index_name)
    ].sort_values('Date').tail(5)
    
    history_html = "<br>".join([
        f"{d}: {v:.3f}" for d, v in zip(plot_history['Date'], plot_history['Mean'])
    ])
    
    popup_html = f"""
    <div style='width:200px'>
        <h4>Plot {plot_id}</h4>
        <b>TRT:</b> {trt_id}<br>
        <b>Current {index_name}:</b> {mean_val:.3f}<br>
        <hr>
        <b>Recent History:</b><br>
        <small>{history_html}</small>
    </div>
    """
    
    folium.CircleMarker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        radius=5,
        popup=folium.Popup(popup_html, max_width=250),
        color='white',
        fillColor=row['color'],
        fillOpacity=0.8,
        weight=1
    ).add_to(m)

geojson = GeoJson(
    plots_w84,
    style_function=style_fn,
    tooltip=GeoJsonTooltip(
        fields=["Plot_ID", "TRT_ID", "Index_Mean"],
        aliases=["Plot", "TRT", f"{index_name}"],
        localize=True,
        sticky=True,
    ),
)
m.add_child(geojson)

st_folium(m, height=560, use_container_width=True)

# Legend
st.sidebar.markdown(f"### {index_name} Legend")
swatch_html = (
    "<div class='legend-gradient' style=\"background: linear-gradient(90deg, rgb(255,0,0), rgb(255,255,0), rgb(0,255,0));\"></div>"
)
st.sidebar.markdown(swatch_html, unsafe_allow_html=True)
minmax_cols = st.sidebar.columns(2)
minmax_cols[0].markdown(f"<span class='small-muted'>{vmin:.2f}</span>", unsafe_allow_html=True)
minmax_cols[1].markdown(f"<div style='text-align:right' class='small-muted'>{vmax:.2f}</div>", unsafe_allow_html=True)

# ============================================================
# SUMMARY STATISTICS
# ============================================================
st.markdown("---")
st.markdown(f"## 📊 {index_name} Summary for {sel_date}")

current_stats = master_zonal_stats[
    (master_zonal_stats['Index'] == index_name) & 
    (master_zonal_stats['Date'] == sel_date)
].copy()

if not current_stats.empty:
    overall_mean = current_stats['Mean'].mean()
    overall_min = current_stats['Mean'].min()
    overall_max = current_stats['Mean'].max()
    total_plots = len(current_stats)
    stressed_plots = (current_stats['Mean'] < stress_threshold).sum()
    
    trt_means = current_stats.groupby('TRT_ID')['Mean'].mean().sort_values(ascending=False)
    best_trt = trt_means.index[0] if len(trt_means) > 0 else "N/A"
    best_trt_mean = trt_means.iloc[0] if len(trt_means) > 0 else 0
    
    # Temporal trend
    index_dates = sorted(catalog[index_name].keys())
    if len(index_dates) > 1:
        first_date = index_dates[0]
        last_date = index_dates[-1]
        
        first_mean = master_zonal_stats[
            (master_zonal_stats['Index'] == index_name) & 
            (master_zonal_stats['Date'] == first_date)
        ]['Mean'].mean()
        
        last_mean = master_zonal_stats[
            (master_zonal_stats['Index'] == index_name) & 
            (master_zonal_stats['Date'] == last_date)
        ]['Mean'].mean()
        
        trend_pct = ((last_mean - first_mean) / first_mean) * 100 if first_mean > 0 else 0
    else:
        trend_pct = 0
    
    # Display metrics
    metric_cols = st.columns(5)
    
    with metric_cols[0]:
        st.metric("Overall Mean", f"{overall_mean:.4f}",
                 delta=f"{trend_pct:+.1f}% trend" if len(index_dates) > 1 else None)
    
    with metric_cols[1]:
        st.metric("Range", f"{overall_min:.4f} → {overall_max:.4f}")
    
    with metric_cols[2]:
        stress_color = "🔴" if stressed_plots > 0 else "🟢"
        st.metric("Stressed Plots", f"{stress_color} {stressed_plots}/{total_plots}",
                 delta=f"< {stress_threshold:.2f}" if stressed_plots > 0 else "None",
                 delta_color="inverse" if stressed_plots > 0 else "off")
    
    with metric_cols[3]:
        st.metric("Best TRT", f"TRT {best_trt}",
                 delta=f"mean: {best_trt_mean:.4f}", delta_color="off")
    
    with metric_cols[4]:
        st.metric("Total Plots", total_plots)
    
    # TRT comparison
    st.markdown("### Treatment Comparison")
    trt_comparison = current_stats.groupby('TRT_ID').agg({
        'Mean': ['mean', 'count']
    }).round(4)
    trt_comparison.columns = ['Mean_Value', 'Plot_Count']
    trt_comparison = trt_comparison.sort_values('Mean_Value', ascending=False).reset_index()
    
    trt_comparison['Rank'] = ['🥇', '🥈', '🥉'] + [''] * (len(trt_comparison) - 3)
    trt_comparison = trt_comparison[['Rank', 'TRT_ID', 'Mean_Value', 'Plot_Count']]
    
    st.dataframe(trt_comparison, use_container_width=True, hide_index=True)
else:
    st.warning("No data available for the selected index and date.")

# ============================================================
# TIME SERIES ANALYSIS
# ============================================================
st.markdown("---")
st.markdown("### 📈 Time Series Analysis")

ts_option = st.radio("Series type:", ["Plot_ID", "TRT_ID"], horizontal=True)
choices = plot_stats_df[ts_option].astype(str).unique()
sel_ids = st.multiselect(f"Select {ts_option}(s):", sorted(choices))
show_nitrogen = st.checkbox("Show Nitrogen Overlay", value=True)

if sel_ids:
    ts = master_zonal_stats[master_zonal_stats['Index'] == index_name].copy()
    
    fig_ts = go.Figure()
    if ts_option == "TRT_ID":
        for sid in sel_ids:
            sid = str(sid)
            sub = ts[ts["TRT_ID"] == sid].groupby("Date", as_index=False)["Mean"].mean()
            color = trt_colors.get(sid, f'#{random.randint(0, 0xFFFFFF):06x}')
            fig_ts.add_trace(go.Scatter(
                x=sub["Date"], y=sub["Mean"],
                mode="lines+markers", name=f"TRT {sid}",
                line=dict(color=color, width=2.5)
            ))
            if show_nitrogen and not N_df.empty:
                Nsub = N_df[N_df["TRT_ID"] == sid]
                if not Nsub.empty:
                    fig_ts.add_trace(go.Bar(
                        x=Nsub["Date"], y=Nsub["Amount"],
                        name=f"N (TRT {sid})",
                        marker_color=color, opacity=0.35, yaxis="y2"
                    ))
    else:
        palette = {}
        for sid in sel_ids:
            sid = str(sid)
            sub = ts[ts["Plot_ID"] == sid].sort_values("Date")
            color = palette.setdefault(sid, f'#{random.randint(0, 0xFFFFFF):06x}')
            fig_ts.add_trace(go.Scatter(
                x=sub["Date"], y=sub["Mean"],
                mode="lines+markers", name=f"Plot {sid}",
                line=dict(color=color)
            ))
            if show_nitrogen and not N_df.empty:
                trt_for_plot = plot_to_trt.get(sid)
                if trt_for_plot:
                    Nsub = N_df[N_df["TRT_ID"] == trt_for_plot]
                    if not Nsub.empty:
                        fig_ts.add_trace(go.Bar(
                            x=Nsub["Date"], y=Nsub["Amount"],
                            name=f"N (TRT {trt_for_plot})",
                            marker_color=trt_colors.get(trt_for_plot, "#888"),
                            opacity=0.35, yaxis="y2"
                        ))
    
    fig_ts.update_layout(
        title=f"{index_name} over time",
        xaxis_title="Date",
        yaxis=dict(title=f"{index_name} (mean)", side="left"),
        yaxis2=dict(title="N amount (lbs/ac)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h"),
        barmode="overlay",
        height=450,
        template="plotly_white"
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Export time series
    csv_ts = ts.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"📥 Download Time Series CSV",
        data=csv_ts,
        file_name=f"timeseries_{index_name}_{ts_option}.csv",
        mime="text/csv"
    )
else:
    st.caption("Select one or more IDs to draw a time series.")

# ============================================================
# GHG INTEGRATION
# ============================================================
if not ghg_df.empty:
    st.markdown("---")
    st.markdown("### 🌍 Crop Health vs. N₂O Emissions")
    
    # =================================================================================
# INTEGRATED ANALYSIS — CROP HEALTH (REMOTE SENSING) vs. N2O EMISSIONS (FIELD)
# =================================================================================
st.subheader("Integrated Analysis: Crop Health vs. N₂O Emissions")

# --- Step 1: Define Hardcoded Path and Parser ---
GHG_FILE_PATH = "/Users/rayhaankabenge/Desktop/KSUTAPS/2025/Dashboard_v2/Data_Source/GHG_2024.xlsx"

@st.cache_data(show_spinner="Processing GHG data...")
def load_ghg_data_from_path(file_path):
    """Loads and transforms wide GHG data from a file path."""
    if not os.path.exists(file_path):
        st.warning(f"GHG data file not found at the specified path: {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(file_path)
        df['TRT_ID'] = df['TRT_ID'].ffill()
        id_vars = ['TRT_ID', 'Plot']
        date_cols = [col for col in df.columns if col not in id_vars]
        df_long = pd.melt(df, id_vars=id_vars, value_vars=date_cols, var_name='Date', value_name='N2O_Flux')
        df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce', dayfirst=True)
        df_long.dropna(subset=['Date', 'N2O_Flux'], inplace=True)
        df_long['TRT_ID'] = df_long['TRT_ID'].astype(str)
        df_long['TRT_ID'] = df_long['TRT_ID'].str.replace('T', '', regex=False)
        df_long['Date'] = df_long['Date'].dt.date
        return df_long
    except Exception as e:
        st.error(f"Error processing GHG file: {e}")
        return pd.DataFrame()

# Load the data
ghg_df = load_ghg_data_from_path(GHG_FILE_PATH)

if not ghg_df.empty:
    st.success("Greenhouse gas data loaded and standardized successfully!")
    
    # --- Step 2: Calculate Summary Statistics ---
    ghg_stats = ghg_df.groupby('TRT_ID').agg({
        'N2O_Flux': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)
    ghg_stats.columns = ['_'.join(col).strip() for col in ghg_stats.columns.values]
    ghg_stats.reset_index(inplace=True)
    
    # Calculate temporal trends
    first_last = ghg_df.groupby('TRT_ID').agg({
        'Date': ['min', 'max'],
        'N2O_Flux': ['first', 'last']
    })
    first_last.columns = ['_'.join(col).strip() for col in first_last.columns.values]
    first_last['trend'] = ((first_last['N2O_Flux_last'] - first_last['N2O_Flux_first']) / 
                           first_last['N2O_Flux_first'] * 100).round(1)
    
    # --- Step 3: Display KPIs ---
    st.markdown("### 📊 N₂O Flux Overview")
    
    kpi_cols = st.columns(4)
    
    with kpi_cols[0]:
        overall_mean = ghg_df['N2O_Flux'].mean()
        st.metric("Overall Mean N₂O Flux", f"{overall_mean:.4f}", 
                 delta=None, delta_color="off")
    
    with kpi_cols[1]:
        overall_max = ghg_df['N2O_Flux'].max()
        max_trt = ghg_df.loc[ghg_df['N2O_Flux'].idxmax(), 'TRT_ID']
        st.metric("Maximum Flux", f"{overall_max:.4f}", 
                 delta=f"TRT {max_trt}", delta_color="off")
    
    with kpi_cols[2]:
        overall_min = ghg_df['N2O_Flux'].min()
        negative_count = (ghg_df['N2O_Flux'] < 0).sum()
        st.metric("Minimum Flux", f"{overall_min:.4f}", 
                 delta=f"{negative_count} negative", delta_color="inverse")
    
    with kpi_cols[3]:
        date_range = f"{ghg_df['Date'].min()} to {ghg_df['Date'].max()}"
        n_dates = ghg_df['Date'].nunique()
        st.metric("Sampling Dates", n_dates, 
                 delta=None, delta_color="off")
    
    st.divider()
    
    # --- Step 4: Auto-Generated Insights ---
    st.markdown("### 🔍 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**Treatment Rankings (by avg flux):**")
        ranked = ghg_stats.sort_values('N2O_Flux_mean', ascending=False)
        for idx, row in ranked.iterrows():
            emoji = "🔴" if idx == 0 else "🟡" if idx == 1 else "🟢"
            st.markdown(f"{emoji} **TRT {row['TRT_ID']}**: {row['N2O_Flux_mean']:.4f} (±{row['N2O_Flux_std']:.4f})")
    
    with insights_col2:
        st.markdown("**Temporal Trends:**")
        for trt_id in sorted(ghg_df['TRT_ID'].unique()):
            if trt_id in first_last.index:
                trend_val = first_last.loc[trt_id, 'trend']
                trend_emoji = "📈" if trend_val > 0 else "📉"
                st.markdown(f"{trend_emoji} **TRT {trt_id}**: {trend_val:+.1f}% change")
    
    # Alert for negative values
    if negative_count > 0:
        st.warning(f"⚠️ **Alert**: {negative_count} negative flux readings detected, indicating possible N₂O uptake or measurement issues.")
    
    st.divider()
    
    # --- Step 5: Treatment Selection ---
    available_trts = sorted(list(set(plot_stats_df['TRT_ID']) & set(ghg_df['TRT_ID'])))
    
    if not available_trts:
        st.warning("Data loaded, but no common TRT_IDs were found after standardization.")
    
    sel_trts_integrated = st.multiselect(
        "Select TRT_ID(s) for detailed comparison:",
        options=available_trts,
        default=available_trts[:2] if available_trts else []
    )

    if sel_trts_integrated:
        # --- Step 6: Prepare Remote Sensing Data ---
        rs_rows = []
        for d, pth in catalog[index_name].items():
            with rasterio.open(pth) as s:
                band = s.read(1).astype("float32")
                band[~np.isfinite(band)] = np.nan
                gdf_native = plots.to_crs(s.crs)
                zs_d = zonal_stats(gdf_native, band, affine=s.transform, stats=["mean"], nodata=np.nan)
                tmp = pd.DataFrame({
                    "TRT_ID": plots["TRT_ID"].astype(str).values,
                    "Mean_Index": [z.get("mean", np.nan) for z in zs_d],
                    "Date": d,
                })
                rs_rows.append(tmp)
        
        rs_ts_df = pd.concat(rs_rows, ignore_index=True).dropna()
        rs_ts_df = rs_ts_df.groupby(['TRT_ID', 'Date'], as_index=False)['Mean_Index'].mean()
        
        ghg_ts_df = ghg_df.groupby(['TRT_ID', 'Date'], as_index=False)['N2O_Flux'].mean()

        # --- Step 7: Dual-Axis Time Series Chart ---
        st.markdown("### 📈 Time Series: Crop Health vs. N₂O Emissions")
        
        fig_integrated = go.Figure()

        for trt_id in sel_trts_integrated:
            color = trt_colors.get(str(trt_id), f'#{random.randint(0, 0xFFFFFF):06x}')
            
            rs_sub = rs_ts_df[rs_ts_df['TRT_ID'] == trt_id]
            fig_integrated.add_trace(go.Scatter(
                x=rs_sub["Date"], y=rs_sub["Mean_Index"],
                mode="lines+markers", name=f"{index_name} (TRT {trt_id})",
                line=dict(color=color, width=2.5), marker=dict(size=8),
                yaxis="y1", hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>"
            ))
            
            ghg_sub = ghg_ts_df[ghg_ts_df['TRT_ID'] == trt_id]
            fig_integrated.add_trace(go.Bar(
                x=ghg_sub["Date"], y=ghg_sub["N2O_Flux"],
                name=f"N₂O Flux (TRT {trt_id})",
                marker_color=color, opacity=0.5,
                yaxis="y2", hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Flux: %{y:.4f}<extra></extra>"
            ))

        fig_integrated.update_layout(
            title="Crop Health vs. N₂O Emissions Over Time",
            xaxis_title="Date",
            yaxis=dict(title=f"<b>{index_name} (Mean)</b>", side="left", color="#1f77b4"),
            yaxis2=dict(title="<b>N₂O Flux</b>", overlaying="y", side="right", showgrid=False, color="#ff7f0e"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode="group", hovermode="x unified", height=500
        )
        
        st.plotly_chart(fig_integrated, use_container_width=True)
        
        # --- Step 8: Heatmap of All Plots ---
        st.markdown("### 🔥 N₂O Flux Heatmap (All Plots)")
        
        # Filter for selected treatments
        ghg_filtered = ghg_df[ghg_df['TRT_ID'].isin(sel_trts_integrated)].copy()
        ghg_filtered['Plot_Label'] = ghg_filtered['TRT_ID'] + '-' + ghg_filtered['Plot'].astype(str)
        
        # Pivot for heatmap
        heatmap_data = ghg_filtered.pivot_table(
            index='Plot_Label', 
            columns='Date', 
            values='N2O_Flux',
            aggfunc='mean'
        )
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[str(d) for d in heatmap_data.columns],
            y=heatmap_data.index,
            colorscale='RdBu_r',
            zmid=0,
            text=heatmap_data.values.round(4),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="N₂O Flux"),
            hovertemplate="Plot: %{y}<br>Date: %{x}<br>Flux: %{z:.4f}<extra></extra>"
        ))
        
        fig_heatmap.update_layout(
            title="N₂O Flux by Plot and Date",
            xaxis_title="Date",
            yaxis_title="Plot (TRT-Plot)",
            height=400 + len(heatmap_data.index) * 20
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # --- Step 9: Box Plot Distribution ---
        st.markdown("### 📦 N₂O Flux Distribution by Treatment")
        
        fig_box = go.Figure()
        
        for trt_id in sel_trts_integrated:
            trt_data = ghg_filtered[ghg_filtered['TRT_ID'] == trt_id]
            color = trt_colors.get(str(trt_id), f'#{random.randint(0, 0xFFFFFF):06x}')
            
            for date in sorted(trt_data['Date'].unique()):
                date_data = trt_data[trt_data['Date'] == date]
                fig_box.add_trace(go.Box(
                    y=date_data['N2O_Flux'],
                    name=f"TRT {trt_id}",
                    x=[str(date)] * len(date_data),
                    marker_color=color,
                    legendgroup=trt_id,
                    showlegend=(date == sorted(trt_data['Date'].unique())[0]),
                    boxmean='sd',
                    hovertemplate="<b>TRT %{fullData.name}</b><br>Date: %{x}<br>Flux: %{y:.4f}<extra></extra>"
                ))
        
        fig_box.update_layout(
            title="N₂O Flux Distribution Over Time",
            xaxis_title="Date",
            yaxis_title="N₂O Flux",
            boxmode='group',
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        
        # --- Step 10: Statistical Summary Table ---
        with st.expander("📊 View Detailed Statistics"):
            summary_stats = ghg_filtered.groupby(['TRT_ID', 'Date']).agg({
                'N2O_Flux': ['mean', 'std', 'min', 'max', 'count']
            }).round(4)
            summary_stats.columns = ['Mean', 'Std Dev', 'Min', 'Max', 'N Plots']
            st.dataframe(summary_stats, use_container_width=True)
        
    else:
        st.caption("Select one or more TRT IDs to visualize the integrated data.")
else:
    st.info("Could not load GHG data. Please check the file path in the code and ensure the file is accessible.")

# ============================================================
# DOWNLOADABLE SUMMARIES
# ============================================================
st.markdown("---")
st.markdown("### 📥 Downloadable Summary Tables")

if not master_zonal_stats.empty:
    plot_summary = master_zonal_stats.groupby(['Plot_ID', 'TRT_ID', 'Index'])['Mean'].mean().reset_index()
    plot_summary_pivot = plot_summary.pivot(index=['Plot_ID', 'TRT_ID'], columns='Index', values='Mean').reset_index()
    plot_summary_pivot.columns.name = None
    
    trt_summary = master_zonal_stats.groupby(['TRT_ID', 'Index'])['Mean'].mean().reset_index()
    trt_summary_pivot = trt_summary.pivot(index='TRT_ID', columns='Index', values='Mean').reset_index()
    trt_summary_pivot.columns.name = None
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        with st.expander("📊 Plot-Level Summary", expanded=False):
            st.dataframe(plot_summary_pivot, use_container_width=True)
            st.caption(f"Total: {len(plot_summary_pivot)} plots")
        
        csv_plot = plot_summary_pivot.to_csv(index=False).encode('utf-8')
        json_plot = plot_summary_pivot.to_json(orient='records', indent=2).encode('utf-8')
        
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button("📄 CSV", csv_plot, "plot_summary_all_indices.csv", "text/csv")
        with dl_col2:
            st.download_button("📋 JSON", json_plot, "plot_summary_all_indices.json", "application/json")
    
    with download_col2:
        with st.expander("📊 TRT-Level Summary", expanded=False):
            st.dataframe(trt_summary_pivot, use_container_width=True)
            st.caption(f"Total: {len(trt_summary_pivot)} treatments")
        
        csv_trt = trt_summary_pivot.to_csv(index=False).encode('utf-8')
        json_trt = trt_summary_pivot.to_json(orient='records', indent=2).encode('utf-8')
        
        dl_col3, dl_col4 = st.columns(2)
        with dl_col3:
            st.download_button("📄 CSV", csv_trt, "trt_summary_all_indices.csv", "text/csv")
        with dl_col4:
            st.download_button("📋 JSON", json_trt, "trt_summary_all_indices.json", "application/json")
    
    # Excel export
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        plot_summary_pivot.to_excel(writer, sheet_name='Plot_Level', index=False)
        trt_summary_pivot.to_excel(writer, sheet_name='TRT_Level', index=False)
    excel_buffer.seek(0)
    
    st.download_button(
        "📊 Download Both Tables (Excel)",
        excel_buffer,
        "summary_all_indices.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# ============================================================
# AI INSIGHTS
# ============================================================
st.markdown("---")
st.markdown("### 🤖 AI Insights & Decision Support")

try:
    from openai import OpenAI
    
    if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Build compact summary
        ai_summary = {
            "current_date": str(sel_date),
            "index": index_name,
            "overall_stats": {
                "mean": float(overall_mean) if not current_stats.empty else None,
                "stressed_plots": int(stressed_plots) if not current_stats.empty else 0,
                "total_plots": int(total_plots) if not current_stats.empty else 0,
                "best_treatment": str(best_trt) if not current_stats.empty else None
            },
            "water_need": {
                "avg_inches": float(current_stats['Water_Need_in'].mean()) if 'Water_Need_in' in current_stats.columns else None,
                "total_cost": float(current_stats['Cost_Estimate'].sum()) if 'Cost_Estimate' in current_stats.columns else None
            },
            # "qa_issues": len(qa_issues),
            # "scenario": {
            #     "planned_irrigation_in": float(sim_irrigation_in),
            #     "cost_per_inch": float(sim_cost_per_inch),
            #     "expected_eto_mm": float(sim_eto_mm),
            #     "expected_rain_mm": float(sim_rainfall_mm)
            
        }
        
        with st.expander("💡 One-Click Analysis", expanded=True):
            user_prompt = st.text_area(
                "Optional: Customize your analysis",
                placeholder="e.g., Focus on stressed plots and cost optimization"
            )
            
            if st.button("🤖 Generate AI Analysis"):
                with st.spinner("Analyzing crop health data..."):
                    try:
                        messages = [
                            {
                                "role": "system",
                                "content": "You are an agronomic data analyst. Provide concise, actionable insights from crop health monitoring data. Format your response in clear sections: Key Findings, Recommendations, and Priority Actions."
                            },
                            {
                                "role": "user",
                                "content": f"Crop health data summary:\n{json.dumps(ai_summary, indent=2)}\n\n{user_prompt or 'Provide comprehensive analysis and recommendations.'}"
                            }
                        ]
                        
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            max_tokens=800,
                            temperature=0.2
                        )
                        
                        st.markdown("**AI Analysis:**")
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI analysis error: {e}")
        
        # Chat interface
        st.markdown("### 💬 Chat with Your Data")
        
        if "crop_chat" not in st.session_state:
            st.session_state.crop_chat = []
        
        for msg in st.session_state.crop_chat:
            st.chat_message(msg["role"]).write(msg["content"])
        
        if prompt := st.chat_input("Ask about your crop health data..."):
            st.session_state.crop_chat.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful crop health advisor. Answer questions based on the provided data summary."
                    },
                    {
                        "role": "user",
                        "content": f"Data context:\n{json.dumps(ai_summary)}\n\nQuestion: {prompt}"
                    }
                ]
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.3
                )
                
                answer = response.choices[0].message.content
                st.session_state.crop_chat.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)
            except Exception as e:
                st.error(f"Chat error: {e}")
    else:
        st.info("💡 Configure OpenAI API key to enable AI insights")
except ImportError:
    st.info("💡 Install OpenAI package: `pip install openai`")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(f"Enhanced Crop Dashboard | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total Plots: {len(plots)} |")

# streamlit run crop_dashboard_enhanced.py