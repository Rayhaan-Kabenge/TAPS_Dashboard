# crop_dashboard_ai_enhanced.py
# ===================================================
# KSUTAPS • AI-Enhanced Crop Health Dashboard
# Features: Full AI integration with function calling, Quick Insights, Advanced Analytics
# ===================================================

from __future__ import annotations

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
from folium.features import GeoJson, GeoJsonTooltip
from streamlit_folium import st_folium
import plotly.graph_objects as go

# Import AI tools (your companion module with function implementations)
import ai_tools
import pandas as pd
import numpy as np
from datetime import date as _date
from typing import Optional, Union, Dict, Any

# ============================================================
# PAGE CONFIG & THEME
# ============================================================
st.set_page_config(page_title="AI-Enhanced Crop Dashboard", layout="wide")

st.markdown("""
<style>
.small-muted {color:#6b7280; font-size:0.9rem;}
.metric-card {
    border:1px solid #e5e7eb; 
    border-radius:12px; 
    padding:12px; 
    background:#ffffff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.legend-gradient {height:14px; width:100%; border:1px solid #ccc; border-radius:6px;}
.insight-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.alert-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
}
.success-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
}
.ai-chat-container {
    background: #f8f9fa;
    border: 2px solid #e9ecef;
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_IMAGE_DIR = "Files/Index"
DEFAULT_SHP_PATH = "Files/2024_Colby_TAPS_Harvest_Area.shp"
DEFAULT_MGMT_XLSX = "Files/Nitrogen_24.xlsx"
DEFAULT_MGMT_N_SHEET = "Sheet1"
GHG_FILE_PATH = "Files/GHG_2024.xlsx"

# --- OpenAI client init (secure) ---
OPENAI_API_KEY = st.secrets["open_ai_key"]

client = None
try:
    if OPENAI_API_KEY and OPENAI_API_KEY.startswith(("sk-", "sk-proj-")):
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        st.info("💡 Set OPENAI_API_KEY in `.streamlit/secrets.toml` or environment to enable the AI assistant.")
except Exception as _e:
    st.warning(f"OpenAI not available: {_e}")
    client = None

# Index ranges
NDVI_MIN, NDVI_MAX = -0.2, 1.0
MCARI_MIN, MCARI_MAX = 0.0, 1.0

# Stress thresholds
DEFAULT_NDVI_STRESS = 0.5
DEFAULT_MCARI_STRESS = 0.3


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
    """Load GHG data (TRT_ID x Date -> N2O_Flux)"""
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


def value_to_css(v, vmin, vmax):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "rgba(0,0,0,0)"
    t = (float(v) - vmin) / (vmax - vmin + 1e-9)
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        r = 255
        g = int(2.0 * t * 255)
    else:
        r = int((1.0 - 2.0 * (t - 0.5)) * 255)
        g = 255
    return f"rgba({r},{g},0,0.9)"


@st.cache_data(show_spinner="📊 Calculating zonal statistics...")
def calculate_all_zonal_stats(catalog: dict, _plots_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Pre-calculate zonal statistics (mean) for ALL indices × dates × plots."""
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
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    return df

def _coerce_date(d) -> Optional[_date]:
    if d is None or pd.isna(d):
        return None
    if isinstance(d, _date):
        return d
    try:
        return pd.to_datetime(d, errors="coerce").date()
    except Exception:
        return None

def calculate_treatment_percentile(
    master_zonal_stats: pd.DataFrame,
    index_name: str,
    date: Optional[Union[str, _date]] = None,
    metric: str = "mean",              # "mean" | "median" | "max"
    trt_id: Optional[str] = None,
    higher_is_better: bool = True,     # True: higher index ⇒ better percentile
    decimals: int = 3
) -> Dict[str, Any]:
    """
    Compute per-treatment performance for a given index (optionally on a specific date)
    and the percentile of a chosen treatment.

    Returns:
      {
        "index_name": str,
        "date": date_used or None,
        "metric": str,
        "n_treatments": int,
        "table": [ { "TRT_ID": str, "value": float, "rank": int, "percentile": float }, ... ],
        # present only if trt_id provided:
        "trt_id": str,
        "value": float,
        "rank": int,
        "percentile": float
      }
    """
    if master_zonal_stats is None or master_zonal_stats.empty:
        return {"error": "master_zonal_stats is empty."}

    idx = str(index_name).upper()
    df = master_zonal_stats[master_zonal_stats["Index"].str.upper() == idx].copy()
    if df.empty:
        return {"error": f"No records for index '{index_name}'."}

    # pick date (default: latest available for this index)
    date_used = _coerce_date(date)
    if date_used is None:
        # ensure Date is date dtype
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        avail = df["Date"].dropna().unique()
        if len(avail) == 0:
            return {"error": f"No valid dates found for index '{index_name}'."}
        date_used = sorted(avail)[-1]
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    df = df[df["Date"] == date_used]
    if df.empty:
        return {"error": f"No data for index '{index_name}' on {date_used}."}

    # aggregate by treatment
    agg_map = {
        "mean": lambda s: float(np.nanmean(s.values)) if len(s) else np.nan,
        "median": lambda s: float(np.nanmedian(s.values)) if len(s) else np.nan,
        "max": lambda s: float(np.nanmax(s.values)) if len(s) else np.nan,
    }
    if metric not in agg_map:
        return {"error": f"Unsupported metric '{metric}'. Use one of: mean, median, max."}

    g = (df.groupby("TRT_ID", as_index=False)["Mean"]
            .apply(agg_map[metric])
            .rename(columns={"Mean": "value"}))
    g["TRT_ID"] = g["TRT_ID"].astype(str)

    # drop NaN values (no valid plots for that TRT at that date)
    g = g[np.isfinite(g["value"])].copy()
    if g.empty:
        return {"error": "No finite treatment values to rank."}

    # ranking & percentile (0–100). If higher_is_better=False, invert with ascending=True.
    # rank: 1 = best
    g["rank"] = g["value"].rank(ascending=not higher_is_better, method="min").astype(int)
    g["percentile"] = (g["value"].rank(pct=True, ascending=not higher_is_better) * 100.0)
    g["percentile"] = g["percentile"].round(1)
    g["value"] = g["value"].round(decimals)

    # sort for readability
    g = g.sort_values("value", ascending=not higher_is_better).reset_index(drop=True)
    n_treatments = int(len(g))

    result: Dict[str, Any] = {
        "index_name": idx,
        "date": date_used,
        "metric": metric,
        "n_treatments": n_treatments,
        "table": g.to_dict(orient="records"),
    }

    if trt_id is not None:
        trt_str = str(trt_id)
        row = g[g["TRT_ID"] == trt_str]
        if row.empty:
            return {
                **result,
                "warning": f"Treatment '{trt_str}' not present on {date_used}.",
            }
        row = row.iloc[0]
        result.update({
            "trt_id": trt_str,
            "value": float(row["value"]),
            "rank": int(row["rank"]),
            "percentile": float(row["percentile"]),
        })

    return result



def _json_default(o):
    import numpy as _np
    import pandas as _pd
    import datetime as _dt
    if isinstance(o, (_np.integer,)): return int(o)
    if isinstance(o, (_np.floating,)): return float(o)
    if isinstance(o, (_np.bool_,)): return bool(o)
    if isinstance(o, (_pd.Timestamp, _dt.datetime, _dt.date)): return o.isoformat()
    if o is _pd.NaT: return None
    return str(o)


def _safe_div(num, den, default=0.0):
    try:
        if den and np.isfinite(den) and float(den) != 0.0:
            return float(num) / float(den)
    except Exception:
        pass
    return default


def _safe_pct(num, den) -> float:
    return round(_safe_div(num, den, 0.0) * 100.0, 1)


def _index_dates_for(idx_name: str) -> list[datetime.date]:
    """Return a sorted list of valid dates for an index (safe for selectboxes)."""
    ds = master_zonal_stats[master_zonal_stats['Index'] == idx_name]['Date'].unique()
    ds = pd.to_datetime(ds, errors="coerce")
    ds = ds.dropna()
    return sorted(d.date() for d in ds)


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


# ============================================================
# DATA PATHS & LOADS
# ============================================================
image_dir = DEFAULT_IMAGE_DIR
shp_path = DEFAULT_SHP_PATH
mgmt_xlsx = DEFAULT_MGMT_XLSX
mgmt_sheet = DEFAULT_MGMT_N_SHEET

catalog = build_image_catalog(image_dir)
if not catalog:
    st.error("❌ No .tif files found. Expected format: NDVI_YYYY-MM-DD.tif")
    st.stop()

plots = read_plots(shp_path)
plot_to_trt = dict(zip(plots["Plot_ID"], plots["TRT_ID"]))

master_zonal_stats = calculate_all_zonal_stats(catalog, plots)

# Consistent dates across frames
if not master_zonal_stats.empty:
    master_zonal_stats['Date'] = pd.to_datetime(master_zonal_stats['Date']).dt.date

N_df = load_management_n(mgmt_xlsx, mgmt_sheet)
if not N_df.empty:
    N_df = N_df.groupby(["TRT_ID", "Date"], as_index=False)["Amount"].sum()
    N_df['Date'] = pd.to_datetime(N_df['Date']).dt.date

ghg_df = load_ghg_data(GHG_FILE_PATH)
if not ghg_df.empty:
    ghg_df['Date'] = pd.to_datetime(ghg_df['Date']).dt.date

# Deterministic TRT colors
rng = random.Random(42)
trt_colors = {trt: f"#{rng.randrange(0x100000, 0xFFFFFF):06x}" for trt in plots["TRT_ID"].unique()}


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Dashboard Controls")

st.sidebar.markdown("### Map Options")
index_name = st.sidebar.radio("Index:", sorted(catalog.keys()), horizontal=True).upper()
dates = sorted(catalog[index_name].keys())
sel_date = st.sidebar.selectbox("Date:", dates, index=len(dates) - 1)
opacity = st.sidebar.slider("Raster opacity", 0.1, 1.0, 0.95, 0.05)

st.sidebar.markdown("### Stress Thresholds")
ndvi_stress_threshold = st.sidebar.slider("NDVI threshold", 0.0, 1.0, DEFAULT_NDVI_STRESS, 0.01)
mcari_stress_threshold = st.sidebar.slider("MCARI2 threshold", 0.0, 1.0, DEFAULT_MCARI_STRESS, 0.01)
stress_threshold = ndvi_stress_threshold if index_name == "NDVI" else mcari_stress_threshold

basemap = st.sidebar.selectbox(
    "Basemap",
    ["Esri.WorldImagery (Satellite)", "CartoDB Positron", "Stamen Terrain"],
    index=0
)

# Value ranges
if index_name == "NDVI":
    vmin, vmax = NDVI_MIN, NDVI_MAX
else:
    vmin, vmax = MCARI_MIN, MCARI_MAX


# ============================================================
# CURRENT DATE PROCESSING (reuse precomputed stats)
# ============================================================
img_path = catalog[index_name][sel_date]

current_stats = master_zonal_stats[
    (master_zonal_stats['Index'] == index_name) &
    (master_zonal_stats['Date'] == sel_date)
].copy()

if current_stats.empty:
    st.error("No zonal stats available for selected date/index.")
    st.stop()

plots_w84 = plots.to_crs("EPSG:4326").copy()
plots_w84 = plots_w84.merge(
    current_stats[['Plot_ID', 'TRT_ID', 'Mean']].rename(columns={'Mean': 'Index_Mean'}),
    on=['Plot_ID', 'TRT_ID'],
    how='left'
)
plots_w84["color"] = plots_w84["Index_Mean"].apply(lambda v: value_to_css(v, vmin, vmax))
plot_stats_df = plots_w84[['Plot_ID', 'TRT_ID', 'Index_Mean', 'geometry', 'color']].copy()

# Prepare RGB(A) overlay for the raster (for visual map layer only)
with rasterio.open(img_path) as src:
    arr = src.read(1).astype("float32")
    arr[~np.isfinite(arr)] = np.nan

    # Reproject to WGS84 for display
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


# ============================================================
# MAIN PAGE
# ============================================================
st.title("🌾 AI-Enhanced Crop Health Dashboard")
st.caption(f"Index: **{index_name}** | Date: **{sel_date}** | Plots: **{len(plots_w84)}**")


# ============================================================
# QUICK INSIGHTS
# ============================================================
st.markdown("---")
st.markdown("### ⚡ Quick Insights")

if not current_stats.empty:
    insight_cols = st.columns(4)

    # Insight 1: Stressed Plots
    stressed_count = int((current_stats['Mean'] < stress_threshold).sum())
    total_now = int(len(current_stats))
    with insight_cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "🚨 Stressed Plots",
            f"{stressed_count}/{total_now}",
            delta=f"{_safe_pct(stressed_count, total_now):.1f}%",
            delta_color="inverse"
        )
        if stressed_count > 0:
            worst_plot = current_stats.nsmallest(1, 'Mean').iloc[0]
            st.caption(f"Worst: Plot {worst_plot['Plot_ID']} ({worst_plot['Mean']:.3f})")
        st.markdown('</div>', unsafe_allow_html=True)

    # Insight 2: Best Treatment
    trt_means = current_stats.groupby('TRT_ID')['Mean'].mean().sort_values(ascending=False)
    with insight_cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if len(trt_means) > 0:
            st.metric("🏆 Top Treatment", f"TRT {trt_means.index[0]}", delta=f"{trt_means.iloc[0]:.3f} avg")
            if len(trt_means) > 1:
                st.caption(f"vs TRT {trt_means.index[1]}: +{(trt_means.iloc[0]-trt_means.iloc[1]):.3f}")
        else:
            st.metric("🏆 Top Treatment", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    # Insight 3: Variability (CV%)
    mu = float(current_stats['Mean'].mean())
    sd = float(current_stats['Mean'].std()) if np.isfinite(current_stats['Mean'].std()) else 0.0
    cv = sd / mu * 100.0 if (np.isfinite(mu) and mu != 0.0 and np.isfinite(sd)) else 0.0
    with insight_cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        label = "Excellent" if cv < 10 else ("Good" if cv < 20 else "Poor")
        st.metric("📊 Uniformity (CV%)", f"{cv:.1f}%", delta=label,
                  delta_color="normal" if cv < 20 else "inverse")
        st.caption(f"Std: {sd:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Insight 4: Recent Trend (avg across plots)
    if len(catalog[index_name]) > 1:
        prev_date = sorted(catalog[index_name].keys())[-2]
        prev_stats = master_zonal_stats[
            (master_zonal_stats['Index'] == index_name) &
            (master_zonal_stats['Date'] == prev_date)
        ]
        prev_mean = float(prev_stats['Mean'].mean()) if not prev_stats.empty else np.nan
        curr_mean = float(current_stats['Mean'].mean())
        change = ((curr_mean - prev_mean) / prev_mean) * 100.0 if (np.isfinite(prev_mean) and prev_mean != 0.0) else 0.0

        with insight_cols[3]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📈 Recent Trend", f"{change:+.1f}%", delta=f"vs {prev_date}",
                      delta_color="normal" if change > 0 else "inverse")
            st.caption(f"Current: {curr_mean:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# AI-POWERED CHAT INTERFACE
# ============================================================
st.markdown("---")
st.markdown("### 🤖 AI Data Assistant")

if client is None:
    st.info("💡 Configure your OpenAI key to enable the AI assistant.")
else:
    # Define available tools (schemas)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stressed_plots",
                "description": "Identify plots with index values below a stress threshold. Returns list of stressed plots with their values.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]},
                        "threshold": {"type": "number"},
                        "date": {"type": "string"},
                        "treatment": {"type": "string"},
                        "n_plots": {"type": "integer", "minimum": 1, "maximum": 100}
                    },
                    "required": ["index_name", "threshold"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_treatment_percentile",
                "description": "Rank treatments by index performance and calculate percentiles",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]},
                        "date": {"type": "string"},
                        "metric": {"type": "string", "enum": ["mean", "median", "max"]}
                    },
                    "required": ["index_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_temporal_trend",
                "description": "Calculate trend over time for specific plots or treatments",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_type": {"type": "string", "enum": ["plot", "treatment"]},
                        "entity_id": {"type": "string"},
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]},
                        "period": {"type": "string", "enum": ["full", "last_30_days", "last_week"]}
                    },
                    "required": ["entity_type", "entity_id", "index_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_nitrogen_response",
                "description": "Analyze how treatments responded to nitrogen application",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "treatment_ids": {"type": "array", "items": {"type": "string"}},
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]},
                        "days_after_application": {"type": "integer"}
                    },
                    "required": ["treatment_ids"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "estimate_water_needs",
                "description": "Calculate irrigation needs based on Kc and weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_type": {"type": "string", "enum": ["plot", "treatment"]},
                        "entity_ids": {"type": "array", "items": {"type": "string"}},
                        "eto_mm": {"type": "number"},
                        "rainfall_mm": {"type": "number"}
                    },
                    "required": ["entity_type", "entity_ids"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "identify_anomalies",
                "description": "Find plots with unusual index patterns (outliers)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]},
                        "date": {"type": "string"},
                        "threshold_stddev": {"type": "number"}
                    },
                    "required": ["index_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_ghg_correlation",
                "description": "Correlate N2O emissions with crop health metrics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "treatment_ids": {"type": "array", "items": {"type": "string"}},
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]}
                    },
                    "required": ["treatment_ids"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "cost_benefit_analysis",
                "description": "Calculate ROI and cost efficiency for treatments",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "treatment_ids": {"type": "array", "items": {"type": "string"}},
                        "include_nitrogen_cost": {"type": "boolean"},
                        "include_water_cost": {"type": "boolean"}
                    },
                    "required": ["treatment_ids"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_prescription_map",
                "description": "Create variable rate application recommendations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]},
                        "date": {"type": "string"},
                        "input_type": {"type": "string", "enum": ["nitrogen", "water", "both"]},
                        "target_uniformity": {"type": "number"}
                    },
                    "required": ["index_name", "input_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "forecast_stress_risk",
                "description": "Predict future stress risk based on trends and weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plot_ids": {"type": "array", "items": {"type": "string"}},
                        "days_ahead": {"type": "integer", "enum": [7, 14, 21]},
                        "weather_scenario": {"type": "string", "enum": ["dry", "normal", "wet"]},
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]}
                    },
                    "required": ["plot_ids", "days_ahead"]
                }
            }
        },
        # --- NEW TOOLS ----
        {
            "type": "function",
            "function": {
                "name": "spatial_uniformity_report",
                "description": "Quantify uniformity (CV, IQR) and list hotspots/coldspots for a date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]},
                        "date": {"type": "string", "description": "YYYY-MM-DD; omit for latest"},
                        "by": {"type": "string", "enum": ["treatment", "overall"]},
                        "hotspot_quantile": {"type": "number", "description": "0-1, default 0.10"}
                    },
                    "required": ["index_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delta_map",
                "description": "Compute changes between two dates (plot or treatment level).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]},
                        "date_from": {"type": "string", "description": "YYYY-MM-DD (earlier)"},
                        "date_to": {"type": "string", "description": "YYYY-MM-DD (later)"},
                        "by": {"type": "string", "enum": ["plot", "treatment"]}
                    },
                    "required": ["index_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "nitrogen_response_curve",
                "description": "For a TRT, compute NDVI response per N application at 7/14/21d; fit slope and R².",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trt_id": {"type": "string"},
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]},
                        "horizons": {"type": "array", "items": {"type": "integer"}}
                    },
                    "required": ["trt_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "stress_diagnostics",
                "description": "Heuristic diagnosis of stress cause per plot (water vs nitrogen vs anomaly).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_name": {"type": "string", "enum": ["NDVI", "MCARI2"]},
                        "date": {"type": "string"},
                        "threshold": {"type": "number"},
                        "eto_mm": {"type": "number", "description": "Weekly ETo in mm"},
                        "rainfall_mm": {"type": "number", "description": "Weekly rainfall in mm"},
                        "nitrogen_gap_pct": {"type": "number", "description": "0-1; default 0.25"}
                    },
                    "required": ["eto_mm", "rainfall_mm"]
                }
            }
        }
        # --- END NEW TOOLS ---
    ]

    def execute_tool(tool_name, tool_args):
        try:
            if tool_name == "get_stressed_plots":
                return ai_tools.get_stressed_plots(master_zonal_stats, **tool_args)
            elif tool_name == "calculate_treatment_percentile":
                return ai_tools.calculate_treatment_percentile(master_zonal_stats, **tool_args)
            elif tool_name == "analyze_temporal_trend":
                return ai_tools.analyze_temporal_trend(master_zonal_stats, **tool_args)
            elif tool_name == "compare_nitrogen_response":
                return ai_tools.compare_nitrogen_response(master_zonal_stats, N_df, **tool_args)
            elif tool_name == "estimate_water_needs":
                return ai_tools.estimate_water_needs(master_zonal_stats, **tool_args)
            elif tool_name == "identify_anomalies":
                return ai_tools.identify_anomalies(master_zonal_stats, **tool_args)
            elif tool_name == "analyze_ghg_correlation":
                return ai_tools.analyze_ghg_correlation(master_zonal_stats, ghg_df, **tool_args)
            elif tool_name == "cost_benefit_analysis":
                return ai_tools.cost_benefit_analysis(master_zonal_stats, N_df, **tool_args)
            elif tool_name == "generate_prescription_map":
                return ai_tools.generate_prescription_map(master_zonal_stats, **tool_args)
            elif tool_name == "forecast_stress_risk":
                return ai_tools.forecast_stress_risk(master_zonal_stats, **tool_args)
            elif tool_name == "spatial_uniformity_report":
                return ai_tools.spatial_uniformity_report(master_zonal_stats, **tool_args)
            elif tool_name == "delta_map":
                return ai_tools.delta_map(master_zonal_stats, **tool_args)
            elif tool_name == "nitrogen_response_curve":
                return ai_tools.nitrogen_response_curve(master_zonal_stats, N_df, **tool_args)
            elif tool_name == "stress_diagnostics":
                return ai_tools.stress_diagnostics(master_zonal_stats, N_df, **tool_args)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    # 2024-aware example questions
    example_questions = [
        "Which plots were most stressed during the 2024 season?",
        "Rank treatments by uniformity (CV) on 2024-08-15 and list coldspots.",
        "Show largest NDVI declines between 2024-07-09 and 2024-07-25 at plot level.",
        "Compute the nitrogen response curve for TRT 34 at 7, 14, and 21 days.",
        "Diagnose likely stress drivers on 2024-08-15 with ETo=42 and rainfall=6.",
        "Identify treatments with NDVI below 0.5 on 2024-07-25.",
        "Compare NDVI and MCARI2 trends between June and August 2024.",
        "Which treatment maintained the highest crop vigor across all 2024 dates?",
        "How did NDVI respond to nitrogen applications for TRT 32 and TRT 34 in 2024?",
        "Find any plots showing abnormal index drops after July 2024.",
        "Generate a nitrogen prescription map based on August 2024 imagery.",
        "Analyze the correlation between N₂O flux and NDVI for 2024.",
        "Estimate irrigation requirements for stressed plots during July 2024.",
        "Forecast stress risk under a dry scenario for late August 2024."
    ]

    st.markdown('<div class="ai-chat-container">', unsafe_allow_html=True)
    with st.expander("💡 Example Questions", expanded=False):
        for i, q in enumerate(example_questions, 1):
            st.markdown(f"{i}. *{q}*")

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []

    # Display history
    for msg in st.session_state.ai_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask about your crop data..."):
        st.session_state.ai_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # -----------------------------
        # Build system prompt + messages
        # -----------------------------
        system_message = f"""You are an expert agronomist analyzing crop health data from a precision agriculture trial.

Current Context:
- Date: {sel_date}
- Index: {index_name}
- Total Plots: {len(plots)}
- Treatments: {', '.join(sorted(plots['TRT_ID'].unique()))}
- Available Data: NDVI, MCARI2, Nitrogen applications, Water needs, N2O emissions
- Date Range: {master_zonal_stats['Date'].min()} to {master_zonal_stats['Date'].max()}

When answering questions:
1. Use the available tools to query specific data
2. Provide actionable insights and recommendations
3. Cite specific numbers and plot/treatment IDs
4. Be concise but thorough
5. If data is missing, say so clearly
"""

        def _sanitize_history(msgs):
            safe = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                if content is None:
                    continue
                if not isinstance(content, str):
                    try:
                        content = json.dumps(content, default=_json_default)
                    except Exception:
                        content = str(content)
                if content.strip() == "":
                    continue
                safe.append({"role": role, "content": content})
            return safe

        history = _sanitize_history(st.session_state.ai_messages)
        messages = [{"role": "system", "content": system_message}, *history]

        # ---------------
        # Chat completion
        # ---------------
        with st.spinner("🤖 Analyzing data..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=2000
                )

                max_iterations = 5
                iteration = 0

                # Tool-call loop
                while response.choices[0].finish_reason == "tool_calls" and iteration < max_iterations:
                    iteration += 1
                    assistant_message = response.choices[0].message
                    assistant_content = assistant_message.content or ""

                    messages.append({
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in (assistant_message.tool_calls or [])
                        ]
                    })

                    # Execute tool calls
                    for tool_call in (assistant_message.tool_calls or []):
                        function_name = tool_call.function.name
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                        except Exception:
                            function_args = {}

                        tool_result = execute_tool(function_name, function_args)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result, default=_json_default)
                        })

                    # Next turn after tools
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=2000
                    )

                # Final answer
                final_answer = response.choices[0].message.content or "(no text returned)"
                st.session_state.ai_messages.append({"role": "assistant", "content": final_answer})

                with st.chat_message("assistant"):
                    st.write(final_answer)

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.ai_messages.append({"role": "assistant", "content": error_msg})

    st.markdown('</div>', unsafe_allow_html=True)

    # Clear chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.ai_messages = []
        st.rerun()


# ============================================================
# SQL-LIKE QUERY INTERFACE
# ============================================================
st.markdown("---")
st.markdown("### 🔍 Advanced Data Query")

with st.expander("📊 SQL-Style Query Builder", expanded=False):
    query_col1, query_col2, query_col3 = st.columns(3)

    with query_col1:
        query_index = st.selectbox("Select Index:", ["NDVI", "MCARI2"])
        query_entity = st.radio("Query by:", ["Plot", "Treatment"])

    with query_col2:
        query_operator = st.selectbox("Condition:", ["<", "<=", ">", ">=", "=", "between"])
        if query_operator == "between":
            query_val1 = st.number_input("Min value:", value=0.0, step=0.01)
            query_val2 = st.number_input("Max value:", value=1.0, step=0.01)
        else:
            query_val = st.number_input("Threshold:", value=0.5, step=0.01)

    with query_col3:
        query_date_option = st.radio("Date filter:", ["Latest", "Specific", "Date range"])
        available_dates = _index_dates_for(query_index)
        if query_date_option == "Specific":
            if available_dates:
                query_date = st.selectbox("Select date:", available_dates, index=len(available_dates) - 1)
            else:
                st.info("No dates available for this index.")
                query_date = None
        elif query_date_option == "Date range":
            if len(available_dates) >= 2:
                query_date_start = st.selectbox("From:", available_dates, index=0)
                query_date_end = st.selectbox("To:", available_dates, index=len(available_dates) - 1)
            else:
                st.info("Need at least two dates for a range.")
                query_date_start = query_date_end = None

    if st.button("🔍 Execute Query"):
        query_df = master_zonal_stats[master_zonal_stats['Index'] == query_index].copy()

        if query_date_option == "Latest":
            query_df = query_df[query_df['Date'] == query_df['Date'].max()]
        elif query_date_option == "Specific" and query_date is not None:
            query_df = query_df[query_df['Date'] == query_date]
        elif query_date_option == "Date range" and (query_date_start is not None and query_date_end is not None):
            query_df = query_df[(query_df['Date'] >= query_date_start) & (query_df['Date'] <= query_date_end)]
        else:
            st.warning("No valid date filter selected.")
            query_df = query_df.iloc[0:0]

        if query_operator == "<":
            query_df = query_df[query_df['Mean'] < query_val]
        elif query_operator == "<=":
            query_df = query_df[query_df['Mean'] <= query_val]
        elif query_operator == ">":
            query_df = query_df[query_df['Mean'] > query_val]
        elif query_operator == ">=":
            query_df = query_df[query_df['Mean'] >= query_val]
        elif query_operator == "=":
            query_df = query_df[query_df['Mean'].round(2) == round(query_val, 2)]
        else:
            query_df = query_df[(query_df['Mean'] >= query_val1) & (query_df['Mean'] <= query_val2)]

        if query_entity == "Treatment":
            result_df = query_df.groupby('TRT_ID').agg({
                'Mean': ['mean', 'count', 'min', 'max'],
                'Date': ['min', 'max']
            }).round(4)
            result_df.columns = ['_'.join(col).strip() for col in result_df.columns.values]
        else:
            result_df = query_df[['Plot_ID', 'TRT_ID', 'Date', 'Mean']].sort_values('Mean')

        st.success(f"✅ Found {len(result_df)} matching records")
        st.dataframe(result_df, use_container_width=True)

        csv_query = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Query Results",
            csv_query,
            f"query_results_{query_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )


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

def _center_from_bounds(bounds, gdf_fallback):
    """Return (lat, lon) center from [L, B, R, T] or fallback to gdf bounds."""
    try:
        L, B, R, T = bounds
        if np.all(np.isfinite([L, B, R, T])):
            return ((T + B) / 2.0, (L + R) / 2.0)
    except Exception:
        pass
    # Fallback: center of vector layer bounds
    minx, miny, maxx, maxy = gdf_fallback.geometry.total_bounds
    return ((miny + maxy) / 2.0, (minx + maxx) / 2.0)

center_lat, center_lon = _center_from_bounds((L2, B2, R2, T2), plots_w84)

# Fallback to polygon bounds if NaN
if not np.isfinite(center_lat) or not np.isfinite(center_lon):
    b = plots_w84.geometry.total_bounds
    center_lon = (b[0] + b[2]) / 2.0
    center_lat = (b[1] + b[3]) / 2.0

m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles=tiles, attr=attr)

# Raster overlay
rgba84 = colorize_ryg(arr84, vmin=vmin, vmax=vmax, alpha=int(opacity * 255))
folium.raster_layers.ImageOverlay(
    image=rgba84,
    bounds=[[B2, L2], [T2, R2]],
    opacity=1.0
).add_to(m)

# Enhanced polygon layer with click popup
def style_fn(feature):
    val = feature["properties"].get("Index_Mean", np.nan)
    outline = "#C62828" if (np.isfinite(val) and val < stress_threshold) else "#222222"
    return {"fillColor": feature["properties"]["color"], "color": outline, "weight": 2, "fillOpacity": 0.6}

# Centroids in planar CRS for stable markers
plots_planar = plots_w84.to_crs(3857)
centroids_planar = plots_planar.geometry.centroid
centroids_wgs84 = gpd.GeoSeries(centroids_planar, crs=3857).to_crs(4326)

for i, row in plots_w84.iterrows():
    c = centroids_wgs84.iloc[i]
    lat, lon = c.y, c.x
    plot_id = row['Plot_ID']
    trt_id = row['TRT_ID']
    mean_val = row['Index_Mean']

    plot_history = master_zonal_stats[
        (master_zonal_stats['Plot_ID'] == plot_id) &
        (master_zonal_stats['Index'] == index_name)
    ].sort_values('Date').tail(5)

    history_html = "<br>".join([f"{d}: {v:.3f}" for d, v in zip(plot_history['Date'], plot_history['Mean'])])

    popup_html = f"""
    <div style='width:200px'>
        <h4>Plot {plot_id}</h4>
        <b>TRT:</b> {trt_id}<br>
        <b>Current {index_name}:</b> {mean_val if pd.notna(mean_val) else 'NA'}<br>
        <hr><b>Recent History:</b><br><small>{history_html}</small>
    </div>
    """
    folium.CircleMarker(
        location=[lat, lon],
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
st.sidebar.markdown(
    "<div class='legend-gradient' style=\"background: linear-gradient(90deg, rgb(255,0,0), rgb(255,255,0), rgb(0,255,0));\"></div>",
    unsafe_allow_html=True
)
minmax_cols = st.sidebar.columns(2)
minmax_cols[0].markdown(f"<span class='small-muted'>{vmin:.2f}</span>", unsafe_allow_html=True)
minmax_cols[1].markdown(f"<div style='text-align:right' class='small-muted'>{vmax:.2f}</div>", unsafe_allow_html=True)


# ============================================================
# SUMMARY STATISTICS
# ============================================================
st.markdown("---")
st.markdown(f"## 📊 {index_name} Summary for {sel_date}")

if not current_stats.empty:
    overall_mean = float(current_stats['Mean'].mean())
    overall_min = float(current_stats['Mean'].min())
    overall_max = float(current_stats['Mean'].max())
    total_plots = int(len(current_stats))
    stressed_plots = int((current_stats['Mean'] < stress_threshold).sum())

    trt_means = current_stats.groupby('TRT_ID')['Mean'].mean().sort_values(ascending=False)
    best_trt = trt_means.index[0] if len(trt_means) > 0 else "N/A"
    best_trt_mean = float(trt_means.iloc[0]) if len(trt_means) > 0 else 0.0

    index_dates = sorted(catalog[index_name].keys())
    if len(index_dates) > 1:
        first_date = index_dates[0]
        last_date = index_dates[-1]
        first_mean = float(master_zonal_stats[
            (master_zonal_stats['Index'] == index_name) &
            (master_zonal_stats['Date'] == first_date)
        ]['Mean'].mean())
        last_mean = float(master_zonal_stats[
            (master_zonal_stats['Index'] == index_name) &
            (master_zonal_stats['Date'] == last_date)
        ]['Mean'].mean())
        trend_pct = ((last_mean - first_mean) / first_mean) * 100.0 if np.isfinite(first_mean) and first_mean != 0.0 else 0.0
    else:
        trend_pct = 0.0

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
        st.metric("Best TRT", f"TRT {best_trt}", delta=f"mean: {best_trt_mean:.4f}", delta_color="off")
    with metric_cols[4]:
        st.metric("Total Plots", total_plots)

    st.markdown("### Treatment Comparison")
    trt_comparison = current_stats.groupby('TRT_ID').agg({'Mean': ['mean', 'count']}).round(4)
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
            fig_ts.add_trace(go.Scatter(x=sub["Date"], y=sub["Mean"], mode="lines+markers",
                                        name=f"TRT {sid}", line=dict(color=color, width=2.5)))
            if show_nitrogen and not N_df.empty:
                Nsub = N_df[N_df["TRT_ID"] == sid]
                if not Nsub.empty:
                    fig_ts.add_trace(go.Bar(
                        x=Nsub["Date"], y=Nsub["Amount"], name=f"N (TRT {sid})",
                        marker_color=color, opacity=0.35, yaxis="y2"
                    ))
    else:
        palette = {}
        for sid in sel_ids:
            sid = str(sid)
            sub = ts[ts["Plot_ID"] == sid].sort_values("Date")
            color = palette.setdefault(sid, f'#{random.randint(0, 0xFFFFFF):06x}')
            fig_ts.add_trace(go.Scatter(x=sub["Date"], y=sub["Mean"], mode="lines+markers",
                                        name=f"Plot {sid}", line=dict(color=color)))
            if show_nitrogen and not N_df.empty:
                trt_for_plot = plot_to_trt.get(sid)
                if trt_for_plot:
                    Nsub = N_df[N_df["TRT_ID"] == trt_for_plot]
                    if not Nsub.empty:
                        fig_ts.add_trace(go.Bar(
                            x=Nsub["Date"], y=Nsub["Amount"], name=f"N (TRT {trt_for_plot})",
                            marker_color=trt_colors.get(trt_for_plot, "#888"), opacity=0.35, yaxis="y2"
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
# AI ANALYTICS PANELS (No-Chat quick tools)
# ============================================================
st.markdown("---")
st.markdown("### 🧠 AI Analytics Panels")

# --- Uniformity & Hotspots ---
with st.expander("📌 Uniformity & Hotspots (by date)", expanded=False):
    colu1, colu2, colu3 = st.columns(3)
    with colu1:
        uni_index = st.selectbox("Index", ["NDVI", "MCARI2"], index=0, key="uni_idx")

    dates_uni = _index_dates_for(uni_index)
    if not dates_uni:
        st.info(f"No dates available for {uni_index}. Load imagery first.")
    else:
        with colu2:
            uni_date = st.selectbox("Date", dates_uni, index=len(dates_uni) - 1, key="uni_date")
        with colu3:
            uni_by = st.selectbox("By", ["treatment", "overall"], index=0, key="uni_by")

        uni_q = st.slider("Hotspot/Coldspot quantile", 0.05, 0.25, 0.10, 0.01, key="uni_q")

        if st.button("Run Uniformity Report", key="btn_uni"):
            res = ai_tools.spatial_uniformity_report(
                master_zonal_stats,
                index_name=uni_index,
                date=uni_date,
                by=uni_by,
                hotspot_quantile=uni_q
            )
            if 'error' in res:
                st.error(res['error'])
            else:
                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("CV (%)", f"{res['cv_percent']:.2f}")
                with m2: st.metric("IQR", f"{res['iqr']:.3f}")
                with m3: st.metric("Mean", f"{res['population_mean']:.3f}")
                with m4: st.metric("Plots", f"{res['total_plots']}")
                st.write("**Coldspots (lowest values)**")
                st.dataframe(pd.DataFrame(res['coldspots']), use_container_width=True)
                st.write("**Hotspots (highest values)**")
                st.dataframe(pd.DataFrame(res['hotspots']), use_container_width=True)
                if res.get('group_stats'):
                    st.write("**Per-Treatment Uniformity**")
                    st.dataframe(pd.DataFrame(res['group_stats']), use_container_width=True)

with st.expander("🏅 Treatment Percentile (by date)", expanded=False):
    tp1, tp2, tp3, tp4 = st.columns(4)
    with tp1:
        tp_index = st.selectbox("Index", ["NDVI", "MCARI2"], index=0, key="tp_idx")
    dates_tp = sorted(master_zonal_stats[master_zonal_stats["Index"]==tp_index]["Date"].unique())
    if dates_tp:
        with tp2:
            tp_date = st.selectbox("Date", dates_tp, index=len(dates_tp)-1, key="tp_date")
    else:
        tp_date = None
        st.info("No dates available for this index.")
    with tp3:
        tp_metric = st.selectbox("Metric", ["mean", "median", "max"], index=0, key="tp_metric")
    with tp4:
        tp_hib = st.checkbox("Higher is better", value=True, key="tp_hib")

    trts_all = sorted(master_zonal_stats["TRT_ID"].astype(str).unique())
    tp_trt = st.selectbox("Treatment (optional)", trts_all, index=0, key="tp_trt_opt")

    if st.button("Compute Percentiles", key="btn_tp"):
        res = calculate_treatment_percentile(
            master_zonal_stats,
            index_name=tp_index,
            date=tp_date,
            metric=tp_metric,
            trt_id=tp_trt,
            higher_is_better=tp_hib
        )
        if "error" in res:
            st.error(res["error"])
        else:
            if "warning" in res:
                st.warning(res["warning"])
            st.dataframe(pd.DataFrame(res["table"]), use_container_width=True)
            if "trt_id" in res:
                st.success(f"TRT {res['trt_id']} → value={res['value']}, "
                           f"rank={res['rank']}/{res['n_treatments']}, "
                           f"percentile={res['percentile']:.1f}")





# --- Change Detection ---
with st.expander("🔄 Change Detection (Δ Between Dates)", expanded=False):
    cold1, cold2, cold3 = st.columns(3)
    with cold1:
        d_index = st.selectbox("Index", ["NDVI", "MCARI2"], index=0, key="d_idx")

    dates_available = _index_dates_for(d_index)
    if len(dates_available) < 2:
        st.info(f"Need at least two dates for {d_index} to compute changes.")
    else:
        with cold2:
            d_from = st.selectbox("From (earlier)", dates_available, index=0, key="d_from")
        with cold3:
            d_to = st.selectbox("To (later)", dates_available, index=len(dates_available) - 1, key="d_to")

        by_level = st.radio("Compare by", ["plot", "treatment"], horizontal=True, key="d_by")

        if st.button("Compute Changes", key="btn_delta"):
            res = ai_tools.delta_map(
                master_zonal_stats,
                index_name=d_index,
                date_from=d_from,
                date_to=d_to,
                by=by_level
            )
            if 'error' in res:
                st.error(res['error'])
            else:
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Avg Δ", f"{res['avg_delta']:.3f}")
                with c2: st.metric("Std Δ", f"{res['std_delta']:.3f}")
                with c3: st.metric("% Improved", f"{res['pct_improved']:.1f}%")
                with c4: st.metric("% Declined", f"{res['pct_declined']:.1f}%")
                df_changes = pd.DataFrame(res['changes'])
                st.dataframe(df_changes, use_container_width=True)
                st.download_button(
                    "📥 Download Δ table (CSV)",
                    df_changes.to_csv(index=False).encode('utf-8'),
                    file_name=f"delta_{d_index}_{d_from}_to_{d_to}.csv",
                    mime="text/csv"
                )

# --- Nitrogen Response Curve ---
with st.expander("🌿 Nitrogen Response Curve (per TRT)", expanded=False):
    if N_df.empty:
        st.info("No nitrogen data loaded.")
    else:
        trts = sorted(N_df['TRT_ID'].astype(str).unique())
        nr1, nr2, nr3 = st.columns(3)
        with nr1:
            nr_trt = st.selectbox("Treatment", trts, key="nr_trt")
        with nr2:
            nr_index = st.selectbox("Index", ["NDVI", "MCARI2"], index=0, key="nr_idx")
        with nr3:
            nr_horiz = st.multiselect("Horizons (days)", [7, 14, 21], default=[7, 14, 21], key="nr_hz")

        if st.button("Compute Response", key="btn_nr"):
            res = ai_tools.nitrogen_response_curve(
                master_zonal_stats, N_df,
                trt_id=nr_trt, index_name=nr_index, horizons=nr_horiz
            )
            if 'error' in res:
                st.error(res['error'])
            else:
                st.write(f"**Applications:** {res['n_applications']} | **Horizons:** {res['horizons_days']}")
                apps_df = pd.json_normalize(res['per_application'])
                st.dataframe(apps_df, use_container_width=True)
                fits_df = pd.DataFrame(
                    [(h, v['slope_delta_per_lb'], v['r_squared'], v['p_value'], v['n_points'])
                     for h, v in res['fits'].items()],
                    columns=['horizon_days', 'slope_delta_per_lb', 'r_squared', 'p_value', 'n_points']
                )
                st.write("**Fitted Δ ~ dose (per horizon)**")
                st.dataframe(fits_df, use_container_width=True)



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
# FOOTER
# ============================================================
st.markdown("---")
st.caption(f"AI-Enhanced Crop Dashboard | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total Plots: {len(plots)}")
