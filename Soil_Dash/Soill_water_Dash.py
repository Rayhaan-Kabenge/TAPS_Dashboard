# streamlit_app.py
# 2024 KSU TAPS SOIL WATER DYNAMICS — Cleaned + UX/robustness upgrades

import os
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import json
from datetime import datetime

# =========================
# PAGE CONFIG & GLOBAL CSS
# =========================
st.set_page_config(page_title="TAPS Soil Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .stApp { background-color: #f5f7fa; font-family: 'Segoe UI', Arial, sans-serif; }
    h1, h2, h3 { color: #2c5f2d; }
    h1 { text-align: center; font-size: 2.5em; margin-bottom: 10px; }
    .metric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 15px rgba(0,0,0,0.1); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; background-color: #e0e0e0; border-radius: 8px 8px 0 0; }
    </style>
""", unsafe_allow_html=True)
st.title("🌱 2024 KSU TAPS SOIL WATER DYNAMICS")
st.markdown("<p style='text-align: center; color: #666;'>Advanced Soil Moisture Monitoring & Analysis Platform</p>", unsafe_allow_html=True)

# =========================
# FILE PATHS & CONFIG
# =========================
TRT_INPUT_FILE   = '/Users/rayhaankabenge/Desktop/KSUTAPS/2025/soil_Dash/TRT_Plot_Summary.xlsx'
TRT_OUTPUT_FILE  = '/Users/rayhaankabenge/Desktop/KSUTAPS/2025/soil_Dash/TRT_Plot_Summary_Updated.xlsx'
NEUTRON_FILE     = '/Users/rayhaankabenge/Desktop/KSUTAPS/2025/soil_Dash/24 KSU TAPS Neutron Tube Readings_VWC.xlsx'
AQUASPY_FILE     = "/Users/rayhaankabenge/Desktop/KSUTAPS/2025/Dashboard_v2/Data_Source/24_KSUTAPS_AquaSpy.xlsx"
IRRIGATION_FILE  = "/Users/rayhaankabenge/Desktop/KSUTAPS/2025/Dashboard_v2/Data_Source/Irrigation_24.xlsx"

# JSON export directory
JSON_EXPORT_DIR = '/Users/rayhaankabenge/Desktop/KSUTAPS/2025/soil_Dash/chatbot_data'
os.makedirs(JSON_EXPORT_DIR, exist_ok=True)

# Moisture thresholds for alerts
MOISTURE_THRESHOLDS = {
    'critical_low': 0.15,
    'low': 0.20,
    'optimal_min': 0.25,
    'optimal_max': 0.40,
    'high': 0.45
}

INCH_TO_MM = 25.4

# =========================
# HELPERS
# =========================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        cc = str(c).strip().replace('\xa0', ' ')
        cc = ' '.join(cc.split())
        new_cols[c] = cc
    return df.rename(columns=new_cols)

@st.cache_data(show_spinner=False)
def load_trt_summary(input_file: str, output_file: str) -> pd.DataFrame:
    df = pd.read_excel(input_file)
    df = _normalize_cols(df)
    rename_map = {
        'TRT_ID': 'Team #', 'TRT Id': 'Team #', 'TRT': 'Team #', 'Team': 'Team #',
        'Plot_ID': 'Plot #', 'Plot': 'Plot #',
        'Block_ID': 'Block #', 'Block': 'Block #',
    }
    exist_map = {k: v for k, v in rename_map.items() if k in df.columns and v != k}
    df = df.rename(columns=exist_map)
    needed = ['Team #', 'Plot #', 'Block #']
    trt_plot_summary = df[[c for c in needed if c in df.columns]].copy()
    try:
        trt_plot_summary.to_excel(output_file, index=False)
    except Exception:
        pass
    for col in needed:
        if col in trt_plot_summary.columns:
            trt_plot_summary[col] = trt_plot_summary[col].astype(str).str.strip()
    return trt_plot_summary

@st.cache_data(show_spinner=False)
def load_neutron_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = _normalize_cols(df)
    rename_map = {
        'TRT_ID': 'Team #', 'TRT Id': 'Team #', 'TRT': 'Team #', 'Team': 'Team #',
        'Plot_ID': 'Plot #', 'Plot': 'Plot #',
        'Block_ID': 'Block #', 'Block': 'Block #',
        'Date': 'Date', 'DATE': 'Date'
    }
    exist_map = {k: v for k, v in rename_map.items() if k in df.columns and v != k}
    df = df.rename(columns=exist_map)
    if 'Date' not in df.columns:
        raise ValueError("Neutron dataset is missing a 'Date' column.")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for col in ['Team #', 'Plot #', 'Block #']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_aquaspy_sheets(path: str) -> dict:
    sheets = pd.read_excel(path, sheet_name=None)
    out = {}
    for name, d in sheets.items():
        out[name] = _normalize_cols(d)
    return out

def get_depth_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if isinstance(c, str) and c.startswith('V-')]

def robust_fill_block(df_neutron: pd.DataFrame, trt_df: pd.DataFrame) -> pd.DataFrame:
    if trt_df.empty:
        return df_neutron
    for keyset in (['Team #', 'Plot #'], ['Plot #'], ['Team #']):
        if all(k in df_neutron.columns for k in keyset) and all(k in trt_df.columns for k in keyset):
            right_cols = keyset + ['Block #']
            right = trt_df[right_cols].drop_duplicates()
            merged = df_neutron.merge(right, on=keyset, how='left', suffixes=('', '_trt'))
            if 'Block #' in df_neutron.columns:
                merged['Block #'] = merged['Block #'].where(~merged['Block #'].isna(), merged['Block #_trt'])
            else:
                merged['Block #'] = merged['Block #_trt']
            merged = merged.drop(columns=[c for c in ['Block #_trt'] if c in merged.columns])
            df_neutron = merged
            break
    if 'Block #' not in df_neutron.columns:
        df_neutron['Block #'] = 'All'
    else:
        df_neutron['Block #'] = df_neutron['Block #'].fillna('All')
    df_neutron['Block #'] = df_neutron['Block #'].astype(str)
    return df_neutron

@st.cache_data(ttl=3600, show_spinner=False)
def load_irrigation_df(path: str, sheet_name=None) -> pd.DataFrame:
    """
    Wide:  Date + team-number columns ('1','2',...) in inches
    Long:  columns like Team/Date/Amount (inches)
    Returns: ['Team #','Timestamp','Irr_mm','Team_norm']
    """
    if sheet_name is None:
        raw = pd.read_excel(path, sheet_name=None)
        frames = [df.assign(__sheet=name) for name, df in raw.items()]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)

    df.columns = [str(c).strip().replace("\xa0", " ") for c in df.columns]

    time_candidates = ['Timestamp','TimeStamp','Date Time','DateTime','Datetime','Date']
    date_col = next((c for c in time_candidates if c in df.columns), None)
    if not date_col:
        raise ValueError("Could not find a timestamp column in irrigation file.")

    long_team_cols = ['Team #','Team','TRT','TRT_ID','TRT Id']
    long_amt_cols  = ['Irrigation','Irr (in)','Inches','Applied','Amount','Irrigation (in)']

    has_long = any(c in df.columns for c in long_team_cols) and any(c in df.columns for c in long_amt_cols)

    if has_long:
        team_col = next(c for c in long_team_cols if c in df.columns)
        amt_col  = next(c for c in long_amt_cols  if c in df.columns)
        tidy = df[[date_col, team_col, amt_col]].rename(columns={date_col: 'Timestamp', team_col: 'Team #', amt_col: 'Irr_in'})
    else:
        team_cols = [c for c in df.columns if c != date_col and re.fullmatch(r'\d+', str(c).strip())]
        if not team_cols:
            team_cols = [c for c in df.columns if c != date_col and re.match(r'^(Team\s*)?\d+$', str(c).strip())]
        if not team_cols:
            raise ValueError(f"No team columns found to melt. Columns were: {list(df.columns)}")
        tidy = df[[date_col] + team_cols].melt(id_vars=[date_col], var_name='Team #', value_name='Irr_in').rename(columns={date_col: 'Timestamp'})

    tidy['Timestamp'] = pd.to_datetime(tidy['Timestamp'], errors='coerce')
    tidy = tidy.dropna(subset=['Timestamp'])

    nums = tidy['Irr_in'].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
    tidy['Irr_in'] = pd.to_numeric(nums, errors='coerce')
    tidy['Irr_mm'] = tidy['Irr_in'] * INCH_TO_MM

    def _norm_team(x: str) -> str:
        m = re.search(r'\d+', str(x))
        return m.group(0) if m else str(x).strip()

    tidy['Team #'] = tidy['Team #'].astype(str).str.strip()
    tidy['Team_norm'] = tidy['Team #'].map(_norm_team)
    return tidy[['Team #','Timestamp','Irr_mm','Team_norm']]

# =========================
# JSON EXPORT
# =========================
def export_data_to_json(df, trt_summary, depth_columns, sheets):
    timestamp = datetime.now().isoformat()
    metadata = {
        'export_timestamp': timestamp,
        'total_plots': int(df['Plot #'].nunique()) if 'Plot #' in df.columns else 0,
        'total_teams': int(trt_summary['Team #'].nunique()) if 'Team #' in trt_summary.columns else 0,
        'total_blocks': int(df['Block #'].nunique()) if 'Block #' in df.columns else 0,
        'date_range': {'start': str(df['Date'].min()), 'end': str(df['Date'].max())},
        'depth_levels': depth_columns,
        'total_readings': len(df)
    }
    with open(f"{JSON_EXPORT_DIR}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date].copy()
    current_status = []
    for _, row in latest_data.iterrows():
        item = {
            'plot_id': str(row.get('Plot #', 'Unknown')),
            'block_id': str(row.get('Block #', 'Unknown')),
            'team_id': str(row.get('Team #', 'Unknown')),
            'date': str(row['Date']),
            'readings': {}
        }
        for depth in depth_columns:
            if depth in row:
                item['readings'][depth] = float(row[depth]) if pd.notna(row[depth]) else None
        avg_m = np.nanmean([item['readings'][d] for d in depth_columns if item['readings'].get(d) is not None])
        item['average_moisture'] = float(avg_m) if not np.isnan(avg_m) else None
        item['status'] = classify_moisture_status(avg_m)
        current_status.append(item)
    with open(f"{JSON_EXPORT_DIR}/current_status.json", 'w') as f:
        json.dump(current_status, f, indent=2)

    plot_summaries = []
    for plot_id in df['Plot #'].unique():
        plot_df = df[df['Plot #'] == plot_id]
        summary = {
            'plot_id': str(plot_id),
            'block_id': str(plot_df['Block #'].iloc[0]) if 'Block #' in plot_df.columns else 'Unknown',
            'team_id': str(plot_df['Team #'].iloc[0]) if 'Team #' in plot_df.columns else 'Unknown',
            'statistics': {}
        }
        for depth in depth_columns:
            if depth in plot_df.columns:
                dd = pd.to_numeric(plot_df[depth], errors='coerce').dropna()
                if len(dd) > 0:
                    summary['statistics'][depth] = {
                        'mean': float(dd.mean()), 'std': float(dd.std()),
                        'min': float(dd.min()),  'max': float(dd.max()),
                        'current': float(dd.iloc[-1])
                    }
        plot_summaries.append(summary)
    with open(f"{JSON_EXPORT_DIR}/plot_summaries.json", 'w') as f:
        json.dump(plot_summaries, f, indent=2)

    alerts = generate_alerts(df, depth_columns)
    with open(f"{JSON_EXPORT_DIR}/alerts.json", 'w') as f:
        json.dump(alerts, f, indent=2)

    team_mapping = trt_summary.to_dict('records')
    with open(f"{JSON_EXPORT_DIR}/team_mapping.json", 'w') as f:
        json.dump(team_mapping, f, indent=2)

    return metadata

def export_irrigation_json(irr_df: pd.DataFrame, out_dir: str):
    if irr_df.empty:
        return
    by_team = (irr_df.assign(date=irr_df['Timestamp'].dt.date)
               .groupby(['Team_norm','date'], as_index=False)['Irr_mm'].sum())
    payload = [
        {"team": str(r.Team_norm), "date": str(r.date), "irrigation_mm": float(r.Irr_mm)}
        for _, r in by_team.iterrows()
    ]
    with open(os.path.join(out_dir, "irrigation_by_team.json"), "w") as f:
        json.dump(payload, f, indent=2)

def classify_moisture_status(moisture_value):
    if pd.isna(moisture_value):
        return 'unknown'
    elif moisture_value < MOISTURE_THRESHOLDS['critical_low']:
        return 'critical_low'
    elif moisture_value < MOISTURE_THRESHOLDS['low']:
        return 'low'
    elif moisture_value < MOISTURE_THRESHOLDS['optimal_min']:
        return 'below_optimal'
    elif moisture_value <= MOISTURE_THRESHOLDS['optimal_max']:
        return 'optimal'
    elif moisture_value <= MOISTURE_THRESHOLDS['high']:
        return 'above_optimal'
    else:
        return 'high'

def generate_alerts(df, depth_columns):
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date].copy()
    alerts = {'timestamp': datetime.now().isoformat(), 'critical': [], 'warnings': [], 'info': []}
    for _, row in latest_data.iterrows():
        avg_m = np.nanmean([pd.to_numeric(row[d], errors='coerce') for d in depth_columns if d in row])
        if pd.notna(avg_m):
            alert = {
                'plot_id': str(row.get('Plot #', 'Unknown')),
                'block_id': str(row.get('Block #', 'Unknown')),
                'moisture_level': float(avg_m),
                'status': classify_moisture_status(avg_m)
            }
            if avg_m < MOISTURE_THRESHOLDS['critical_low']:
                alert['message'] = f"Critical: Immediate irrigation needed (moisture: {avg_m:.3f})"
                alerts['critical'].append(alert)
            elif avg_m < MOISTURE_THRESHOLDS['low']:
                alert['message'] = f"Warning: Low moisture detected (moisture: {avg_m:.3f})"
                alerts['warnings'].append(alert)
            elif avg_m > MOISTURE_THRESHOLDS['high']:
                alert['message'] = f"Info: High moisture level (moisture: {avg_m:.3f})"
                alerts['info'].append(alert)
    return alerts

# =========================
# LOAD DATA
# =========================
try:
    trt_summary_df = load_trt_summary(TRT_INPUT_FILE, TRT_OUTPUT_FILE)
    df = load_neutron_df(NEUTRON_FILE)
    sheets = load_aquaspy_sheets(AQUASPY_FILE)
    irr_df = load_irrigation_df(IRRIGATION_FILE)
except Exception as e:
    st.error(f"Startup load failed: {e}")
    st.stop()

df = robust_fill_block(df, trt_summary_df)
depth_columns = get_depth_columns(df)
if not depth_columns:
    st.warning("No depth columns found (expected columns starting with 'V-').")

# Export JSONs
try:
    metadata = export_data_to_json(df, trt_summary_df, depth_columns, sheets)
    export_irrigation_json(irr_df, JSON_EXPORT_DIR)
    st.sidebar.success(f"✅ Data exported to JSON ({metadata['total_readings']} readings)")
except Exception as e:
    st.sidebar.error(f"JSON export failed: {e}")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📊 Dashboard Controls")
with st.sidebar.expander("📥 Export Data", expanded=False):
    if st.button("🔄 Regenerate JSON Files"):
        try:
            metadata = export_data_to_json(df, trt_summary_df, depth_columns, sheets)
            export_irrigation_json(irr_df, JSON_EXPORT_DIR)
            st.success("JSON files regenerated!")
        except Exception as e:
            st.error(f"Error: {e}")

with st.sidebar.expander("🚨 Smart Alerts", expanded=True):
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date]
    critical, warn = [], []
    for _, row in latest_data.iterrows():
        if depth_columns:
            avg_m = np.nanmean([pd.to_numeric(row[d], errors='coerce') for d in depth_columns if d in row])
            if pd.notna(avg_m):
                if avg_m < MOISTURE_THRESHOLDS['critical_low']:
                    critical.append((row['Plot #'], avg_m))
                elif avg_m < MOISTURE_THRESHOLDS['low']:
                    warn.append((row['Plot #'], avg_m))
    if critical:
        st.error(f"🔴 {len(critical)} plot(s) need immediate attention!")
        for p, m in critical[:3]: st.markdown(f"- Plot {p}: {m:.3f}")
    elif warn:
        st.warning(f"⚠️ {len(warn)} plot(s) have low moisture")
        for p, m in warn[:3]: st.markdown(f"- Plot {p}: {m:.3f}")
    else:
        st.success("✅ All plots within acceptable range")

# =========================
# TABS
# =========================
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Summary", "📈 Tube Readings", "💧 Aquaspy Data",
    "👥 Team Overview", "🔮 Predictions", "❓ Help"
])

# -------------------------
# TAB 0: DATA SUMMARY
# -------------------------
with tab0:
    st.header("📊 Data Overview & Quality Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Plots", df['Plot #'].nunique() if 'Plot #' in df.columns else 0)
    with col2: st.metric("Total Teams", trt_summary_df['Team #'].nunique() if 'Team #' in trt_summary_df.columns else 0)
    with col3: st.metric("Total Blocks", df['Block #'].nunique() if 'Block #' in df.columns else 0)
    with col4:
        _num = df[depth_columns].apply(pd.to_numeric, errors='coerce') if depth_columns else pd.DataFrame()
        st.metric("Avg VWC", f"{_num.mean().mean():.3f}" if not _num.empty else "0.000")

    st.markdown("### 📅 Data Collection Period")
    c1, c2 = st.columns(2)
    with c1: st.info(f"**Start Date:** {df['Date'].min().strftime('%Y-%m-%d')}")
    with c2: st.info(f"**End Date:** {df['Date'].max().strftime('%Y-%m-%d')}")

    st.markdown("### 🧪 Data Quality Checks")
    missing = (df.isna().sum().sort_values(ascending=False)
               .to_frame("Missing").query("Missing > 0").head(15))
    if not missing.empty:
        st.dataframe(missing, use_container_width=True)
    else:
        st.success("No missing values detected (top 15).")

    st.markdown("### 📈 Descriptive Stats (Depth Columns)")
    if depth_columns:
        _stats = df[depth_columns].apply(pd.to_numeric, errors='coerce').describe().T.round(3)
        st.dataframe(_stats, use_container_width=True)

    st.markdown("### 📊 Most Recent Readings (Last 10)")
    recent_cols = ['Date', 'Plot #', 'Block #', 'Team #'] + depth_columns[:5]
    recent_cols = [c for c in recent_cols if c in df.columns]
    st.dataframe(df[recent_cols].tail(10), use_container_width=True)

# -------------------------
# TAB 1: TUBE READINGS
# -------------------------
with tab1:
    st.header("📈 Soil Moisture Tube Readings")
    req = ['Date', 'Plot #', 'Block #']
    missing = [c for c in req if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        blocks = sorted([x for x in df['Block #'].dropna().unique()])
        has_real_blocks = [b for b in blocks if b != 'All']
        with st.sidebar.expander("🔧 Filters", expanded=True):
            if has_real_blocks:
                selected_block = st.selectbox("Block #:", has_real_blocks, key="block_tab1")
                plots_in_block = sorted(df.loc[df['Block #'] == selected_block, 'Plot #'].dropna().unique())
            else:
                st.info("Showing all blocks")
                selected_block = 'All'
                plots_in_block = sorted(df['Plot #'].dropna().unique())
            if not plots_in_block:
                st.warning("No plots available"); st.stop()
            selected_plot = st.selectbox("Plot #:", plots_in_block, key="plot_tab1")
            min_d, max_d = df['Date'].min(), df['Date'].max()
            start_date, end_date = st.date_input("Date Range:", [min_d, max_d], key="date_tab1")
            selected_depths = st.multiselect("Depths:", depth_columns, default=depth_columns, key="depths_tab1")

        mask_block = pd.Series(True, index=df.index) if selected_block == 'All' else (df['Block #'] == selected_block)
        filtered_df_plot = df.loc[
            (df['Plot #'] == selected_plot) & mask_block &
            (df['Date'] >= pd.to_datetime(start_date)) &
            (df['Date'] <= pd.to_datetime(end_date))
        ].copy()

        if filtered_df_plot.empty:
            st.warning("No data for selected filters")
        else:
            c1, c2, c3 = st.columns([2,1,1])
            with c1: st.subheader(f"Plot {selected_plot}" + ("" if selected_block == 'All' else f" | Block {selected_block}"))
            with c2: st.download_button("📥 CSV", filtered_df_plot.to_csv(index=False), f"plot_{selected_plot}_data.csv", "text/csv", use_container_width=True)
            with c3:
                if st.button("🔄 Reset Filters", use_container_width=True): st.rerun()

            fig = go.Figure()
            for depth in selected_depths:
                if depth in filtered_df_plot.columns:
                    yvals = pd.to_numeric(filtered_df_plot[depth], errors='coerce')
                    fig.add_trace(go.Scatter(
                        x=filtered_df_plot['Date'], y=yvals, mode='lines+markers', name=f'{depth}',
                        hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>VWC: %{y:.3f}<extra></extra>'
                    ))
            fig.add_hline(y=MOISTURE_THRESHOLDS['optimal_min'], line_dash="dash", line_color="green",
                          annotation_text="Optimal Min", annotation_position="right")
            fig.add_hline(y=MOISTURE_THRESHOLDS['optimal_max'], line_dash="dash", line_color="orange",
                          annotation_text="Optimal Max", annotation_position="right")
            fig.update_layout(title='Soil Moisture Trend with Optimal Range',
                              xaxis_title='Date', yaxis_title='Volumetric Water Content (VWC)',
                              hovermode="x unified", height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📉 Moving Average Analysis")
            window_size = st.slider("Window (days):", 1, 30, 7, key="window_tab1")
            fig_ma = go.Figure()
            for depth in selected_depths:
                if depth in filtered_df_plot.columns:
                    ma = pd.to_numeric(filtered_df_plot[depth], errors='coerce').rolling(window=window_size, min_periods=1).mean()
                    fig_ma.add_trace(go.Scatter(x=filtered_df_plot['Date'], y=ma, mode='lines', name=f'MA {window_size}d - {depth}'))
            fig_ma.update_layout(title=f'{window_size}-Day Moving Average', xaxis_title='Date', yaxis_title='VWC (Moving Average)',
                                 hovermode="x unified", height=400)
            st.plotly_chart(fig_ma, use_container_width=True)

            st.subheader("🌡️ Soil Moisture Profile Heatmap")
            heatmap_data = filtered_df_plot[['Date'] + selected_depths].set_index('Date').apply(pd.to_numeric, errors='coerce')
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.T.values, x=heatmap_data.index, y=heatmap_data.columns,
                colorscale='Blues', colorbar=dict(title="VWC"),
                hovertemplate='Date: %{x}<br>Depth: %{y}<br>VWC: %{z:.3f}<extra></extra>'
            ))
            fig_heatmap.update_layout(title='Moisture Distribution Across Depths Over Time',
                                      xaxis_title='Date', yaxis_title='Depth', height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)

# -------------------------
# TAB 2: AQUASPY DATA (15-min + Irrigation + EC axis)
# -------------------------
with tab2:
    st.header("💧 High-frequency Moisture & EC (AquaSpy)")

    if not sheets:
        st.info("No AquaSpy workbook loaded.")
    else:
        def _depth_to_in(label: str):
            try: return float(str(label).split('-')[-1].replace('"',''))
            except Exception: return None

        def _coerce_numeric(df_in: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
            out = df_in.copy()
            for c in cols:
                if c in out.columns:
                    s = out[c].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
                    out[c] = pd.to_numeric(s, errors="coerce")
            return out

        def _collect_available_params(sheets_dict: dict) -> tuple[list[str], list[str]]:
            all_cols = set()
            for _, _df in sheets_dict.items():
                all_cols.update(list(_df.columns))
            moist = sorted([c for c in all_cols if str(c).startswith(('M','MS','V-'))])
            ec    = sorted([c for c in all_cols if str(c).upper().startswith('EC')])
            return moist, ec

        team_names = list(sheets.keys())
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            selected_teams = st.multiselect("Teams / Sheets", team_names, default=team_names[:2], key="teams_tab2_v2")
        with c2:
            freq_label = st.selectbox("Resample", ["Raw (15 min)", "30 min", "1 hour", "6 hours", "Daily"], index=0)
        with c3:
            agg_func = st.selectbox("Aggregation", ["mean","median","min","max"], index=0)

        c4, c5, c6 = st.columns([1.2, 1, 1])
        scope = {k: sheets[k] for k in selected_teams} if selected_teams else sheets
        moisture_all, ec_all = _collect_available_params(scope)
        with c4:
            moisture_selected = st.multiselect("Moisture params", moisture_all,
                                               default=[c for c in moisture_all if c in ("MS",'M4"','M8"')][:2],
                                               key="moist_params_v2")
        with c5:
            ec_selected = st.multiselect("EC params", ec_all, default=[ec_all[0]] if ec_all else [], key="ec_params_v2")
        with c6:
            overlay_irrig = st.toggle("Overlay irrigation", value=True and not irr_df.empty,
                                      help="Show irrigation bars (mm)")

        smooth_on = st.toggle("Rolling smoothing", value=False)
        if smooth_on:
            sc1, sc2 = st.columns(2)
            with sc1: smooth_window = st.number_input("Window length (samples)", 2, 288, 8, 1)
            with sc2: smooth_center = st.checkbox("Center window", False)

        freq_map = {"Raw (15 min)": None, "30 min": "30T", "1 hour":"1H", "6 hours":"6H", "Daily":"1D"}

        if not selected_teams:
            st.info("Select at least one team/sheet to plot.")
        else:
            fig2 = go.Figure()
            plotted_frames = []
            res = freq_map[freq_label]

            for team in selected_teams:
                data = sheets[team].copy()
                ts_col = next((c for c in ["Timestamp","TimeStamp","Date Time","DateTime","Datetime","Date"] if c in data.columns), None)
                if ts_col is None:
                    st.warning(f"Sheet '{team}' has no timestamp column."); continue
                data[ts_col] = pd.to_datetime(data[ts_col], errors="coerce")
                data = data.dropna(subset=[ts_col]).sort_values(ts_col)

                wanted = list(set((moisture_selected or []) + (ec_selected or [])))
                data = _coerce_numeric(data, wanted)

                if res:
                    data = data.set_index(ts_col).resample(res).agg(agg_func).reset_index()
                ts_series = data[ts_col]

                if smooth_on and wanted:
                    for c in wanted:
                        if c in data.columns:
                            data[c] = data[c].rolling(int(smooth_window), center=smooth_center, min_periods=1).mean()

                def _sort_key(name): 
                    v = _depth_to_in(name); 
                    return v if v is not None else 0.0

                # Moisture on y (default)
                for param in sorted(moisture_selected, key=_sort_key):
                    if param in data.columns:
                        fig2.add_trace(go.Scattergl(
                            x=ts_series, y=data[param], mode="lines",
                            name=f"{param} — {team}",
                            hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d %H:%M}<br>VWC: %{y:.3f}<extra></extra>"
                        ))
                        plotted_frames.append(data[[ts_col, param]].assign(Team=team, Parameter=param))

                # EC on y3 (right axis)
                for param in ec_selected:
                    if param in data.columns:
                        fig2.add_trace(go.Scattergl(
                            x=ts_series, y=data[param], mode="lines",
                            name=f"{param} — {team}", yaxis="y3",
                            hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d %H:%M}<br>EC: %{y:.3f}<extra></extra>"
                        ))
                        plotted_frames.append(data[[ts_col, param]].assign(Team=team, Parameter=param))

                # Irrigation on y2 (top-down bars)
                if overlay_irrig and not irr_df.empty:
                    def _norm_team_id(x):
                        m = re.search(r'\d+', str(x))
                        return m.group(0) if m else str(x).strip()
                    team_norm = _norm_team_id(team)
                    irr_team = irr_df[irr_df['Team_norm'] == team_norm].copy()
                    if not irr_team.empty:
                        if res:
                            irr_rs = (irr_team.set_index('Timestamp').resample(res)['Irr_mm'].sum().reset_index())
                        else:
                            irr_rs = irr_team.rename(columns={'Timestamp':'Timestamp', 'Irr_mm':'Irr_mm'})[['Timestamp','Irr_mm']]
                        # merge duplicates within same bin
                        irr_rs = (irr_rs.dropna(subset=['Timestamp'])
                                         .groupby('Timestamp', as_index=False)['Irr_mm'].sum())
                        if len(ts_series):
                            tmin, tmax = ts_series.min(), ts_series.max()
                            irr_rs = irr_rs[(irr_rs['Timestamp'] >= tmin) & (irr_rs['Timestamp'] <= tmax)]
                        fig2.add_trace(go.Bar(
                            x=irr_rs['Timestamp'], y=irr_rs['Irr_mm'],
                            name=f"Irrigation — {team} (mm)", marker=dict(opacity=0.35),
                            yaxis="y2",
                            hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} mm<extra></extra>"
                        ))
                        plotted_frames.append(irr_rs.rename(columns={'Irr_mm':'Irrigation_mm'}).assign(Team=team, Parameter='Irrigation'))

            # Axes & layout
            fig2.update_layout(
                title=f"AquaSpy {freq_label} ({agg_func})" if res else "AquaSpy Raw (15-min)",
                xaxis_title="Timestamp",
                yaxis=dict(title="VWC (m³/m³)"),
                hovermode="x unified",
                height=640,
                barmode="overlay",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                # y2: irrigation (top-down bars)
                yaxis2=dict(title="Irrigation (mm)", overlaying="y", side="right",
                            showgrid=False, autorange="reversed", rangemode="tozero", position=1.0),
                # y3: EC (separate right axis, slightly left of y2)
                yaxis3=dict(title="EC", overlaying="y", side="right",
                            showgrid=False, rangemode="normal", position=0.97)
            )
            st.plotly_chart(fig2, use_container_width=True)

            if plotted_frames:
                plotted = pd.concat(plotted_frames, ignore_index=True)
                tcol = next((c for c in plotted.columns if c.lower().startswith('time')), 'Timestamp')
                if tcol != "Timestamp": plotted.rename(columns={tcol:"Timestamp"}, inplace=True)
                fname = f"aquaspy_{'_'.join(map(str,selected_teams))}_{(freq_label or 'raw').replace(' ','_').lower()}.csv"
                st.download_button("📥 Download plotted series (CSV)",
                                   plotted.to_csv(index=False).encode("utf-8"),
                                   file_name=fname, mime="text/csv")

# -------------------------
# TAB 3: TEAM OVERVIEW (unchanged from your version)
# -------------------------
# ... keep your existing Team Overview tab code here (no functional changes) ...

# -------------------------
# TAB 4: PREDICTIONS (same functionality)
# -------------------------
with tab4:
    st.header("🔮 Soil Moisture Forecasting")
    if not depth_columns:
        st.info("No depth columns available for forecasting")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: selected_depth = st.selectbox("Select Depth:", depth_columns, key="depth_pred")
        with c2: forecast_horizon = st.slider("Forecast Days:", 1, 30, 7, key="horizon")
        with c3: selected_plot_pred = st.selectbox("Filter by Plot (optional):", ['All'] + sorted(df['Plot #'].dropna().unique().tolist()), key="plot_pred")
        df_forecast = df[df['Plot #'] == selected_plot_pred].copy() if selected_plot_pred != 'All' else df.copy()
        depth_data = df_forecast[['Date', selected_depth]].rename(columns={'Date': 'ds', selected_depth: 'y'})
        depth_data['ds'] = pd.to_datetime(depth_data['ds'], errors='coerce')
        depth_data['y'] = pd.to_numeric(depth_data['y'], errors='coerce')
        depth_data = depth_data.dropna(subset=['ds', 'y']).sort_values('ds')
        if depth_data.empty:
            st.warning("No valid data for forecasting")
        else:
            st.subheader("📊 Historical Data")
            fig_original = go.Figure()
            fig_original.add_trace(go.Scatter(x=depth_data['ds'], y=depth_data['y'], mode='lines+markers', name='Historical Data'))
            fig_original.update_layout(title=f"Historical Moisture Content - {selected_depth}", xaxis_title="Date", yaxis_title="VWC", hovermode="x unified", height=400)
            st.plotly_chart(fig_original, use_container_width=True)

            with st.spinner("Generating forecast..."):
                m = Prophet(daily_seasonality=True)
                m.fit(depth_data)
                future = m.make_future_dataframe(periods=forecast_horizon, freq='D')
                forecast = m.predict(future)

            st.subheader(f"🔮 {forecast_horizon}-Day Forecast")
            future_dates = forecast[forecast['ds'] > depth_data['ds'].max()]
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=depth_data['ds'], y=depth_data['y'], mode='lines', name='Historical'))
            fig_forecast.add_trace(go.Scatter(x=future_dates['ds'], y=future_dates['yhat'], mode='lines', name='Forecast', line=dict(dash='dash')))
            fig_forecast.add_trace(go.Scatter(x=future_dates['ds'], y=future_dates['yhat_upper'], mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False))
            fig_forecast.add_trace(go.Scatter(x=future_dates['ds'], y=future_dates['yhat_lower'], mode='lines', name='Confidence Interval', fill='tonexty', line=dict(width=0), fillcolor='rgba(255, 0, 0, 0.2)'))
            fig_forecast.update_layout(
                title=f"Moisture Forecast - Next {forecast_horizon} Days",
                xaxis_title="Date",
                yaxis_title="VWC",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Forecast summary
            st.markdown("### 📋 Forecast Summary")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                current_avg = depth_data['y'].tail(7).mean()
                st.metric("Current Avg (7 days)", f"{current_avg:.3f}")
            with sc2:
                forecast_avg = future_dates['yhat'].mean()
                st.metric("Forecast Avg", f"{forecast_avg:.3f}",
                          delta=f"{(forecast_avg - current_avg):.3f}" if pd.notna(current_avg) else None)
            with sc3:
                trend = "Increasing" if pd.notna(current_avg) and forecast_avg > current_avg else "Decreasing"
                st.metric("Trend", trend)

            # Forecast table + download
            st.markdown("### 📅 Daily Forecast Values")
            forecast_table = future_dates[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_table.columns = ['Date', 'Predicted VWC', 'Lower Bound', 'Upper Bound']
            forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')
            forecast_table = forecast_table.round(3)
            st.dataframe(forecast_table, use_container_width=True)

            st.download_button(
                "📥 Download Forecast",
                forecast_table.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_{selected_depth}_{forecast_horizon}d.csv",
                mime="text/csv"
            )

# -------------------------
# TAB 5: HELP
# -------------------------
with tab5:
    st.header("❓ Help & Documentation")
    st.markdown("""
    **Welcome to the KSU TAPS Soil Dashboard**

    This dashboard helps you monitor and analyze soil moisture (VWC), EC, and irrigation overlays.

    **Tabs**
    1. **Data Summary** — sanity checks, KPIs, missing values, quick stats  
    2. **Tube Readings** — per-plot VWC trends, moving averages, depth heatmap  
    3. **AquaSpy Data** — high-frequency moisture & EC with irrigation bars (mm), resampling & smoothing  
    4. **Team Overview** — (your existing tab content)  
    5. **Predictions** — Prophet-based short-term forecasts with confidence intervals  

    **Key Concepts**
    - **VWC** (m³/m³) typical range 0.10–0.50; optimal 0.25–0.40  
    - **EC**: proxy for salinity; monitor for buildup  
    - **Irrigation overlay**: bars from the top on a reversed secondary axis, in **mm** (converted from inches)  
    - **Resampling**: aggregate 15-min data to 30min/1h/6h/Daily (mean/median/min/max)  

    **Tips**
    - Use the resample + smoothing options in AquaSpy to reduce noise  
    - Overlay irrigation to visually link moisture jumps with applications  
    - Use CSV downloads to export exactly what you plotted  
    - JSON exports are generated for chatbot integration in: `chatbot_data/`  

    **Troubleshooting**
    - If a sheet shows no timestamps, check the column names (`Timestamp`, `DateTime`, or `Date`)  
    - If irrigation overlay is empty, verify the irrigation file and team labels (numeric team ids)  
    - Forecasts require a reasonably continuous time series for the selected depth  
    """)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("""
    <footer style='text-align: center; padding: 20px; color: #666;'>
        <p><strong>2024 KSU TAPS Soil Water Dynamics Dashboard</strong></p>
        <p>Developed by The Chefs Team • 
        <a href="https://github.com/your-repo" target="_blank">GitHub</a> • 
        <a href="mailto:support@ksutaps.edu">Contact</a></p>
        <p style='font-size: 12px; margin-top: 10px;'>
            ✨ Enhanced with Smart Alerts, High-frequency AquaSpy views, EC axis, Irrigation overlays, and JSON export
        </p>
    </footer>
""", unsafe_allow_html=True)

