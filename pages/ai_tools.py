"""
ai_tools.py
===========
Analytical functions for AI-powered crop health data analysis.
These functions are called by the OpenAI assistant via function calling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta
from scipy import stats


# ============================================================
# Shared Utilities (date hygiene, safe math, time-aware regressions)
# ============================================================

def _ensure_date_col(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    """Return a copy with a normalized date column (python date objects)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.date
    return out


def _as_date(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return pd.to_datetime(x, errors="coerce").date()
    except Exception:
        return None


def _safe_div(num, den, default=0.0):
    try:
        if den and np.isfinite(den) and float(den) != 0.0:
            return float(num) / float(den)
    except Exception:
        pass
    return default


def _safe_pct(num, den) -> float:
    return round(_safe_div(num, den, 0.0) * 100.0, 1)


def _clip01(x):
    if not np.isfinite(x):
        return np.nan
    return float(np.clip(x, 0.0, 1.0))


def _fit_linreg_time(df: pd.DataFrame, y_col: str = "Mean", date_col: str = "Date"):
    """Linear regression using actual day offsets (handles irregular sampling)."""
    if df is None or len(df) < 2:
        return None
    dfd = df.copy()
    dfd = _ensure_date_col(dfd, date_col)
    base = pd.to_datetime(dfd[date_col]).min()
    x = (pd.to_datetime(dfd[date_col]) - base).dt.days.to_numpy()
    y = dfd[y_col].astype(float).to_numpy()

    if len(np.unique(x)) < 2 or np.allclose(y, y[0]):
        return None

    slope, intercept, r, p, _ = stats.linregress(x, y)
    return dict(slope=float(slope), intercept=float(intercept), r2=float(r**2), p=float(p), base=base)


# ============================================================
# 1. STRESSED PLOTS IDENTIFICATION
# ============================================================

def get_stressed_plots(master_zonal_stats: pd.DataFrame, index_name: str = 'NDVI',
                       threshold: float = 0.5, date=None, treatment=None, n_plots: int = 10):
    df = master_zonal_stats.copy()
    df = _ensure_date_col(df)
    df = df[df['Index'] == index_name].copy()
    if df.empty:
        return {'error': 'No data for index', 'index': index_name}

    d = _as_date(date) or df['Date'].max()
    df = df[df['Date'] == d]

    if treatment is not None:
        df = df[df['TRT_ID'].astype(str) == str(treatment)]

    if df.empty:
        return {'error': 'No data for filters', 'date': str(d), 'index': index_name}

    below = df['Mean'] < threshold
    stressed = df[below].sort_values('Mean').head(int(max(1, n_plots)))

    return {
        'date': str(d),
        'index': index_name,
        'threshold': float(threshold),
        'total_stressed': int(below.sum()),
        'total_plots': int(len(df)),
        'stressed_percentage': _safe_pct(below.sum(), len(df)),
        'top_stressed_plots': stressed[['Plot_ID', 'TRT_ID', 'Mean']].to_dict('records')
    }


# ============================================================
# 2. TREATMENT PERCENTILE RANKING
# ============================================================

def calculate_treatment_percentile(master_zonal_stats: pd.DataFrame, index_name: str = 'NDVI',
                                   date=None, metric: str = 'mean'):
    df = master_zonal_stats.copy()
    df = _ensure_date_col(df)
    df = df[df['Index'] == index_name].copy()
    if df.empty:
        return {'error': 'No data for index', 'index': index_name}

    d = _as_date(date) or df['Date'].max()
    df = df[df['Date'] == d]
    if df.empty:
        return {'error': 'No data for selected date', 'date': str(d), 'index': index_name}

    if metric == 'mean':
        trt_values = df.groupby('TRT_ID')['Mean'].mean()
    elif metric == 'median':
        trt_values = df.groupby('TRT_ID')['Mean'].median()
    else:
        trt_values = df.groupby('TRT_ID')['Mean'].max()

    trt_values = trt_values.sort_values(ascending=False)
    n = len(trt_values)
    if n == 0:
        return {'error': 'No treatments found for selected filters'}

    results = []
    for rank, (trt_id, value) in enumerate(trt_values.items(), 1):
        # 0th percentile for top rank; 100th for bottom if you prefer the opposite, flip formula.
        percentile = 0.0 if n <= 1 else round((rank - 1) * 100.0 / (n - 1), 1)
        results.append({
            'TRT_ID': str(trt_id),
            'rank': int(rank),
            'value': round(float(value), 4),
            'percentile': float(percentile)
        })

    return {'date': str(d), 'index': index_name, 'metric': metric, 'rankings': results}


# ============================================================
# 3. TEMPORAL TREND ANALYSIS
# ============================================================

def analyze_temporal_trend(master_zonal_stats: pd.DataFrame, entity_type: str = 'plot',
                           entity_id=None, index_name: str = 'NDVI', period: str = 'full'):
    df = master_zonal_stats.copy()
    df = _ensure_date_col(df)
    df = df[df['Index'] == index_name].copy()

    if entity_type == 'plot':
        df = df[df['Plot_ID'].astype(str) == str(entity_id)]
    else:
        df = df[df['TRT_ID'].astype(str) == str(entity_id)]
        df = df.groupby('Date', as_index=False)['Mean'].mean()

    if df.empty:
        return {'error': 'No data for entity', 'entity_type': entity_type, 'entity_id': str(entity_id)}

    max_date = df['Date'].max()
    if period == 'last_30_days':
        df = df[df['Date'] >= (max_date - timedelta(days=30))]
    elif period == 'last_week':
        df = df[df['Date'] >= (max_date - timedelta(days=7))]

    df = df.sort_values('Date')
    if len(df) < 2:
        return {'error': 'Insufficient data points for trend analysis'}

    fit = _fit_linreg_time(df, y_col='Mean', date_col='Date')
    first_val = float(df['Mean'].iloc[0])
    last_val = float(df['Mean'].iloc[-1])
    pct_change = round(((last_val - first_val) / first_val) * 100, 2) if first_val > 0 else 0.0

    if fit is None:
        slope = 0.0
        r2 = 0.0
        p = 1.0
        trend = 'Flat'
    else:
        slope = float(fit['slope'])
        r2 = float(fit['r2'])
        p = float(fit['p'])
        trend = 'Improving' if slope > 0 else ('Declining' if slope < 0 else 'Flat')

    return {
        'entity_type': entity_type,
        'entity_id': str(entity_id),
        'index': index_name,
        'period': period,
        'n_observations': int(len(df)),
        'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
        'first_value': round(first_val, 4),
        'last_value': round(last_val, 4),
        'slope': round(slope, 6),
        'r_squared': round(r2, 4),
        'p_value': round(p, 4),
        'percent_change': pct_change,
        'trend': trend,
        'significance': 'Significant' if p < 0.05 else 'Not Significant',
        'time_series': df[['Date', 'Mean']].to_dict('records')
    }


# ============================================================
# 4. NITROGEN RESPONSE EFFICIENCY
# ============================================================

def compare_nitrogen_response(master_zonal_stats: pd.DataFrame, N_df: pd.DataFrame,
                              treatment_ids, index_name: str = 'NDVI',
                              days_after_application: int = 14):
    df = _ensure_date_col(master_zonal_stats)
    ndf = _ensure_date_col(N_df)
    df = df[(df['Index'] == index_name)]
    if df.empty or ndf is None or ndf.empty:
        return {'error': 'Insufficient data for analysis'}

    results = []
    for trt_id in treatment_ids or []:
        trt_id = str(trt_id)
        n_apps = ndf[ndf['TRT_ID'].astype(str) == trt_id].copy()
        if n_apps.empty:
            continue

        trt_index = df[df['TRT_ID'].astype(str) == trt_id].sort_values('Date')
        if trt_index.empty:
            continue

        # Use date-wise means for stable comparisons
        trt_daily = trt_index.groupby('Date', as_index=False)['Mean'].mean()

        for _, app in n_apps.iterrows():
            app_date = app['Date']
            n_amount = float(app['Amount'])

            # Index before application (last value <= app_date)
            before_data = trt_daily[trt_daily['Date'] <= app_date]
            if before_data.empty:
                continue
            index_before = float(before_data.iloc[-1]['Mean'])

            # Index after application (nearest >= app_date + window)
            target_date = app_date + timedelta(days=int(days_after_application))
            after_data = trt_daily[trt_daily['Date'] >= target_date]
            if after_data.empty:
                continue
            index_after = float(after_data.iloc[0]['Mean'])

            index_change = index_after - index_before
            efficiency = _safe_div(index_change, n_amount, default=0.0)

            results.append({
                'TRT_ID': trt_id,
                'application_date': str(app_date),
                'n_amount_lbs': float(n_amount),
                'index_before': round(index_before, 4),
                'index_after': round(index_after, 4),
                'index_change': round(index_change, 4),
                'efficiency': round(efficiency, 6),
                'days_evaluated': int(days_after_application)
            })

    if not results:
        return {'error': 'No nitrogen application data found for specified treatments'}

    efficiencies = [r['efficiency'] for r in results]
    return {
        'index': index_name,
        'days_after_application': int(days_after_application),
        'n_treatments_analyzed': int(len(treatment_ids or [])),
        'n_applications': int(len(results)),
        'avg_efficiency': round(float(np.mean(efficiencies)), 6),
        'max_efficiency': round(float(np.max(efficiencies)), 6),
        'min_efficiency': round(float(np.min(efficiencies)), 6),
        'applications': results
    }


# ============================================================
# 5. WATER NEED ESTIMATION
# ============================================================

def estimate_water_needs(master_zonal_stats: pd.DataFrame, entity_type: str = 'plot',
                         entity_ids=None, eto_mm: float = 35.0, rainfall_mm: float = 10.0):
    df = _ensure_date_col(master_zonal_stats)
    ndvi = df[df['Index'] == 'NDVI'].copy()
    if ndvi.empty:
        return {'error': 'No NDVI data available'}

    latest_date = ndvi['Date'].max()
    ndvi = ndvi[ndvi['Date'] == latest_date]

    if entity_type == 'plot':
        ndvi = ndvi[ndvi['Plot_ID'].astype(str).isin([str(e) for e in (entity_ids or [])])]
        entity_col = 'Plot_ID'
    else:
        ndvi = ndvi[ndvi['TRT_ID'].astype(str).isin([str(e) for e in (entity_ids or [])])]
        ndvi = ndvi.groupby('TRT_ID', as_index=False)['Mean'].mean()
        entity_col = 'TRT_ID'

    if ndvi.empty:
        return {'error': 'No matching entities for water need estimation', 'date': str(latest_date)}

    def calc_kc(ndvi_val):
        if not np.isfinite(ndvi_val):
            return np.nan
        ndvi_v = float(np.clip(ndvi_val, 0.0, 1.0))
        fc = np.clip(1.26 * ndvi_v - 0.18, 0.0, 1.0)
        kc = np.clip(1.13 * fc + 0.14, 0.1, 1.2)
        return float(kc)

    ndvi['Kc'] = ndvi['Mean'].map(calc_kc)
    if ndvi['Kc'].dropna().empty:
        return {'error': 'No valid NDVI values to compute Kc', 'date': str(latest_date)}

    # Effective rainfall: simple USDA-like heuristic
    if rainfall_mm < 25:
        effective_rain = rainfall_mm * 0.95
    else:
        effective_rain = 25 * 0.95 + (rainfall_mm - 25) * 0.75

    ndvi['Water_Need_mm'] = ndvi['Kc'] * float(eto_mm) - effective_rain
    ndvi['Water_Need_mm'] = ndvi['Water_Need_mm'].clip(lower=0)
    ndvi['Water_Need_in'] = ndvi['Water_Need_mm'] / 25.4

    results = ndvi[[entity_col, 'Mean', 'Kc', 'Water_Need_mm', 'Water_Need_in']].to_dict('records')

    return {
        'date': str(latest_date),
        'entity_type': entity_type,
        'eto_mm': float(eto_mm),
        'rainfall_mm': float(rainfall_mm),
        'effective_rainfall_mm': round(float(effective_rain), 2),
        'avg_kc': round(float(ndvi['Kc'].mean()), 3),
        'avg_water_need_in': round(float(ndvi['Water_Need_in'].mean()), 2),
        'entities': results
    }


# ============================================================
# 6. ANOMALY DETECTION
# ============================================================

def identify_anomalies(master_zonal_stats: pd.DataFrame, index_name: str = 'NDVI',
                       date=None, threshold_stddev: float = 2.0):
    df = _ensure_date_col(master_zonal_stats)
    df = df[df['Index'] == index_name].copy()
    if df.empty:
        return {'error': 'No data for index', 'index': index_name}

    d = _as_date(date) or df['Date'].max()
    df = df[df['Date'] == d]
    if df.empty:
        return {'error': 'No data for selected date', 'date': str(d), 'index': index_name}

    mean_val = float(df['Mean'].mean())
    std_val = float(df['Mean'].std()) if np.isfinite(df['Mean'].std()) else 0.0

    if not np.isfinite(std_val) or std_val == 0.0:
        return {
            'date': str(d),
            'index': index_name,
            'threshold_stddev': float(threshold_stddev),
            'population_mean': round(mean_val, 4),
            'population_std': 0.0,
            'total_plots': int(len(df)),
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0,
            'anomalous_plots': []
        }

    z = (df['Mean'] - mean_val) / std_val
    df['z_score'] = z
    df['anomaly'] = df['z_score'].abs() > float(threshold_stddev)
    anomalies = df[df['anomaly']].copy()
    anomalies['deviation'] = _safe_div((anomalies['Mean'] - mean_val), mean_val, 0.0) * 100.0

    return {
        'date': str(d),
        'index': index_name,
        'threshold_stddev': float(threshold_stddev),
        'population_mean': round(mean_val, 4),
        'population_std': round(std_val, 4),
        'total_plots': int(len(df)),
        'anomalies_detected': int(len(anomalies)),
        'anomaly_percentage': _safe_pct(len(anomalies), len(df)),
        'anomalous_plots': anomalies[['Plot_ID', 'TRT_ID', 'Mean', 'z_score', 'deviation']].to_dict('records')
    }


# ============================================================
# 7. GHG CORRELATION ANALYSIS
# ============================================================

def analyze_ghg_correlation(master_zonal_stats: pd.DataFrame, ghg_df: pd.DataFrame,
                            treatment_ids, index_name: str = 'NDVI'):
    df = _ensure_date_col(master_zonal_stats)
    gdf = _ensure_date_col(ghg_df)
    if df is None or df.empty or gdf is None or gdf.empty:
        return {'error': 'Insufficient data for correlation'}

    results = []
    for trt_id in treatment_ids or []:
        trt_id = str(trt_id)
        idx = df[(df['TRT_ID'].astype(str) == trt_id) & (df['Index'] == index_name)].groupby('Date', as_index=False)['Mean'].mean()
        ghg = gdf[gdf['TRT_ID'].astype(str) == trt_id].groupby('Date', as_index=False)['N2O_Flux'].mean()

        merged = pd.merge(idx, ghg, on='Date', how='inner')
        if len(merged) < 3:
            continue

        corr = merged['Mean'].corr(merged['N2O_Flux'])
        corr = float(corr) if np.isfinite(corr) else 0.0
        merged['efficiency'] = merged['Mean'] / (merged['N2O_Flux'].abs() + 1e-3)

        results.append({
            'TRT_ID': trt_id,
            'n_observations': int(len(merged)),
            'avg_index': round(float(merged['Mean'].mean()), 4),
            'avg_n2o_flux': round(float(merged['N2O_Flux'].mean()), 4),
            'correlation': round(corr, 4),
            'avg_efficiency': round(float(merged['efficiency'].mean()), 4)
        })

    if not results:
        return {'error': 'No overlapping data found for specified treatments'}

    return {
        'index': index_name,
        'n_treatments_analyzed': int(len(results)),
        'avg_correlation': round(float(np.mean([r['correlation'] for r in results])), 4),
        'treatments': results
    }


# ============================================================
# 8. COST-BENEFIT ANALYSIS
# ============================================================

def cost_benefit_analysis(master_zonal_stats: pd.DataFrame, N_df: pd.DataFrame, treatment_ids,
                          include_nitrogen_cost: bool = True, include_water_cost: bool = True,
                          n_cost_per_lb: float = 0.50, water_cost_per_inch: float = 25.0,
                          water_inches: float = 1.0):
    df = _ensure_date_col(master_zonal_stats)
    ndf = _ensure_date_col(N_df)
    df = df[df['Index'] == 'NDVI']
    if df.empty:
        return {'error': 'No NDVI data available'}

    results = []
    for trt_id in treatment_ids or []:
        trt_id = str(trt_id)
        avg_ndvi = float(df[df['TRT_ID'].astype(str) == trt_id]['Mean'].mean())
        avg_ndvi = avg_ndvi if np.isfinite(avg_ndvi) else 0.0

        total_cost = 0.0
        costs = {}

        if include_nitrogen_cost and (ndf is not None) and (not ndf.empty):
            n_total = float(ndf[ndf['TRT_ID'].astype(str) == trt_id]['Amount'].sum())
            n_cost = n_total * float(n_cost_per_lb)
            costs['nitrogen'] = round(n_cost, 2)
            total_cost += n_cost

        if include_water_cost:
            w_cost = float(water_inches) * float(water_cost_per_inch)
            costs['water'] = round(w_cost, 2)
            total_cost += w_cost

        performance_score = avg_ndvi * 100.0
        cost_eff = _safe_div(performance_score, total_cost, 0.0)

        results.append({
            'TRT_ID': trt_id,
            'avg_ndvi': round(avg_ndvi, 4),
            'performance_score': round(performance_score, 2),
            'total_cost': round(total_cost, 2),
            'cost_breakdown': costs,
            'cost_efficiency': round(cost_eff, 4)
        })

    if not results:
        return {'error': 'No data found for specified treatments'}

    results.sort(key=lambda x: x['cost_efficiency'], reverse=True)
    return {
        'n_treatments_analyzed': int(len(results)),
        'analysis_includes': {'nitrogen_cost': bool(include_nitrogen_cost), 'water_cost': bool(include_water_cost)},
        'treatments': results
    }


# ============================================================
# 9. PRESCRIPTION MAP GENERATION
# ============================================================

def generate_prescription_map(master_zonal_stats: pd.DataFrame, index_name: str = 'NDVI',
                              date=None, input_type: str = 'nitrogen', target_uniformity: float = 15.0):
    df = _ensure_date_col(master_zonal_stats)
    df = df[df['Index'] == index_name].copy()
    if df.empty:
        return {'error': 'No data for index', 'index': index_name}

    d = _as_date(date) or df['Date'].max()
    df = df[df['Date'] == d]
    if df.empty:
        return {'error': 'No data for selected date', 'date': str(d), 'index': index_name}

    mean_val = float(df['Mean'].mean())
    std_val = float(df['Mean'].std()) if np.isfinite(df['Mean'].std()) else 0.0
    if not np.isfinite(mean_val) or mean_val == 0.0:
        return {'error': 'Population mean is zero/invalid for prescriptions', 'date': str(d)}

    current_cv = (std_val / mean_val) * 100.0 if mean_val != 0 else 0.0

    # Lower index -> higher input factor
    df['deviation_from_mean'] = mean_val - df['Mean']
    df['prescription_factor'] = 1.0 + (df['deviation_from_mean'] / mean_val)
    df['prescription_factor'] = df['prescription_factor'].clip(0.5, 1.5)

    if input_type in ['nitrogen', 'both']:
        base_n_rate = 100.0  # lbs/acre
        df['N_recommendation'] = (df['prescription_factor'] * base_n_rate).round(1)

    if input_type in ['water', 'both']:
        base_water_rate = 1.0  # inches
        df['Water_recommendation'] = (df['prescription_factor'] * base_water_rate).round(2)

    cols = ['Plot_ID', 'TRT_ID', 'Mean', 'prescription_factor']
    if input_type in ['nitrogen', 'both']:
        cols.append('N_recommendation')
    if input_type in ['water', 'both']:
        cols.append('Water_recommendation')

    prescriptions = df[cols].copy()
    # Normalize keys in output
    if 'N_recommendation' in prescriptions.columns:
        prescriptions = prescriptions.rename(columns={'N_recommendation': 'N_recommendation_lbs'})
    if 'Water_recommendation' in prescriptions.columns:
        prescriptions = prescriptions.rename(columns={'Water_recommendation': 'Water_recommendation_in'})

    return {
        'date': str(d),
        'index': index_name,
        'input_type': input_type,
        'current_cv_percent': round(current_cv, 2),
        'target_cv_percent': float(target_uniformity),
        'population_mean': round(mean_val, 4),
        'n_plots': int(len(df)),
        'prescriptions': prescriptions.to_dict('records')
    }


# ============================================================
# 10. STRESS RISK FORECAST
# ============================================================

def forecast_stress_risk(master_zonal_stats: pd.DataFrame, plot_ids, days_ahead: int = 7,
                         weather_scenario: str = 'normal', index_name: str = 'NDVI'):
    df = _ensure_date_col(master_zonal_stats)
    df = df[df['Index'] == index_name].copy()
    if df.empty:
        return {'error': 'No data for index', 'index': index_name}

    weather_factors = {'dry': -0.05, 'normal': 0.00, 'wet': 0.02}
    weather_factor = float(weather_factors.get(weather_scenario, 0.0))

    results = []
    for plot_id in (plot_ids or []):
        plot_id = str(plot_id)
        plot_data = df[df['Plot_ID'].astype(str) == plot_id].sort_values('Date').tail(10)
        if len(plot_data) < 2:
            continue

        fit = _fit_linreg_time(plot_data, y_col='Mean', date_col='Date')
        current_value = float(plot_data['Mean'].iloc[-1])

        if fit is None:
            forecasted_value = current_value + weather_factor * (days_ahead / 7.0)
        else:
            # Predict value at (last_day_offset + days_ahead)
            last_day_offset = (pd.to_datetime(plot_data['Date']).max() - fit['base']).days
            forecasted_value = fit['intercept'] + fit['slope'] * (last_day_offset + int(days_ahead))
            forecasted_value += weather_factor * (days_ahead / 7.0)

        forecasted_value = _clip01(forecasted_value)
        stress_threshold = 0.5
        risk_score = _clip01(max(0.0, (stress_threshold - forecasted_value) / stress_threshold)) * 100.0

        if risk_score > 70:
            risk_level = 'High'
        elif risk_score > 40:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        results.append({
            'Plot_ID': plot_id,
            'current_value': round(current_value, 4),
            'forecasted_value': round(forecasted_value, 4),
            'trend_slope': round(float(fit['slope']) if fit else 0.0, 6),
            'risk_score': round(float(risk_score), 1),
            'risk_level': risk_level,
            'days_ahead': int(days_ahead),
            'weather_scenario': weather_scenario
        })

    if not results:
        return {'error': 'Insufficient data for forecasting'}

    return {
        'index': index_name,
        'days_ahead': int(days_ahead),
        'weather_scenario': weather_scenario,
        'n_plots_analyzed': int(len(results)),
        'high_risk_count': int(sum(1 for r in results if r['risk_level'] == 'High')),
        'forecasts': results
    }
# ============================================================
# A. UNIFORMITY & HOTSPOT MAPPING
# ============================================================
def spatial_uniformity_report(master_zonal_stats, index_name='NDVI', date=None,
                              by='treatment', hotspot_quantile=0.10):
    """
    Quantify spatial uniformity and surface hotspots/coldspots.

    Args:
        master_zonal_stats: DataFrame with columns [Plot_ID, TRT_ID, Index, Date, Mean]
        index_name: 'NDVI' or 'MCARI2'
        date: specific date (YYYY-MM-DD) or None -> latest
        by: 'treatment' or 'overall' (CV per treatment or single CV for whole field)
        hotspot_quantile: bottom/top quantile to flag hotspots/coldspots (0-1)

    Returns:
        dict with cv, iqr, thresholds, hotspot/coldspot lists and group stats
    """
    df = master_zonal_stats[master_zonal_stats['Index'] == index_name].copy()
    if df.empty:
        return {'error': f'No {index_name} data available.'}

    if date is None:
        date = df['Date'].max()
    else:
        date = pd.to_datetime(date).date()

    df = df[df['Date'] == date].copy()
    if df.empty:
        return {'error': f'No {index_name} data on {date}.'}

    vals = df['Mean'].astype(float)
    mean_val = float(vals.mean())
    std_val = float(vals.std(ddof=0))
    cv_pct = float((std_val / mean_val) * 100) if mean_val > 0 else 0.0
    q_low = float(vals.quantile(hotspot_quantile))
    q_hi  = float(vals.quantile(1.0 - hotspot_quantile))
    iqr   = float(vals.quantile(0.75) - vals.quantile(0.25))

    coldspots = df[df['Mean'] <= q_low].sort_values('Mean').loc[:, ['Plot_ID','TRT_ID','Mean']]
    hotspots  = df[df['Mean'] >= q_hi].sort_values('Mean', ascending=False).loc[:, ['Plot_ID','TRT_ID','Mean']]

    # group stats (per treatment) if requested
    group_stats = []
    if by == 'treatment':
        g = df.groupby('TRT_ID')['Mean']
        for trt, s in g:
            m = float(s.mean())
            sd = float(s.std(ddof=0))
            cv = float((sd / m) * 100) if m > 0 else 0.0
            group_stats.append({'TRT_ID': str(trt), 'mean': round(m,4), 'std': round(sd,4), 'cv_percent': round(cv,2)})

        group_stats = sorted(group_stats, key=lambda x: x['cv_percent'], reverse=True)

    return {
        'date': str(date),
        'index': index_name,
        'total_plots': int(len(df)),
        'population_mean': round(mean_val, 4),
        'population_std': round(std_val, 4),
        'cv_percent': round(cv_pct, 2),
        'iqr': round(iqr, 4),
        'thresholds': {
            'coldspot_max': round(q_low, 4),
            'hotspot_min': round(q_hi, 4),
            'quantile': hotspot_quantile
        },
        'coldspots': coldspots.assign(Mean=lambda x: x['Mean'].round(4)).to_dict('records'),
        'hotspots': hotspots.assign(Mean=lambda x: x['Mean'].round(4)).to_dict('records'),
        'group_stats': group_stats
    }
# ============================================================
# B. CHANGE DETECTION BETWEEN DATES
# ============================================================
def delta_map(master_zonal_stats, index_name='NDVI', date_from=None, date_to=None, by='plot'):
    """
    Highlight where the index improved/declined between two dates.

    Args:
        master_zonal_stats: DataFrame with columns [Plot_ID, TRT_ID, Index, Date, Mean]
        index_name: 'NDVI' or 'MCARI2'
        date_from: earlier date (None -> earliest available)
        date_to: later date (None -> latest available)
        by: 'plot' or 'treatment' (treatment uses mean across plots)

    Returns:
        dict with summary and per-entity deltas (sorted by delta ascending)
    """
    df = master_zonal_stats[master_zonal_stats['Index'] == index_name].copy()
    if df.empty:
        return {'error': f'No {index_name} data available.'}

    # Resolve dates
    dates_sorted = sorted(df['Date'].unique())
    if not dates_sorted:
        return {'error': 'No dates available.'}

    if date_from is None:
        date_from = dates_sorted[0]
    else:
        date_from = pd.to_datetime(date_from).date()

    if date_to is None:
        date_to = dates_sorted[-1]
    else:
        date_to = pd.to_datetime(date_to).date()

    if date_from >= date_to:
        return {'error': f'date_from ({date_from}) must be before date_to ({date_to}).'}

    # Aggregate by entity/date
    if by == 'treatment':
        dfg = df.groupby(['TRT_ID', 'Date'], as_index=False)['Mean'].mean()
        left_key, right_key, entity = 'TRT_ID', 'TRT_ID', 'TRT_ID'
    else:
        dfg = df.groupby(['Plot_ID', 'TRT_ID', 'Date'], as_index=False)['Mean'].mean()
        left_key, right_key, entity = ['Plot_ID','TRT_ID'], ['Plot_ID','TRT_ID'], 'Plot_ID'

    df_from = dfg[dfg['Date'] == date_from].copy()
    df_to   = dfg[dfg['Date'] == date_to].copy()
    merged = pd.merge(df_from, df_to, left_on=left_key, right_on=right_key, how='inner', suffixes=('_from','_to'))

    if merged.empty:
        return {'error': 'No overlapping entities between the two dates.'}

    merged['delta'] = merged['Mean_to'] - merged['Mean_from']
    merged['pct_change'] = np.where(merged['Mean_from'] > 0,
                                    100.0 * (merged['Mean_to'] - merged['Mean_from']) / merged['Mean_from'],
                                    np.nan)

    deltas = merged['delta'].astype(float)
    sigma = float(deltas.std(ddof=0))
    improved = int((merged['delta'] > 0).sum())
    declined = int((merged['delta'] < 0).sum())

    # Prepare output rows (sorted by delta asc = worst first)
    cols_out = [entity, 'TRT_ID'] if by == 'plot' else ['TRT_ID']
    records = merged.sort_values('delta').loc[:, cols_out + ['Mean_from','Mean_to','delta','pct_change']]
    out_rows = records.copy()
    for c in ['Mean_from','Mean_to','delta','pct_change']:
        out_rows[c] = out_rows[c].astype(float).round(4)

    return {
        'index': index_name,
        'date_from': str(date_from),
        'date_to': str(date_to),
        'entities_compared': int(len(merged)),
        'avg_delta': round(float(deltas.mean()), 4),
        'std_delta': round(sigma, 4),
        'pct_improved': round(100.0 * improved / len(merged), 1),
        'pct_declined': round(100.0 * declined / len(merged), 1),
        'changes': out_rows.to_dict('records')
    }
# ============================================================
# C. NITROGEN RESPONSE CURVE (per TRT)
# ============================================================
def nitrogen_response_curve(master_zonal_stats, N_df, trt_id, index_name='NDVI', horizons=(7,14,21)):
    """
    For a given treatment, compute NDVI response to each N application
    at multiple horizons (e.g., 7/14/21 days), and fit marginal gain per lb.

    Args:
        master_zonal_stats: DataFrame with [Plot_ID, TRT_ID, Index, Date, Mean]
        N_df: DataFrame with [TRT_ID, Date, Amount]
        trt_id: treatment ID as str/int
        index_name: 'NDVI' or 'MCARI2'
        horizons: iterable of day horizons (ints)

    Returns:
        dict with per-application responses and fitted slope/R² per horizon
    """
    trt_id = str(trt_id)

    if N_df is None or N_df.empty:
        return {'error': 'Nitrogen application data is empty.'}

    df = master_zonal_stats[(master_zonal_stats['Index'] == index_name) &
                            (master_zonal_stats['TRT_ID'] == trt_id)].copy()
    if df.empty:
        return {'error': f'No {index_name} data for TRT {trt_id}.'}

    # Aggregate to treatment mean by date
    series = df.groupby('Date', as_index=False)['Mean'].mean().sort_values('Date')
    if series.empty:
        return {'error': f'No time series for TRT {trt_id}.'}

    # N applications for this TRT
    apps = N_df[N_df['TRT_ID'] == trt_id].copy()
    if apps.empty:
        return {'error': f'No nitrogen applications for TRT {trt_id}.'}

    apps = apps.sort_values('Date')
    horizons = sorted(set(int(h) for h in horizons if int(h) > 0))
    per_app = []

    # Helper: nearest value on/before date; nearest on/after date
    def _nearest_on_or_before(d):
        sub = series[series['Date'] <= d]
        return float(sub.iloc[-1]['Mean']) if not sub.empty else np.nan

    def _nearest_on_or_after(d):
        sub = series[series['Date'] >= d]
        return float(sub.iloc[0]['Mean']) if not sub.empty else np.nan

    # Collect responses
    for _, row in apps.iterrows():
        app_date = row['Date']
        dose = float(row['Amount']) if pd.notna(row['Amount']) else 0.0
        before = _nearest_on_or_before(app_date)
        horizon_vals = {}

        for h in horizons:
            target = app_date + timedelta(days=int(h))
            after = _nearest_on_or_after(target)
            delta = (after - before) if (np.isfinite(before) and np.isfinite(after)) else np.nan
            mgain = (delta / dose) if (dose > 0 and np.isfinite(delta)) else np.nan
            horizon_vals[int(h)] = {
                'index_after': round(after, 4) if np.isfinite(after) else None,
                'delta': round(delta, 4) if np.isfinite(delta) else None,
                'marginal_gain_per_lb': round(mgain, 6) if np.isfinite(mgain) else None
            }

        per_app.append({
            'application_date': str(app_date),
            'dose_lbs': round(dose, 2),
            'index_before': round(before, 4) if np.isfinite(before) else None,
            'horizons': horizon_vals
        })

    # Fit slope (delta ~ dose) for each horizon (simple OLS) if >=2 apps with valid data
    fits = {}
    for h in horizons:
        xs, ys = [], []
        for a in per_app:
            dose = a['dose_lbs']
            info = a['horizons'].get(int(h), {})
            delta = info.get('delta', None)
            if (dose is not None) and (delta is not None):
                xs.append(float(dose))
                ys.append(float(delta))
        if len(xs) >= 2 and len(ys) >= 2 and np.std(xs) > 0:
            slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
            fits[int(h)] = {
                'slope_delta_per_lb': round(float(slope), 6),
                'intercept': round(float(intercept), 6),
                'r_squared': round(float(r_value**2), 4),
                'p_value': round(float(p_value), 4),
                'n_points': len(xs)
            }
        else:
            fits[int(h)] = {
                'slope_delta_per_lb': None,
                'intercept': None,
                'r_squared': None,
                'p_value': None,
                'n_points': len(xs)
            }

    return {
        'index': index_name,
        'TRT_ID': trt_id,
        'horizons_days': horizons,
        'n_applications': len(per_app),
        'per_application': per_app,
        'fits': fits
    }
# ============================================================
# D. STRESS DIAGNOSTICS (water vs nitrogen vs anomaly)
# ============================================================
def stress_diagnostics(master_zonal_stats, N_df, eto_mm, rainfall_mm, date=None,
                       index_name='NDVI', threshold=0.5, nitrogen_gap_pct=0.25):
    """
    Provide a simple heuristic 'best guess' of likely stress driver per plot.

    Args:
        master_zonal_stats: DataFrame [Plot_ID, TRT_ID, Index, Date, Mean]
        N_df: DataFrame [TRT_ID, Date, Amount] (can be empty)
        eto_mm: reference ET over recent period (e.g., last week) in mm
        rainfall_mm: rainfall over recent period in mm
        date: specific date or None -> latest
        index_name: 'NDVI' or 'MCARI2'
        threshold: index threshold for stress flag
        nitrogen_gap_pct: if TRT total N is < (1 - nitrogen_gap_pct) * median(TRT N), flag as N-limited

    Returns:
        dict with reasons per stressed plot and summarized counts
    """
    df = master_zonal_stats[master_zonal_stats['Index'] == index_name].copy()
    if df.empty:
        return {'error': f'No {index_name} data available.'}

    if date is None:
        date = df['Date'].max()
    else:
        date = pd.to_datetime(date).date()
    df = df[df['Date'] == date].copy()
    if df.empty:
        return {'error': f'No {index_name} data on {date}.'}

    # Stressed plots
    stressed = df[df['Mean'] < float(threshold)].copy()
    if stressed.empty:
        return {
            'date': str(date),
            'index': index_name,
            'threshold': threshold,
            'stressed_count': 0,
            'notes': 'No plots below the stress threshold.'
        }

    # Nitrogen context (cumulative up to date, per TRT)
    trt_n_totals = {}
    if N_df is not None and not N_df.empty:
        n_up_to = N_df[N_df['Date'] <= date].copy()
        if not n_up_to.empty:
            trt_n_totals = n_up_to.groupby('TRT_ID')['Amount'].sum().astype(float).to_dict()

    # Median N across treatments (0 if not available)
    if trt_n_totals:
        median_n = float(np.median(list(trt_n_totals.values())))
    else:
        median_n = 0.0

    # Simple water heuristic: high ETo + low rainfall → “water-limited”
    # (tune thresholds to your climate; these are generic)
    eto_high = eto_mm >= 35.0   # ~5 mm/day × 7 days
    rain_low = rainfall_mm <= 10.0

    results = []
    counters = {'water': 0, 'nitrogen': 0, 'anomaly': 0}

    for _, row in stressed.iterrows():
        plot_id = str(row['Plot_ID'])
        trt_id = str(row['TRT_ID'])
        idx_val = float(row['Mean'])

        # Default reason
        reason = 'anomaly'
        evidence = {'index_value': round(idx_val, 4)}

        # Water diagnosis
        if eto_high and rain_low:
            reason = 'water'
            evidence.update({'eto_mm': float(eto_mm), 'rainfall_mm': float(rainfall_mm)})

        # Nitrogen diagnosis (only if N tracking exists and water wasn’t already flagged)
        trt_total_n = float(trt_n_totals.get(trt_id, 0.0))
        if reason != 'water' and median_n > 0:
            if trt_total_n < (1.0 - float(nitrogen_gap_pct)) * median_n:
                reason = 'nitrogen'
                evidence.update({
                    'trt_total_n_lbs': round(trt_total_n, 2),
                    'median_trt_n_lbs': round(median_n, 2),
                    'gap_percent': int(nitrogen_gap_pct * 100)
                })

        counters[reason] += 1
        results.append({
            'Plot_ID': plot_id,
            'TRT_ID': trt_id,
            'index_value': round(idx_val, 4),
            'reason': reason,
            'evidence': evidence
        })

    return {
        'date': str(date),
        'index': index_name,
        'threshold': threshold,
        'stressed_count': int(len(stressed)),
        'diagnosis_counts': counters,
        'plots': results,
        'assumptions': {
            'water_rule': 'High ETo (>=35 mm/week) AND low rainfall (<=10 mm/week) → water-limited',
            'nitrogen_rule': f'TRT total N < (1 - {nitrogen_gap_pct}) × median(TRT N) → nitrogen-limited'
        }
    }
# ai_tools.py (add this function)


