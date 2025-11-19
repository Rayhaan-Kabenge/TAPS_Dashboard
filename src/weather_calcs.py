from __future__ import annotations

import datetime as dt
import json
import math
from typing import Dict, Iterable, Optional

import pandas as pd
import requests
import streamlit as st


def cached_request(url: str, params: Dict, timeout: int = 15, cache_minutes: int = 30):
    """Cache API requests using Streamlit session_state."""
    cache_key = f"{url}_{json.dumps(params, sort_keys=True)}"
    now = dt.datetime.now()

    if "weather_cache" not in st.session_state:
        st.session_state.weather_cache = {}

    cache = st.session_state.weather_cache
    if cache_key in cache:
        cached_data, timestamp = cache[cache_key]
        if (now - timestamp).total_seconds() < cache_minutes * 60:
            return cached_data, "cached"

    try:
        response = requests.get(url, params=params, timeout=timeout)
        data = response.json()
        cache[cache_key] = (data, now)
        return data, "live"
    except Exception:
        if cache_key in cache:
            return cache[cache_key][0], "cached_fallback"
        return {}, "error"


def ascedaily(
    rfcrp: str,
    z: float,
    lat: float,
    doy: int,
    israd: float,
    tmax: float,
    tmin: float,
    vapr: float = float("nan"),
    tdew: float = float("nan"),
    rhmax: float = float("nan"),
    rhmin: float = float("nan"),
    wndsp: float = float("nan"),
    wndht: float = 2.0,
) -> float:
    """ASCE Standardized Penman-Monteith ET calculation."""
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
        ea = (emin * rhmax / 100.0 + emax * rhmin / 100.0) / 2.0
    elif not math.isnan(rhmax):
        ea = emin * rhmax / 100.0
    elif not math.isnan(rhmin):
        ea = emax * rhmin / 100.0
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
    if math.isnan(wndsp):
        wndsp = 2.0
    u2 = wndsp * (4.87 / math.log(67.8 * wndht - 5.42))
    if rfcrp == "S":
        Cn, Cd = 900.0, 0.34
    else:
        Cn, Cd = 1600.0, 0.38
    etsz = 0.408 * Udelta * (rn - g) + psycon * (Cn / (tavg + 273.0)) * u2 * (es - ea)
    etsz = etsz / (Udelta + psycon * (1.0 + Cd * u2))
    return etsz


def calculate_effective_rainfall(precip_mm: float, method: str = "usda") -> float:
    """Estimate effective rainfall using the USDA method."""
    if method == "usda":
        if precip_mm < 25:
            return precip_mm * 0.95
        return 25 * 0.95 + (precip_mm - 25) * 0.75
    return precip_mm * 0.8


def gdd_excel_style(tmin: float, tmax: float, tbase: float, tmax_cap: float) -> float:
    """Calculate Growing Degree Days using the Excel-style method."""
    tavg = (tmin + tmax) / 2.0
    if (tmax <= tmax_cap) and (tmin >= tbase):
        return tavg - tbase
    if (tmax <= tmax_cap) and (tmax >= tbase) and (tmin < tbase):
        return ((tmax + tbase) / 2.0) - tbase
    if (tmax > tmax_cap) and (tmin >= tbase):
        return ((tmax_cap + tmin) / 2.0) - tbase
    if (tmax > tmax_cap) and (tmin < tbase):
        return ((tmax_cap + tbase) / 2.0) - tbase
    return 0.0


def compute_gdd_columns(
    df: pd.DataFrame,
    date_col: str,
    tmin_col: str,
    tmax_col: str,
    tbase: float,
    tcap: float,
    planting_date: pd.Timestamp,
    harvest_date: Optional[pd.Timestamp],
    gdd_daily_col: str = "GDD (°C·d)",
    gdd_fp_daily_col: str = "FromPlanting_GDD (°C·d)",
    gdd_fp_cum_col: str = "ΣGDD (from planting)",
) -> pd.DataFrame:
    """Compute GDD metrics for a dataset."""
    out = df.copy()
    std_daily, fp_daily = [], []
    planting_dt = pd.to_datetime(planting_date)
    harvest_dt = pd.to_datetime(harvest_date) if harvest_date is not None else None

    for _, row in out.iterrows():
        g = gdd_excel_style(float(row[tmin_col]), float(row[tmax_col]), tbase, tcap)
        std_daily.append(round(g, 3))
        date_val = row[date_col]
        in_window = date_val >= planting_dt
        if harvest_dt is not None:
            in_window = in_window and date_val <= harvest_dt
        fp_daily.append(round(g, 3) if in_window else 0.0)

    out[gdd_daily_col] = std_daily
    out[gdd_fp_daily_col] = fp_daily
    out[gdd_fp_cum_col] = out[gdd_fp_daily_col].cumsum()
    return out


def calculate_irrigation_decision(
    recent_precip_mm: float,
    eto_mm: float,
    soil_moisture_percent: Optional[float] = None,
    growth_stage: str = "vegetative",
    cost_per_inch: float = 25.0,
) -> Dict[str, object]:
    """Decision engine for irrigation recommendations based on water balance."""
    effective_rain = calculate_effective_rainfall(recent_precip_mm)
    water_deficit = eto_mm - effective_rain

    stage_thresholds = {
        "vegetative": 15,
        "reproductive": 10,
        "grain_fill": 12,
        "maturity": 20,
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
        "soil_moisture_percent": soil_moisture_percent,
        "reasoning": reasons or ["Sufficient moisture from recent rainfall"],
    }


def generate_smart_alerts(forecast_df: pd.DataFrame) -> list[Dict[str, str]]:
    """Generate actionable alerts from forecast data."""
    alerts = []
    if forecast_df is None or forecast_df.empty:
        return alerts

    now = pd.Timestamp.now()
    next_24h = forecast_df[forecast_df["Datetime"].between(now, now + pd.Timedelta(hours=24))]
    next_48h = forecast_df[forecast_df["Datetime"].between(now, now + pd.Timedelta(hours=48))]

    if next_24h.empty:
        return alerts

    if (next_24h["Temp_Min"] <= 0).any():
        frost_time = next_24h[next_24h["Temp_Min"] <= 0]["Datetime"].min()
        alerts.append(
            {
                "type": "danger",
                "icon": "❄️",
                "title": "FROST RISK",
                "message": f"Freezing temps expected at {frost_time.strftime('%I%p %a')}",
                "action": "Delay irrigation. Protect sensitive crops.",
            }
        )

    if (next_24h["Temp_Max"] >= 35).any():
        alerts.append(
            {
                "type": "warn",
                "icon": "🔥",
                "title": "HEAT STRESS",
                "message": f"High temps up to {next_24h['Temp_Max'].max():.0f}°C expected",
                "action": "Increase irrigation frequency. Monitor crop stress.",
            }
        )

    if (next_48h["Wind"] >= 6.0).any():
        alerts.append(
            {
                "type": "warn",
                "icon": "💨",
                "title": "HIGH WIND",
                "message": f"Wind speeds up to {next_48h['Wind'].max():.1f} m/s",
                "action": "Avoid spraying. Risk of drift/evaporation.",
            }
        )

    if (next_24h["Rain_3h"] >= 10).any() or (next_24h["POP"] >= 70).any():
        total_rain = next_24h["Rain_3h"].sum()
        alerts.append(
            {
                "type": "info",
                "icon": "🌧️",
                "title": "HEAVY RAIN EXPECTED",
                "message": f"Up to {total_rain:.1f}mm in next 24h",
                "action": "Delay irrigation and field operations.",
            }
        )

    ideal_conditions = (
        (next_48h["Temperature"].between(10, 28))
        & (next_48h["Humidity"].between(40, 80))
        & (next_48h["Wind"] < 6.0)
        & (next_48h["POP"] < 20)
    )
    if ideal_conditions.any():
        ideal_df = next_48h[ideal_conditions]
        window_start = ideal_df["Datetime"].min()
        window_end = ideal_df["Datetime"].max()
        alerts.append(
            {
                "type": "success",
                "icon": "✅",
                "title": "SPRAY WINDOW",
                "message": f"Ideal conditions from {window_start.strftime('%I%p')} to {window_end.strftime('%I%p %a')}",
                "action": "Schedule spray applications during this window.",
            }
        )

    return alerts


def summarize_weather_period(
    df: pd.DataFrame,
    date_col: str,
    precip_col: str,
    tmax_col: str,
    tmin_col: str,
    eto_col: str,
) -> Dict[str, object]:
    """Summaries for precipitation, temperature, and ET."""
    if df.empty:
        return {}
    subset = df[[date_col, precip_col, tmax_col, tmin_col, eto_col]].copy()
    subset[date_col] = pd.to_datetime(subset[date_col])
    summary = {
        "date_range": {
            "start": str(subset[date_col].min().date()),
            "end": str(subset[date_col].max().date()),
        },
        "precip_mm": {
            "total": float(subset[precip_col].sum()),
            "mean": float(subset[precip_col].mean()),
            "max": float(subset[precip_col].max()),
            "pct_75": float(subset[precip_col].quantile(0.75)),
        },
        "temperature_c": {
            "tmax_mean": float(subset[tmax_col].mean()),
            "tmin_mean": float(subset[tmin_col].mean()),
            "tmax_max": float(subset[tmax_col].max()),
            "tmin_min": float(subset[tmin_col].min()),
        },
        "eto_mm": {
            "total": float(subset[eto_col].sum()) if eto_col in subset.columns else 0.0,
            "mean": float(subset[eto_col].mean()) if eto_col in subset.columns else 0.0,
        },
        "records": int(len(subset)),
    }
    return summary


def compute_water_balance(
    df: pd.DataFrame,
    date_col: str,
    precip_col: str,
    eto_col: str,
) -> pd.DataFrame:
    """Return dataframe with cumulative precip/ETo and balance."""
    if df.empty or precip_col not in df.columns or eto_col not in df.columns:
        return pd.DataFrame()
    water_balance = df[[date_col, precip_col, eto_col]].copy()
    water_balance["Cumulative_Precip"] = water_balance[precip_col].cumsum()
    water_balance["Cumulative_ETo"] = water_balance[eto_col].cumsum()
    water_balance["Balance"] = water_balance["Cumulative_Precip"] - water_balance["Cumulative_ETo"]
    return water_balance


def temperature_calendar(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Pivot for historical temperature calendar."""
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    temp_df = df[[date_col, value_col]].copy()
    temp_df["Month"] = temp_df[date_col].dt.month
    temp_df["DayOfMonth"] = temp_df[date_col].dt.day
    avg_daily = temp_df.groupby(["Month", "DayOfMonth"])[value_col].mean().reset_index()
    pivot = avg_daily.pivot_table(index="DayOfMonth", columns="Month", values=value_col)
    return pivot.sort_index(axis=1)


@st.cache_data(show_spinner=False)
def compute_et(
    df: pd.DataFrame,
    z_m: float,
    latitude_rad: float,
    rs_col: str,
    tmax_col: str,
    tmin_col: str,
    rhmax_col: str,
    rhmin_col: str,
    wind_col: str,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Calculate reference ET (ETo/ETr) for all records."""
    lat_deg = float(latitude_rad) * 180.0 / math.pi
    eto_vals, etr_vals = [], []
    for _, row in df.iterrows():
        eto = ascedaily(
            "S",
            z_m,
            lat_deg,
            int(row["DOY"]),
            row[rs_col],
            row[tmax_col],
            row[tmin_col],
            rhmax=row[rhmax_col],
            rhmin=row[rhmin_col],
            wndsp=row[wind_col],
            wndht=2.0,
        )
        etr = ascedaily(
            "T",
            z_m,
            lat_deg,
            int(row["DOY"]),
            row[rs_col],
            row[tmax_col],
            row[tmin_col],
            rhmax=row[rhmax_col],
            rhmin=row[rhmin_col],
            wndsp=row[wind_col],
            wndht=2.0,
        )
        eto_vals.append(eto)
        etr_vals.append(etr)
    out = df.copy()
    out["ETo (mm)"] = pd.to_numeric(eto_vals, errors="coerce")
    out["ETr (mm)"] = pd.to_numeric(etr_vals, errors="coerce")
    return out


def agg_df(
    df: pd.DataFrame,
    date_col: str,
    freq: str,
    cols_sum: Optional[Iterable[str]] = None,
    cols_mean: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Aggregate weather data over a period."""
    cols_sum = list(cols_sum or [])
    cols_mean = list(cols_mean or [])
    grouped = df.set_index(date_col).groupby(pd.Grouper(freq=freq))
    parts = []
    if cols_sum:
        parts.append(grouped[cols_sum].sum())
    if cols_mean:
        parts.append(grouped[cols_mean].mean())
    if parts:
        agg = pd.concat(parts, axis=1).reset_index()
        return agg
    return df.copy()
