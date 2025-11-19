"""
AI-facing helpers for weather analytics.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Optional

import pandas as pd

from . import weather_calcs


def get_weather_summary_payload(
    df: pd.DataFrame,
    date_col: str,
    precip_col: str,
    tmax_col: str,
    tmin_col: str,
    eto_col: str,
) -> Dict[str, object]:
    """High-level stats for AI JSON payloads."""
    return weather_calcs.summarize_weather_period(df, date_col, precip_col, tmax_col, tmin_col, eto_col)


def get_gdd_status(
    df: pd.DataFrame,
    date_col: str,
    gdd_col: str,
    planting_date: dt.date,
    target: Optional[float] = None,
) -> Dict[str, object]:
    """Return cumulative GDD and relative progress."""
    if df.empty or gdd_col not in df.columns:
        return {}
    df_sorted = df.sort_values(date_col)
    current = float(df_sorted[gdd_col].iloc[-1])
    days_since = (df_sorted[date_col].iloc[-1] - pd.Timestamp(planting_date)).days
    progress = {
        "current_gdd": current,
        "days_since_planting": int(days_since),
        "target_gdd": target,
        "percent_complete": round((current / target * 100.0), 2) if target else None,
    }
    return progress


def get_irrigation_recommendation(
    df: pd.DataFrame,
    date_col: str,
    precip_col: str,
    eto_col: str,
    recent_days: int,
    growth_stage: str,
    cost_per_inch: float,
) -> Dict[str, object]:
    """Use recent precip/ET totals to recommend irrigation."""
    if df.empty or precip_col not in df.columns or eto_col not in df.columns:
        return {}
    cutoff = df[date_col].max() - pd.Timedelta(days=recent_days - 1)
    recent = df[df[date_col] >= cutoff]
    precip_mm = float(recent[precip_col].sum())
    eto_mm = float(recent[eto_col].sum())
    decision = weather_calcs.calculate_irrigation_decision(
        recent_precip_mm=precip_mm,
        eto_mm=eto_mm,
        growth_stage=growth_stage,
        cost_per_inch=cost_per_inch,
    )
    decision["period_days"] = recent_days
    decision["period_start"] = str(cutoff.date())
    decision["period_end"] = str(df[date_col].max().date())
    return decision


def get_eto_for_period(
    df: pd.DataFrame,
    date_col: str,
    eto_col: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, object]:
    """Summarize ETo for a specified window."""
    if df.empty or eto_col not in df.columns:
        return {}
    data = df[[date_col, eto_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    start_dt = pd.to_datetime(start) if start else data[date_col].min()
    end_dt = pd.to_datetime(end) if end else data[date_col].max()
    window = data[data[date_col].between(start_dt, end_dt)]
    if window.empty:
        return {}
    result = {
        "date_range": {"start": str(start_dt.date()), "end": str(end_dt.date())},
        "records": int(len(window)),
        "eto_mm": {
            "total": float(window[eto_col].sum()),
            "mean": float(window[eto_col].mean()),
            "max": float(window[eto_col].max()),
            "min": float(window[eto_col].min()),
        },
    }
    return result


def get_forecast_summary(forecast_df: pd.DataFrame) -> Dict[str, object]:
    """Aggregate upcoming forecast data."""
    if forecast_df is None or forecast_df.empty:
        return {}
    next_48 = forecast_df.head(16).copy()
    summary = {
        "horizon_hours": 48,
        "temp_min": float(next_48["Temp_Min"].min()),
        "temp_max": float(next_48["Temp_Max"].max()),
        "rain_total_mm": float(next_48["Rain_3h"].sum()),
        "avg_wind_ms": float(next_48["Wind"].mean()),
    }
    return summary


def get_alerts(forecast_df: pd.DataFrame):
    return weather_calcs.generate_smart_alerts(forecast_df)


def get_smart_alerts(forecast_df: pd.DataFrame) -> Dict[str, object]:
    """Wrapper returning smart alerts for AI consumption."""
    alerts = weather_calcs.generate_smart_alerts(forecast_df)
    return {"alerts": alerts or []}


WEATHER_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_gdd_status",
            "description": "Return cumulative GDD progress relative to planting and optional targets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "number", "description": "Optional target GDD for the season."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_irrigation_recommendation",
            "description": "Summarize irrigation needs over a recent window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recent_days": {"type": "integer", "default": 7},
                    "growth_stage": {"type": "string"},
                    "cost_per_inch": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_eto_for_period",
            "description": "Summarize ETo totals/means for a selected date window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast_summary",
            "description": "Summarize the uploaded/queried forecast window.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_smart_alerts",
            "description": "Return hazard alerts derived from the forecast.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def execute_weather_tool(
    tool_name: str,
    args: Dict,
    df: pd.DataFrame,
    context: Dict[str, object],
    forecast_df: pd.DataFrame,
) -> Dict[str, object]:
    """Execute tool functions with shared context."""
    if tool_name == "get_gdd_status":
        return get_gdd_status(
            df=df,
            date_col=context["date_col"],
            gdd_col=context["gdd_col"],
            planting_date=context["planting_date"],
            target=args.get("target"),
        )
    if tool_name == "get_irrigation_recommendation":
        return get_irrigation_recommendation(
            df=df,
            date_col=context["date_col"],
            precip_col=context["precip_col"],
            eto_col=context["eto_col"],
            recent_days=args.get("recent_days", 7),
            growth_stage=args.get("growth_stage", context.get("growth_stage", "vegetative")),
            cost_per_inch=args.get("cost_per_inch", context.get("cost_per_inch", 25.0)),
        )
    if tool_name == "get_eto_for_period":
        return get_eto_for_period(
            df=df,
            date_col=context["date_col"],
            eto_col=context["eto_col"],
            start=args.get("start"),
            end=args.get("end"),
        )
    if tool_name == "get_forecast_summary":
        return get_forecast_summary(forecast_df)
    if tool_name == "get_smart_alerts":
        return get_smart_alerts(forecast_df)
    return {"error": f"Unknown tool: {tool_name}"}
