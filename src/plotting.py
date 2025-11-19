"""
Plotly figure constructors shared across Streamlit pages.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def apply_common_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply a shared layout baseline to Plotly figures."""
    layout = dict(template="plotly_white", legend=dict(orientation="h"))
    if title:
        layout["title"] = title
    fig.update_layout(**layout)
    return fig


def weather_trend_figure(
    df: pd.DataFrame,
    date_col: str,
    precip_col: str,
    value_cols: Sequence[str],
    cumulative: bool = False,
) -> go.Figure:
    """Overlay weather variables with precipitation on a secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = df[date_col]
    rain_series = df[precip_col] if precip_col in df.columns else None

    if rain_series is not None and rain_series.notna().any():
        fig.add_bar(
            name="Precip (mm)" if not cumulative else "Cumulative Precip (mm)",
            x=x,
            y=rain_series,
            opacity=0.55,
            secondary_y=True,
        )

    def _finite_bounds(series_list: Iterable[pd.Series]):
        vals = pd.concat(series_list, axis=0) if series_list else pd.Series(dtype=float)
        vals = pd.to_numeric(vals, errors="coerce")
        vals = vals[pd.notna(vals)]
        if vals.empty:
            return None, None
        return float(vals.min()), float(vals.max())

    primary_series = []
    for col in value_cols:
        if col not in df.columns:
            continue
        series = df[col]
        primary_series.append(series)
        fig.add_trace(
            go.Scatter(name=col, x=x, y=series, mode="lines+markers", line=dict(width=2), marker=dict(size=5)),
            secondary_y=False,
        )

    ymin, ymax = _finite_bounds(primary_series)
    if ymin is not None and ymax is not None and ymin != ymax:
        pad = 0.05 * (ymax - ymin)
        fig.update_yaxes(range=[ymin - pad, ymax + pad], secondary_y=False)

    if rain_series is not None and rain_series.notna().any():
        rmin, rmax = _finite_bounds([rain_series])
        if rmax is not None:
            pad2 = 0.1 * max(1.0, (rmax - (rmin or 0)))
            fig.update_yaxes(range=[0, rmax + pad2], secondary_y=True)

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Value", secondary_y=False)
    fig.update_yaxes(
        title_text="Precip / Cumulative (mm)" if cumulative else "Precip (mm)",
        secondary_y=True,
    )
    return fig


def water_balance_figure(df: pd.DataFrame, date_col: str) -> go.Figure:
    """Plot cumulative precipitation vs ETo along with balance."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df["Cumulative_Precip"],
            mode="lines",
            name="Cumulative Precip",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df["Cumulative_ETo"],
            mode="lines",
            name="Cumulative ETo",
            line=dict(color="orange", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df["Balance"],
            mode="lines",
            name="Balance (Precip - ETo)",
            line=dict(color="green", width=2, dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(0,255,0,0.08)",
        )
    )
    fig.update_layout(height=350, template="plotly_white", legend=dict(orientation="h"), yaxis_title="Cumulative (mm)")
    fig.update_xaxes(title_text="Date")
    return fig


def temperature_calendar_heatmap(pivot: pd.DataFrame, title: str) -> go.Figure:
    """Heatmap for historical daily averages."""
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x_labels = [month_names[i - 1] for i in pivot.columns]
    fig = px.imshow(
        pivot,
        x=x_labels,
        y=pivot.index,
        color_continuous_scale=px.colors.sequential.thermal,
        aspect="auto",
        title=title,
        labels=dict(x="Month", y="Day of Month", color="Avg Temp (°C)"),
    )
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10), title_x=0.5)
    fig.update_traces(hovertemplate="<b>Month:</b> %{x}<br><b>Day:</b> %{y}<br><b>Avg Temp:</b> %{z:.1f}°C<extra></extra>")
    fig.update_yaxes(autorange="reversed")
    return fig


def gdd_daily_bar(df: pd.DataFrame, date_col: str, gdd_col: str) -> go.Figure:
    fig = px.bar(df, x=date_col, y=gdd_col, title="Daily GDD (°C·day)", template="plotly_white")
    fig.update_traces(marker_color="lightgreen")
    fig.update_layout(height=360)
    return fig


def gdd_cumulative_line(df: pd.DataFrame, date_col: str, gdd_col: str) -> go.Figure:
    fig = px.line(df, x=date_col, y=gdd_col, title="ΣGDD from planting (°C·day)", markers=True, template="plotly_white")
    fig.update_traces(line_color="darkgreen")
    fig.update_layout(height=360)
    return fig


def rainfall_bar_chart(df: pd.DataFrame, date_col: str, precip_col: str, rolling_days: int = 7) -> go.Figure:
    """Daily rainfall hyetograph with optional rolling mean overlay."""
    if df.empty or precip_col not in df.columns:
        return go.Figure()
    fig = px.bar(
        df,
        x=date_col,
        y=precip_col,
        title="Daily Rainfall",
        labels={date_col: "Date", precip_col: "Rain (mm)"},
        template="plotly_white",
    )
    fig.update_traces(marker_color="#4f8bc9", opacity=0.75)
    if rolling_days and rolling_days > 1:
        rolling = df[[date_col, precip_col]].copy()
        rolling_sorted = rolling.sort_values(date_col)
        rolling_sorted["Rolling"] = rolling_sorted[precip_col].rolling(rolling_days, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=rolling_sorted[date_col],
                y=rolling_sorted["Rolling"],
                name=f"{rolling_days}-day avg",
                mode="lines",
                line=dict(color="#c44536", width=2),
            )
        )
    fig.update_layout(height=360, bargap=0.15)
    return fig


def et_reference_figure(df: pd.DataFrame, date_col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df["ETo (mm)"], mode="lines+markers", name="ETo (short reference)"))
    fig.add_trace(go.Scatter(x=df[date_col], y=df["ETr (mm)"], mode="lines+markers", name="ETr (tall reference)"))
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
        template="plotly_white",
        yaxis_title="ET (mm)",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Date")
    return fig


def crop_time_series_figure(
    ts: pd.DataFrame,
    date_col: str,
    value_col: str,
    entity_col: str,
    selections: Sequence[str],
    palette: dict[str, str],
    nitrogen_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Plot NDVI/MCARI2 time series for plots or treatments with optional nitrogen overlay."""
    fig = go.Figure()
    for sid in selections:
        sid = str(sid)
        subset = ts[ts[entity_col] == sid].sort_values(date_col)
        color = palette.get(sid, "#3778c2")
        fig.add_trace(
            go.Scatter(x=subset[date_col], y=subset[value_col], mode="lines+markers", name=sid, line=dict(color=color))
        )
        if nitrogen_df is not None:
            nitrogen_subset = nitrogen_df[nitrogen_df["TRT_ID"] == sid]
            if not nitrogen_subset.empty:
                fig.add_trace(
                    go.Bar(
                        x=nitrogen_subset["Date"],
                        y=nitrogen_subset["Amount"],
                        name=f"N (TRT {sid})",
                        marker_color=color,
                        opacity=0.35,
                        yaxis="y2",
                    )
                )

    fig.update_layout(
        title="Index over time",
        xaxis_title="Date",
        yaxis=dict(title=value_col, side="left"),
        legend=dict(orientation="h"),
        height=450,
        template="plotly_white",
        barmode="overlay",
    )
    fig.update_layout(yaxis2=dict(title="N amount (lbs/ac)", overlaying="y", side="right", showgrid=False))
    return fig


def forecast_daily_temp_band(dd: pd.DataFrame, temp_unit: str) -> go.Figure:
    """Temperature band chart for a single day's forecast."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd["Datetime"],
            y=dd["Temp_Min"],
            mode="lines",
            line=dict(width=0),
            name="Min",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dd["Datetime"],
            y=dd["Temp_Max"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255,165,0,0.15)",
            line=dict(width=0),
            name="Max",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dd["Datetime"],
            y=dd["Temperature"],
            mode="lines+markers",
            name=f"Temp ({temp_unit})",
            line=dict(width=2),
        )
    )
    fig.update_layout(
        height=340,
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h"),
    )
    return fig
