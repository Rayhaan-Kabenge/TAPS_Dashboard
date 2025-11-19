from __future__ import annotations

import datetime as dt
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure

import streamlit as st

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - geopandas is optional for some commands
    gpd = None  # type: ignore


def json_default(obj: Any) -> Any:
    """JSON serializer that gracefully handles numpy/pandas objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return str(obj)


def figure_to_png_bytes(fig: Figure, scale: int = 2) -> Optional[bytes]:
    """Convert a Plotly figure into PNG bytes for downloading."""
    buf = io.BytesIO()
    try:
        fig.write_image(buf, format="png", scale=scale)
        return buf.getvalue()
    except Exception:
        return None


def download_button_for_figure(fig: Figure, filename: str = "chart.png", label: str = "⬇️ Download PNG"):
    """Render a Streamlit download button for a Plotly figure."""
    data = figure_to_png_bytes(fig)
    if data is None:
        st.caption("PNG export needs `kaleido` (`pip install -U kaleido`).")
        return
    st.download_button(label, data, file_name=filename, mime="image/png")


def value_to_css(value: Union[float, int, None], vmin: float, vmax: float, alpha: float = 0.9) -> str:
    """Return an RGBA string (red→yellow→green) for a numeric value."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "rgba(0,0,0,0)"
    span = max(vmax - vmin, 1e-9)
    t = np.clip((float(value) - vmin) / span, 0.0, 1.0)
    if t < 0.5:
        r = 255
        g = int(2.0 * t * 255)
    else:
        r = int((1.0 - 2.0 * (t - 0.5)) * 255)
        g = 255
    opacity = np.clip(alpha, 0.0, 1.0)
    return f"rgba({r},{g},0,{opacity:.2f})"


def colorize_ryg(array: np.ndarray, vmin: float, vmax: float, alpha: int = 220) -> np.ndarray:
    """Colorize a raster array with a red→yellow→green gradient."""
    arr = array.astype("float32", copy=False)
    mask = ~np.isfinite(arr)
    span = max(vmax - vmin, 1e-9)
    t = np.clip((arr - vmin) / span, 0.0, 1.0)
    r = np.where(t < 0.5, 255.0, (1.0 - 2.0 * (t - 0.5)) * 255.0)
    g = np.where(t < 0.5, (2.0 * t) * 255.0, 255.0)
    b = np.zeros_like(r)
    rgba = np.dstack([
        np.clip(r, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        b.astype(np.uint8),
        np.where(mask, 0, np.clip(alpha, 0, 255)).astype(np.uint8),
    ])
    return rgba


def safe_division(num: Union[float, int], den: Union[float, int], default: float = 0.0) -> float:
    """Division helper that avoids ZeroDivisionError and non-finite values."""
    try:
        if den and np.isfinite(den) and float(den) != 0.0:
            return float(num) / float(den)
    except Exception:
        pass
    return default


def safe_percent(num: Union[float, int], den: Union[float, int]) -> float:
    """Return a percentage (0-100) with graceful fallback."""
    return round(safe_division(num, den, 0.0) * 100.0, 1)


def coerce_date(value: Any) -> Optional[dt.date]:
    """Convert arbitrary values to python date objects when possible."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return pd.to_datetime(value, errors="coerce").date()
    except Exception:
        return None


def index_dates_for(master_zonal_stats: pd.DataFrame, index_name: str) -> List[dt.date]:
    """Return sorted unique dates for a given index."""
    if master_zonal_stats is None or master_zonal_stats.empty:
        return []
    df = master_zonal_stats[master_zonal_stats["Index"] == index_name]
    if df.empty:
        return []
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna().dt.date.unique()
    return sorted(dates)


def compute_treatment_percentile(
    master_zonal_stats: pd.DataFrame,
    index_name: str,
    date: Optional[Union[str, dt.date]] = None,
    metric: str = "mean",
    trt_id: Optional[str] = None,
    higher_is_better: bool = True,
    decimals: int = 3,
) -> Dict[str, Any]:
    """Compute percentile ranking for treatments on a specific date."""
    if master_zonal_stats is None or master_zonal_stats.empty:
        return {"error": "master_zonal_stats is empty."}

    df = master_zonal_stats[master_zonal_stats["Index"].str.upper() == index_name.upper()].copy()
    if df.empty:
        return {"error": f"No records for index '{index_name}'."}

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    date_used = coerce_date(date) or (df["Date"].dropna().max() if not df["Date"].dropna().empty else None)
    if date_used is None:
        return {"error": f"No valid dates found for index '{index_name}'."}

    df = df[df["Date"] == date_used]
    if df.empty:
        return {"error": f"No data for index '{index_name}' on {date_used}."}

    agg_map = {
        "mean": lambda s: float(np.nanmean(s.values)) if len(s) else np.nan,
        "median": lambda s: float(np.nanmedian(s.values)) if len(s) else np.nan,
        "max": lambda s: float(np.nanmax(s.values)) if len(s) else np.nan,
    }
    if metric not in agg_map:
        return {"error": f"Unsupported metric '{metric}'. Use one of: mean, median, max."}

    grouped = (
        df.groupby("TRT_ID", as_index=False)["Mean"]
        .apply(agg_map[metric])
        .rename(columns={"Mean": "value"})
    )
    grouped["TRT_ID"] = grouped["TRT_ID"].astype(str)
    grouped = grouped[np.isfinite(grouped["value"])].copy()
    if grouped.empty:
        return {"error": "No finite treatment values to rank."}

    grouped["rank"] = grouped["value"].rank(ascending=not higher_is_better, method="min").astype(int)
    grouped["percentile"] = grouped["value"].rank(pct=True, ascending=not higher_is_better) * 100.0
    grouped["percentile"] = grouped["percentile"].round(1)
    grouped["value"] = grouped["value"].round(decimals)
    grouped = grouped.sort_values("value", ascending=not higher_is_better).reset_index(drop=True)

    result: Dict[str, Any] = {
        "index_name": index_name.upper(),
        "date": date_used,
        "metric": metric,
        "n_treatments": int(len(grouped)),
        "table": grouped.to_dict(orient="records"),
    }

    if trt_id is not None:
        trt_str = str(trt_id)
        row = grouped[grouped["TRT_ID"] == trt_str]
        if row.empty:
            result["warning"] = f"Treatment '{trt_str}' not present on {date_used}."
        else:
            row = row.iloc[0]
            result.update(
                {
                    "trt_id": trt_str,
                    "value": float(row["value"]),
                    "rank": int(row["rank"]),
                    "percentile": float(row["percentile"]),
                }
            )
    return result


def center_from_bounds(
    bounds: Optional[Sequence[float]],
    fallback_gdf: Optional["gpd.GeoDataFrame"],
) -> Tuple[float, float]:
    """Return (lat, lon) for a bounding box or fall back to a GeoDataFrame extent."""
    if bounds is not None:
        try:
            left, bottom, right, top = bounds
            if np.all(np.isfinite([left, bottom, right, top])):
                return (float((top + bottom) / 2.0), float((left + right) / 2.0))
        except Exception:
            pass
    if fallback_gdf is not None:
        minx, miny, maxx, maxy = fallback_gdf.geometry.total_bounds
        return (float((miny + maxy) / 2.0), float((minx + maxx) / 2.0))
    return (0.0, 0.0)
