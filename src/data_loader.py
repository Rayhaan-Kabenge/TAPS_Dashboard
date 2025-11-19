from __future__ import annotations

import io
import os
import re
import tempfile
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats


FNAME_RE = re.compile(r"^(?P<index>[A-Za-z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$", re.IGNORECASE)


def _standardize_source(source):
    """Return a path-like or BytesIO for pandas/geopandas readers."""
    if source is None:
        raise ValueError("A file path or uploaded file must be provided.")
    if isinstance(source, (str, Path)):
        return source
    if hasattr(source, "read"):
        data = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        return io.BytesIO(data)
    raise ValueError("Unsupported source type. Provide a path or file-like object.")


def _write_temp_file(data: bytes, suffix: str = "") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def _extract_zip_to_tempdir(zip_bytes: bytes) -> str:
    tmpdir = tempfile.mkdtemp()
    with ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(tmpdir)
    return tmpdir


def load_climate(
    path_or_file,
    sheet: str,
    required_columns: Iterable[str],
    date_column: str,
    numeric_columns: Iterable[str],
) -> pd.DataFrame:
    """Load and normalize climate data from Excel."""
    df = pd.read_excel(_standardize_source(path_or_file), sheet_name=sheet)
    df.rename(columns={c: str(c).strip().replace("\u00A0", " ") for c in df.columns}, inplace=True)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in `{sheet}`: {missing}")

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(numeric_columns))
    df["DOY"] = df[date_column].dt.dayofyear
    return df.sort_values(date_column).reset_index(drop=True)


def list_tifs(directory: str) -> list[str]:
    """Return sorted .tif files from a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(glob(os.path.join(directory, "*.tif")))


def build_image_catalog(directory: str) -> Dict[str, Dict[datetime.date, str]]:
    """Build a nested {index: {date: path}} catalog for imagery."""
    catalog: Dict[str, Dict[datetime.date, str]] = {}
    for path in list_tifs(directory):
        fn = os.path.basename(path)
        match = FNAME_RE.match(fn)
        if match:
            idx = match.group("index").upper()
            date_str = match.group("date")
        else:
            parts = fn.split("_", 1)
            if len(parts) != 2:
                continue
            idx = parts[0].upper()
            date_str = os.path.splitext(parts[1])[0]
        try:
            dt_val = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue
        catalog.setdefault(idx, {})[dt_val] = path
    for key in list(catalog.keys()):
        catalog[key] = dict(sorted(catalog[key].items(), key=lambda item: item[0]))
    return catalog


def build_image_catalog_from_archive(zip_file) -> Dict[str, Dict[datetime.date, str]]:
    """Build image catalog from a ZIP upload."""
    if zip_file is None:
        raise ValueError("Imagery archive is required.")
    if hasattr(zip_file, "read"):
        zip_bytes = zip_file.read()
        zip_file.seek(0)
    else:
        with open(zip_file, "rb") as f:
            zip_bytes = f.read()
    directory = _extract_zip_to_tempdir(zip_bytes)
    return build_image_catalog(directory)


def load_management_n(xlsx_path, sheet: str) -> pd.DataFrame:
    """Parse nitrogen management schedule."""
    try:
        df = pd.read_excel(_standardize_source(xlsx_path), sheet_name=sheet)
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

    for col in df.columns:
        lc = col.lower()
        if lc == "trt_id":
            continue
        if "planting date amount" in lc:
            planting_amt_col = col
        elif "planting date" in lc:
            planting_date_col = col
        elif "lbs" in lc:
            continue
        elif date_hdr_re.match(col):
            date_like_cols.append(col)

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
        df_dates = df.melt(
            id_vars=["TRT_ID"],
            value_vars=date_like_cols,
            var_name="Date",
            value_name="Amount",
        )
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


def load_zonal_stats(file_obj, file_type: str = "excel") -> pd.DataFrame:
    """Load precomputed zonal statistics from CSV or Excel."""
    buffer = _standardize_source(file_obj)
    if file_type == "csv":
        df = pd.read_csv(buffer)
    else:
        df = pd.read_excel(buffer)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    return df


def load_ghg_data(file_path) -> pd.DataFrame:
    """Load greenhouse gas flux data."""
    try:
        buffer = _standardize_source(file_path)
    except ValueError:
        return pd.DataFrame()
    try:
        df = pd.read_excel(buffer)
    except Exception:
        return pd.DataFrame()
    df["TRT_ID"] = df["TRT_ID"].ffill()
    id_vars = ["TRT_ID", "Plot"]
    date_cols = [col for col in df.columns if col not in id_vars]
    df_long = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=date_cols,
        var_name="Date",
        value_name="N2O_Flux",
    )
    df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce", dayfirst=True)
    df_long.dropna(subset=["Date", "N2O_Flux"], inplace=True)
    df_long["TRT_ID"] = df_long["TRT_ID"].astype(str).str.replace("T", "", regex=False)
    df_long["Date"] = df_long["Date"].dt.date
    return df_long


def read_plots(shp_path) -> gpd.GeoDataFrame:
    """Load plot polygons and normalize ID columns."""
    source = _standardize_source(shp_path)
    if isinstance(source, io.BytesIO):
        zip_bytes = source.getvalue()
        tmp_path = _write_temp_file(zip_bytes, suffix=".zip")
        gdf = gpd.read_file(f"zip://{tmp_path}")
    else:
        gdf = gpd.read_file(source)
    if "Plot_ID" not in gdf.columns:
        candidates = [c for c in gdf.columns if c.lower() in ("plot_id", "plotid", "plot", "name", "id")]
        gdf["Plot_ID"] = gdf[candidates[0]] if candidates else np.arange(len(gdf)).astype(str)
    if "TRT_ID" not in gdf.columns:
        gdf["TRT_ID"] = "N/A"
    gdf["Plot_ID"] = gdf["Plot_ID"].astype(str)
    gdf["TRT_ID"] = gdf["TRT_ID"].astype(str)
    return gdf


def calculate_all_zonal_stats(catalog: Dict[str, Dict[datetime.date, str]], plots_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute mean zonal stats for every index/date combination."""
    if catalog is None or not catalog or plots_gdf is None or plots_gdf.empty:
        return pd.DataFrame()

    all_stats = []
    for index_name, date_dict in catalog.items():
        for date_val, img_path in date_dict.items():
            try:
                with rasterio.open(img_path) as src:
                    arr = src.read(1).astype("float32")
                    arr[~np.isfinite(arr)] = np.nan

                    plots_native = plots_gdf.to_crs(src.crs)
                    zs = zonal_stats(
                        plots_native,
                        arr,
                        affine=src.transform,
                        stats=["mean"],
                        nodata=np.nan,
                        all_touched=False,
                    )

                    for idx, stat_dict in enumerate(zs):
                        all_stats.append(
                            {
                                "Plot_ID": plots_gdf.iloc[idx]["Plot_ID"],
                                "TRT_ID": plots_gdf.iloc[idx]["TRT_ID"],
                                "Index": index_name,
                                "Date": date_val,
                                "Mean": stat_dict.get("mean", np.nan),
                            }
                        )
            except Exception:
                continue

    if not all_stats:
        return pd.DataFrame()

    df = pd.DataFrame(all_stats)
    df["Plot_ID"] = df["Plot_ID"].astype(str)
    df["TRT_ID"] = df["TRT_ID"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    return df


def load_team_metadata(file_obj) -> pd.DataFrame:
    """Generic loader for team configuration tables."""
    buffer = _standardize_source(file_obj)
    df = pd.read_excel(buffer)
    return df


def load_climate_with_et(
    path_or_file,
    sheet: str,
    required_columns: Iterable[str],
    date_column: str,
    numeric_columns: Iterable[str],
    elevation_m: float,
    latitude_rad: float,
    rs_col: str,
    tmax_col: str,
    tmin_col: str,
    rhmax_col: str,
    rhmin_col: str,
    wind_col: str,
):
    """
    Convenience wrapper that loads a climate sheet and computes ETo/ETr.
    """
    from . import weather_calcs  # local import to avoid circular dependency

    df = load_climate(
        path_or_file,
        sheet,
        required_columns=required_columns,
        date_column=date_column,
        numeric_columns=numeric_columns,
    )
    df_et = weather_calcs.compute_et(
        df,
        elevation_m,
        latitude_rad,
        rs_col=rs_col,
        tmax_col=tmax_col,
        tmin_col=tmin_col,
        rhmax_col=rhmax_col,
        rhmin_col=rhmin_col,
        wind_col=wind_col,
        date_col=date_column,
    )
    return df_et
