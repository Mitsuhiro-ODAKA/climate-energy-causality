# src/data_utils.py
"""
Data-acquisition helpers for the Heat-Health-Income mini-study.
Usage:
    from src.data_utils import download_era5, fetch_malaria_pr
"""

import os
from pathlib import Path
from typing import List
import requests
import cdsapi
import pandas as pd
import xarray as xr
ERA_DIR = Path("../data/raw/era5")

# ---------------------------------------------------------------------
# ERA5 Single-level monthly average variables (temperature, relative humidity, etc.)
# ---------------------------------------------------------------------

def download_era5(
    years: List[int],
    area: List[float],
    variables: List[str],
    out_dir: str = "../data/raw/era5",
) -> List[Path]:

    out_paths = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for yr in years:
        if variables == ["2m_temperature"]:
            target = Path(out_dir, f"era5_t2m_{yr}.nc")
        elif variables == ["2m_relative_humidity"]:
            target = Path(out_dir, f"era5_rh2m_{yr}.nc")
        else:
            pass
        if target.exists():
            print(f"ERA5 {yr} already exists, skip.")
            out_paths.append(target)
            continue

        print(f"Requesting ERA5 for {yr} ...")
        client.retrieve(
            "/reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "data_format": "netcdf",
                "download_format": ["unarchived"],
                "variable": variables,
                "year": [str(yr)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "time":  ["00:00"],          # daily snapshot (UTC)
                "area": area,              # [N,W,S,E]
                "grid": [0.25, 0.25]
            },
            str(target),
        )
        out_paths.append(target)

    return out_paths

# ---------------------------------------------------------------------
#  World-Bank WDI 汎用ダウンローダ
# ---------------------------------------------------------------------
import requests, pandas as pd
from pathlib import Path

def fetch_wdi_indicator(
    iso: str,
    indicator: str,
    start_year: int = 2000,
    end_year: int = 2023,
    out_path: str | None = None,
) -> Path:

    if out_path is None:
        out_path = f"data/raw/wdi_{indicator.lower()}.csv"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    url = (
        f"https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}"
        "?format=json&per_page=6000"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    meta, vals = r.json()
    df = (
        pd.DataFrame(vals)[["date", "value"]]
        .rename(columns={"date": "year", "value": indicator})
        .dropna()
        .astype({"year": int})
        .query("year >= @start_year and year <= @end_year")
        .sort_values("year")
        .reset_index(drop=True)
    )
    df.to_csv(out_path, index=False)
    print(f"✓ WDI {indicator} saved → {out_path}")
    return Path(out_path)
