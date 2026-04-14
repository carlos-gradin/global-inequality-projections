"""
fetch_wpp.py
------------
Downloads the UN World Population Prospects 2024 total-population file
(Medium variant, country level) and saves a compact Parquet with
(c3, year, population) for 2022..2100.

The app multiplies country populations by 0.01 per percentile when it
aggregates inequality across countries, so we store population in
*absolute people* (UN WPP reports in thousands; we multiply by 1000).

Run once:
    python scripts/fetch_wpp.py
"""

from pathlib import Path
import gzip
import io
import sys
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "app" / "data"
OUT_FILE     = OUT_DIR / "wpp_population.parquet"

WPP_URL = (
    "https://population.un.org/wpp/assets/Excel%20Files/"
    "1_Indicator%20(Standard)/CSV_FILES/"
    "WPP2024_TotalPopulationBySex.csv.gz"
)

YEAR_MIN, YEAR_MAX = 2022, 2100   # enough headroom past the 2050 default
VARIANT  = "Medium"
LOC_TYPE = "Country/Area"         # excludes World / region aggregates


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Download the gzipped CSV (~17 MB) and decompress in memory.
    # ------------------------------------------------------------------
    print(f"Downloading  {WPP_URL}")
    r = requests.get(WPP_URL, timeout=180)
    r.raise_for_status()
    raw = gzip.decompress(r.content)
    print(f"  decompressed: {len(raw)/1e6:.1f} MB")

    # ------------------------------------------------------------------
    # 2) Parse; keep only Medium variant, country level, years of interest.
    # ------------------------------------------------------------------
    df = pd.read_csv(
        io.BytesIO(raw),
        usecols=["ISO3_code", "LocTypeName", "Variant", "Time", "PopTotal"],
        dtype={"ISO3_code": "string", "LocTypeName": "string",
               "Variant": "string"},
        low_memory=False,
    )
    print(f"  raw rows: {len(df):,}")

    m = (
        (df["Variant"] == VARIANT)
        & (df["LocTypeName"] == LOC_TYPE)
        & (df["Time"].between(YEAR_MIN, YEAR_MAX))
        & df["ISO3_code"].notna()
    )
    d = df.loc[m, ["ISO3_code", "Time", "PopTotal"]].copy()

    # Convert UN WPP thousands -> absolute people.
    d["population"] = d["PopTotal"] * 1_000.0

    d = d.rename(columns={"ISO3_code": "c3", "Time": "year"})
    d = d[["c3", "year", "population"]].sort_values(["c3", "year"]).reset_index(drop=True)

    print(f"  kept rows: {len(d):,}   "
          f"({d['c3'].nunique()} countries, "
          f"years {d['year'].min()}..{d['year'].max()})")

    # ------------------------------------------------------------------
    # 3) Save as Parquet.
    # ------------------------------------------------------------------
    d.to_parquet(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE.relative_to(PROJECT_ROOT)}  "
          f"({OUT_FILE.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
