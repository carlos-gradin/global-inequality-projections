"""
build_baseline.py
-----------------
Produces two small Parquet files that the Streamlit app loads at startup:

    app/data/baseline_2022.parquet   <- 2022 country x percentile income levels/shares
    app/data/regions.parquet         <- ISO3 -> region_wb, incomegroup, is_china, is_india

Input:
    wiidglobal_long.dta  (in the project root; ~248 MB, long format with
                          100 percentile rows per country-year).

Why pre-compute?
    - The raw .dta is too big to ship with the app.
    - Streamlit Cloud deployments have repository size limits; we only need
      the 2022 cross-section (~21k rows, <2 MB as Parquet).
    - Loading Parquet in the app is near-instant.

Run once from the project root:
    python scripts/build_baseline.py
"""

from pathlib import Path
import sys
import pyreadstat
import pandas as pd

# ----------------------------------------------------------------------
# Paths (resolved relative to the script, so it works from any cwd)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DTA      = PROJECT_ROOT / "wiidglobal_long.dta"
OUT_DIR      = PROJECT_ROOT / "app" / "data"
OUT_BASELINE = OUT_DIR / "baseline_2022.parquet"
OUT_REGIONS  = OUT_DIR / "regions.parquet"

# Only the columns we need from the raw dataset.
KEEP_COLS = [
    "country", "c3", "year", "percentile",
    "region_wb", "incomegroup",
    "population", "gdp",
    "income_share", "income_level",
    "interpolated",
]

BASELINE_YEAR = 2022


def main() -> None:
    if not SRC_DTA.exists():
        sys.exit(f"ERROR: cannot find input file: {SRC_DTA}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load only needed columns from the .dta (saves RAM).
    # ------------------------------------------------------------------
    print(f"Reading {SRC_DTA.name} ...")
    df, meta = pyreadstat.read_dta(str(SRC_DTA), usecols=KEEP_COLS)
    print(f"  loaded {len(df):,} rows")

    # The raw file stores region_wb and incomegroup as numeric codes.
    # Map them to their human-readable string labels now so the app can
    # display / filter on them without another lookup step.
    region_labels      = meta.variable_value_labels.get("region_wb", {})
    incomegroup_labels = meta.variable_value_labels.get("incomegroup", {})
    if region_labels:
        df["region_wb"] = df["region_wb"].map(region_labels)
    if incomegroup_labels:
        df["incomegroup"] = df["incomegroup"].map(incomegroup_labels)

    # ------------------------------------------------------------------
    # 2) Keep the 2022 slice and drop the pre-aggregated regional rows
    #    (those have an empty c3 code; we will aggregate regions ourselves
    #    using country-level rows + weights).
    # ------------------------------------------------------------------
    d = df[(df["year"] == BASELINE_YEAR) & (df["c3"].astype(str).str.len() == 3)].copy()
    print(f"  {BASELINE_YEAR} country rows: {len(d):,} "
          f"({d['c3'].nunique()} countries)")

    # Sanity: every country should have exactly 100 percentile rows.
    rows_per_country = d.groupby("c3").size()
    bad = rows_per_country[rows_per_country != 100]
    if len(bad) > 0:
        print(f"  WARNING: {len(bad)} countries do not have 100 percentiles:")
        print(bad)

    # ------------------------------------------------------------------
    # 3) Build the country-level 'regions' table (one row per country).
    #    These attributes do not depend on the percentile.
    # ------------------------------------------------------------------
    regions = (
        d[["c3", "country", "region_wb", "incomegroup", "population", "gdp",
           "interpolated"]]
        .drop_duplicates("c3")
        .reset_index(drop=True)
    )
    # Flags for giants (for the optional CN/IN split in Mode B).
    regions["is_china"] = (regions["c3"] == "CHN").astype(int)
    regions["is_india"] = (regions["c3"] == "IND").astype(int)

    # ------------------------------------------------------------------
    # 4) Keep only the per-percentile columns needed for projection.
    # ------------------------------------------------------------------
    baseline = d[["c3", "percentile", "income_share", "income_level"]].copy()

    # Ensure percentile is a small int (compact on disk).
    baseline["percentile"] = baseline["percentile"].astype("int16")

    baseline.sort_values(["c3", "percentile"], inplace=True, ignore_index=True)

    # ------------------------------------------------------------------
    # 5) Write the Parquet outputs.
    # ------------------------------------------------------------------
    baseline.to_parquet(OUT_BASELINE, index=False)
    regions.to_parquet(OUT_REGIONS, index=False)

    print(f"\nWrote:")
    print(f"  {OUT_BASELINE.relative_to(PROJECT_ROOT)}  "
          f"({OUT_BASELINE.stat().st_size/1024:.0f} KB,  {len(baseline):,} rows)")
    print(f"  {OUT_REGIONS.relative_to(PROJECT_ROOT)}  "
          f"({OUT_REGIONS.stat().st_size/1024:.0f} KB,  {len(regions):,} rows)")


if __name__ == "__main__":
    main()
