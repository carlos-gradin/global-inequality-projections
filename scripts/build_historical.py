"""
build_historical.py
-------------------
Extracts a compact historical panel of per-percentile income levels from
the WIID Companion dataset (wiidglobal_long.dta) for 2000-2022.

The WIID file is a perfectly balanced panel (survey-year gaps are
interpolated and post-last-survey years are extrapolated upstream, by the
WIID compilers), so every (c3, year, percentile) cell has a value.  That
lets the app compute a benchmark "historical continuation" scenario:
for each (country, percentile) the compound annual growth rate over the
last N years (default 10) is projected forward from 2022.

Output:
    app/data/historical_2000_2022.parquet
        columns: c3, year, percentile (1..100), income_level
        rows:    211 countries x 23 years x 100 percentiles = 485,300
"""

from pathlib import Path
import sys

import pyreadstat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIID_DTA     = PROJECT_ROOT / "wiidglobal_long.dta"
OUT          = PROJECT_ROOT / "app" / "data" / "historical_2000_2022.parquet"

START_YEAR = 2000
END_YEAR   = 2022


def main() -> None:
    if not WIID_DTA.exists():
        sys.exit(f"Missing input: {WIID_DTA}")

    print(f"Reading {WIID_DTA.name}...")
    df, _ = pyreadstat.read_dta(str(WIID_DTA))

    # Keep only real countries (ISO3 = 3 chars) and the year window.
    df = df[df["c3"].str.len() == 3].copy()
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]

    # Cast + slim.
    df["year"]       = df["year"].astype(int)
    df["percentile"] = df["percentile"].astype(int)
    out = df[["c3", "year", "percentile", "income_level"]].copy()
    out = out.sort_values(["c3", "year", "percentile"]).reset_index(drop=True)

    # Sanity check: balanced panel (211 countries * 23 years * 100 pcts).
    n_c = out["c3"].nunique()
    n_y = out["year"].nunique()
    exp = n_c * n_y * 100
    print(f"  countries: {n_c}, years: {n_y}, rows: {len(out)} "
          f"(balanced expected: {exp})")
    if len(out) != exp:
        print("  WARNING: panel is not balanced; benchmark growth may "
              "have NaN cells for some (country, percentile).")

    # Non-positive incomes would break log/ratio-based CAGR.  Report them.
    bad = (out["income_level"] <= 0).sum()
    if bad:
        print(f"  WARNING: {bad} rows with non-positive income_level "
              "(CAGR will be skipped for those).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
