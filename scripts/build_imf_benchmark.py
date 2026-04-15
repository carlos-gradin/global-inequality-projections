"""
build_imf_benchmark.py
----------------------
Builds an "IMF benchmark" lookup of country-level annual real-GDP growth
rates for 2027..2031 (the IMF WEO April 2026 forecast horizon beyond the
backfill period).  The app's Compare-scenarios tab can optionally overlay
this benchmark against user scenarios.

Output:
    app/data/imf_benchmark_2027_2031.parquet
        columns: c3, year, growth_rate, source
        rows:    211 WIID countries x 5 years = 1,055

Waterfall priority (same logic as build_backfill.py, but for 2027..2031):
    1. IMF WEO April 2026 (NGDP_RPCH) for the country-year; tagged
       'WEO_actual' if the year is <= LATEST_ACTUAL_ANNUAL_DATA for that
       country (rare in 2027+, but possible if a country has a fiscal year
       extending past the publication date), 'WEO_forecast' otherwise.
    2. Population-weighted regional mean of WEO values for the same year,
       where region is region_wb; tagged 'WEO_regional_mean'.  Used for
       the small set of countries WEO does not cover in one or more years.
    3. Zero growth as a last-resort fallback (tagged 'ZERO_FALLBACK').

Usage from the app:
    The engine extends the per-country rate held constant for years beyond
    the last year in this file (default: hold 2031's rate for 2032+).
    See engine.project(..., post2026_scalar_by_year=...).

Run once from the project root:
    python scripts/build_imf_benchmark.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEO_XLSX     = PROJECT_ROOT / "projections_data" / "WEOApr2026all.xlsx"
REGIONS      = PROJECT_ROOT / "app" / "data" / "regions.parquet"
WPP          = PROJECT_ROOT / "app" / "data" / "wpp_population.parquet"
OUT          = PROJECT_ROOT / "app" / "data" / "imf_benchmark_2027_2031.parquet"

BENCH_YEARS = [2027, 2028, 2029, 2030, 2031]


def load_weo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rates_long, meta).  Same parsing logic as build_backfill.py,
    but melted on the benchmark year range."""
    weo = pd.read_excel(WEO_XLSX, sheet_name="Countries")
    weo = weo[weo["INDICATOR.ID"] == "NGDP_RPCH"].copy()

    def _parse_latest(v):
        if pd.isna(v):
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            s = str(v)
            if s.startswith("FY"):
                try:
                    return int(s[2:6])
                except Exception:
                    return None
            return None

    meta = weo[["COUNTRY.ID", "LATEST_ACTUAL_ANNUAL_DATA"]].copy()
    meta["latest_actual"] = meta["LATEST_ACTUAL_ANNUAL_DATA"].apply(_parse_latest)
    meta = meta.rename(columns={"COUNTRY.ID": "c3"})[["c3", "latest_actual"]]

    long = weo.melt(
        id_vars=["COUNTRY.ID"],
        value_vars=BENCH_YEARS,
        var_name="year",
        value_name="growth_pct",
    ).rename(columns={"COUNTRY.ID": "c3"})
    long["year"] = long["year"].astype(int)
    long = long.dropna(subset=["growth_pct"])
    long = long.merge(meta, on="c3", how="left")
    long["is_actual"] = long.apply(
        lambda r: (r["latest_actual"] is not None)
                  and (r["year"] <= r["latest_actual"]),
        axis=1,
    )
    return long[["c3", "year", "growth_pct", "is_actual"]], meta


def main() -> None:
    if not WEO_XLSX.exists():
        sys.exit(f"Missing input: {WEO_XLSX}")
    if not REGIONS.exists():
        sys.exit(f"Missing input: {REGIONS}")

    weo_long, _ = load_weo()
    regions = pd.read_parquet(REGIONS)[["c3", "region_wb"]]
    wpp = pd.read_parquet(WPP) if WPP.exists() else None

    print(f"WEO rows in 2027-31: {len(weo_long)}")
    print(f"WIID countries: {len(regions)}")

    # ------------------------------------------------------------------
    # Step 1: assemble per-country WEO rate (actual/forecast).
    # ------------------------------------------------------------------
    records: list[dict] = []
    for c3 in regions["c3"]:
        for yr in BENCH_YEARS:
            hit = weo_long[(weo_long["c3"] == c3) & (weo_long["year"] == yr)]
            if len(hit):
                r = hit.iloc[0]
                records.append({
                    "c3": c3, "year": yr,
                    "growth_rate": float(r["growth_pct"]) / 100.0,
                    "source": "WEO_actual" if r["is_actual"] else "WEO_forecast",
                })
            else:
                records.append({"c3": c3, "year": yr,
                                "growth_rate": np.nan,
                                "source": "MISSING"})

    out = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Step 2: fill MISSING with population-weighted WEO regional mean.
    # ------------------------------------------------------------------
    weo_with_reg = weo_long.merge(regions, on="c3", how="inner")
    if wpp is not None:
        weo_with_reg = weo_with_reg.merge(
            wpp[["c3", "year", "population"]], on=["c3", "year"], how="left"
        )

    def _region_mean(region: str, year: int) -> float:
        sub = weo_with_reg[(weo_with_reg["region_wb"] == region) &
                           (weo_with_reg["year"] == year)].copy()
        if sub.empty:
            return np.nan
        if "population" in sub.columns and sub["population"].notna().any():
            sub = sub.dropna(subset=["population"])
            return float(np.average(sub["growth_pct"], weights=sub["population"]) / 100.0)
        return float(sub["growth_pct"].mean() / 100.0)

    miss_mask = out["source"] == "MISSING"
    if miss_mask.any():
        out = out.merge(regions, on="c3", how="left")
        for i in out.index[miss_mask]:
            reg = out.at[i, "region_wb"]
            yr  = int(out.at[i, "year"])
            out.at[i, "growth_rate"] = _region_mean(reg, yr)
            out.at[i, "source"]      = "WEO_regional_mean"
        out = out.drop(columns=["region_wb"])

    # ------------------------------------------------------------------
    # Step 3: last-resort zero fallback + report.
    # ------------------------------------------------------------------
    still_missing = out["growth_rate"].isna()
    if still_missing.any():
        print(f"WARNING: {still_missing.sum()} cells still missing after "
              "regional mean; setting to 0 and tagging 'ZERO_FALLBACK'.")
        out.loc[still_missing, "growth_rate"] = 0.0
        out.loc[still_missing, "source"] = "ZERO_FALLBACK"

    print("\nSource tally:")
    print(out["source"].value_counts())
    print("\nBy year x source:")
    print(out.groupby(["year", "source"]).size().unstack(fill_value=0))

    imputed = out[out["source"] == "WEO_regional_mean"]["c3"].unique()
    if len(imputed):
        print(f"\nCountries imputed with regional mean (in >=1 year): "
              f"{sorted(imputed)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out[["c3", "year", "growth_rate", "source"]].to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(out)} rows, "
          f"{out['c3'].nunique()} countries x {len(BENCH_YEARS)} years)")


if __name__ == "__main__":
    main()
