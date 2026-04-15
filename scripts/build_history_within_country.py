"""
build_history_within_country.py
-------------------------------
Precomputes per-country-year inequality indices for historical years
(1950..2021) from wiidglobal_long.dta, so the new "Within-country
distributions" tab can show the historical context without recomputing
every time the app loads.

Output:
    app/data/history_country_indices.parquet
        columns: c3, year, population, mean_income, gini, ge_m1, ge0,
                 ge1, ge2, atk_050, atk_1, atk_2, bottom20, bottom40,
                 middle50, top10, top20, palma, s80s20
        rows:    211 countries x 72 years = ~15,000

The indices are computed on the 100 percentile income levels of each
(country, year) cell.  Weights within a country-year are uniform (one
percentile = 1 % of country population), so the measures reduce to
unweighted indices on 100 values — matching exactly what
engine.country_indices() does for the projected panel, so history and
projection can be concatenated seamlessly.

Run once (idempotent):
    python scripts/build_history_within_country.py
"""

from pathlib import Path
import sys
import pyreadstat
import pandas as pd

# Make the app's engine importable so we re-use the same formulas.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from engine import (_country_year_unweighted,     # noqa: E402
                    _COUNTRY_INDEX_COLS)

SRC_DTA   = PROJECT_ROOT / "wiidglobal_long.dta"
OUT_FILE  = PROJECT_ROOT / "app" / "data" / "history_country_indices.parquet"

YEAR_MIN = 1950
YEAR_MAX = 2021

KEEP_COLS = ["c3", "year", "percentile", "income_level", "population"]


def main() -> None:
    if not SRC_DTA.exists():
        sys.exit(f"ERROR: cannot find input file: {SRC_DTA}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {SRC_DTA.name} ...")
    df, _ = pyreadstat.read_dta(str(SRC_DTA), usecols=KEEP_COLS)
    print(f"  loaded {len(df):,} rows")

    # Keep only real countries (c3 of length 3) and historical years.
    mask = (
        (df["year"].between(YEAR_MIN, YEAR_MAX))
        & (df["c3"].astype(str).str.len() == 3)
    )
    d = df.loc[mask, KEEP_COLS].copy()
    d = d.dropna(subset=["income_level", "population"])
    d = d[d["population"] > 0]
    print(f"  kept rows: {len(d):,}  "
          f"({d['c3'].nunique()} countries, "
          f"years {int(d['year'].min())}..{int(d['year'].max())})")

    # Sort so the groupby loop is deterministic.
    d = d.sort_values(["c3", "year", "percentile"])

    rows = []
    for (c, yr), sub in d.groupby(["c3", "year"], sort=False):
        # Expect 100 rows; skip any malformed (c3, year).
        if len(sub) != 100:
            continue
        y = sub["income_level"].to_numpy(dtype=float)
        r = _country_year_unweighted(y)
        r["c3"]         = c
        r["year"]       = int(yr)
        # Country population is the same across the 100 percentile rows,
        # so any of them is fine.
        r["population"] = float(sub["population"].iloc[0])
        rows.append(r)

    out = pd.DataFrame(rows)
    cols = ["c3", "year", "population"] + _COUNTRY_INDEX_COLS
    out = out[cols].sort_values(["c3", "year"]).reset_index(drop=True)

    out.to_parquet(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE.relative_to(PROJECT_ROOT)}  "
          f"({OUT_FILE.stat().st_size/1024:.0f} KB,  {len(out):,} rows)")

    print("\nPreview (first and last 5 rows):")
    print(out.head().round(4).to_string(index=False))
    print("...")
    print(out.tail().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
