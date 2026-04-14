"""
build_history.py
----------------
Precomputes global inequality indices for historical years (1950..2021)
from wiidglobal_long.dta, so the app can display them next to the
projected series without recomputing every time.

Outputs:
    app/data/history_indices.parquet
        Global indices per year (columns: year, mean_income, gini,
        ge_m1, ge0, ge1, ge2, atk_025..atk_2, bottom20, bottom40,
        middle50, top10, top20, palma, s80s20).

    app/data/history_between_within.parquet
        Per-year country-level between/within decomposition for gini,
        ge0, ge1, ge2.  For each measure m we store five columns:
            m         = I(y)            total
            m_b       = I(y^b)          smoothed (no within-country ineq.)
            m_w       = I(y^w)          standardized (no between-country ineq.)
            m_sh_b    = Shapley between = (I(y^b)+I(y)-I(y^w))/2
            m_sh_w    = Shapley within  = (I(y^w)+I(y)-I(y^b))/2

Method: for each historical year we pool every (country, percentile) cell,
weight by (country_population * 0.01), and apply the same index formulas
used in app/engine.py (imported directly to guarantee consistency).

Weights come from WIID's own `population` column (country-year). We do NOT
use UN WPP here because WIID already stores a matching population with
the distribution it reports — keeping weights and distribution aligned.

Run once (idempotent) from the project root:
    python scripts/build_history.py
"""

from pathlib import Path
import sys
import pyreadstat
import pandas as pd

# Make the app's engine importable so we re-use its index formulas.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from engine import _year_indices, between_within_country  # noqa: E402

SRC_DTA   = PROJECT_ROOT / "wiidglobal_long.dta"
OUT_DIR   = PROJECT_ROOT / "app" / "data"
OUT_FILE  = OUT_DIR / "history_indices.parquet"
OUT_BW    = OUT_DIR / "history_between_within.parquet"

# Historical range — stops at 2021 because 2022 is the projection base year
# and the projected panel already covers 2022 onwards.
YEAR_MIN = 1950
YEAR_MAX = 2021

KEEP_COLS = ["c3", "year", "percentile", "income_level", "population"]


def main() -> None:
    if not SRC_DTA.exists():
        sys.exit(f"ERROR: cannot find input file: {SRC_DTA}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load the minimum columns we need (saves RAM).
    # ------------------------------------------------------------------
    print(f"Reading {SRC_DTA.name} ...")
    df, _ = pyreadstat.read_dta(str(SRC_DTA), usecols=KEEP_COLS)
    print(f"  loaded {len(df):,} rows")

    # ------------------------------------------------------------------
    # 2) Keep only real countries (c3 has 3 letters) and historical years.
    # ------------------------------------------------------------------
    mask = (
        (df["year"].between(YEAR_MIN, YEAR_MAX))
        & (df["c3"].astype(str).str.len() == 3)
    )
    d = df.loc[mask, KEEP_COLS].copy()
    # Weight = country population * 0.01  (one percentile = 1 % of pop)
    d["weight"] = d["population"] * 0.01
    d = d.dropna(subset=["income_level", "weight"])
    d = d[d["weight"] > 0]
    print(f"  kept rows: {len(d):,}  "
          f"({d['c3'].nunique()} countries, "
          f"years {int(d['year'].min())}..{int(d['year'].max())})")

    # ------------------------------------------------------------------
    # 3) Compute indices year by year using the engine's function.
    #    _year_indices expects columns: income_level, weight.
    # ------------------------------------------------------------------
    rows = []
    for yr, df_y in d.groupby("year", sort=True):
        r = _year_indices(df_y)
        r["year"] = int(yr)
        rows.append(r)

    out = pd.DataFrame(rows)
    # Keep the same column order as engine.indices() for a clean merge.
    cols = ["year", "mean_income",
            "gini", "ge_m1", "ge0", "ge1", "ge2",
            "atk_025", "atk_050", "atk_075", "atk_1", "atk_2",
            "bottom20", "bottom40", "middle50", "top10", "top20",
            "palma", "s80s20"]
    out = out[cols].sort_values("year").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4) Save the global-index parquet.
    # ------------------------------------------------------------------
    out.to_parquet(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE.relative_to(PROJECT_ROOT)}  "
          f"({OUT_FILE.stat().st_size/1024:.0f} KB,  {len(out)} rows)")
    print("\nPreview (indices):")
    print(out[["year", "gini", "ge0", "ge1", "ge2",
               "bottom40", "top10", "palma"]].round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # 5) Country-level between/within decomposition for every year.
    #    `between_within_country` expects a panel with columns
    #    c3, year, percentile, income_level, weight — which `d` already
    #    has (we just need to keep those columns and rename percentile
    #    to match the engine's expectations).
    # ------------------------------------------------------------------
    panel_like = d[["c3", "year", "percentile", "income_level",
                    "weight"]].copy()
    bw = between_within_country(panel_like)
    bw.to_parquet(OUT_BW, index=False)
    print(f"\nWrote {OUT_BW.relative_to(PROJECT_ROOT)}  "
          f"({OUT_BW.stat().st_size/1024:.0f} KB,  {len(bw)} rows)")
    print("\nPreview (between/within, MLD):")
    print(bw[["year", "ge0", "ge0_b", "ge0_w",
              "ge0_sh_b", "ge0_sh_w"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
