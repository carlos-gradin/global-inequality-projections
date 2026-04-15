"""
engine.py
---------
Inequality-projection engine for the app.

Three public functions:

    project(baseline, regions, wpp, spec) -> panel
        Grow the 2022 country x percentile income levels to every year
        between 2022 and spec.target_year using the chosen growth spec,
        and attach a population weight from UN WPP.

    indices(panel) -> per-year DataFrame of global inequality indices.

    decompose_ge(panel, group) -> per-year Theil(GE1) and MLD(GE0) split
        into a within- and between-group component.

"Global" means: pool every country-percentile cell and weight by
(population_of_country_in_that_year * 0.01) since each percentile
represents 1 % of its country's population.

All formulas are standard; see e.g. Cowell, "Measuring Inequality", 3rd ed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


# ======================================================================
# Growth specification
# ======================================================================

@dataclass
class GrowthSpec:
    """
    All the user inputs needed to build an annualised growth-rate array.

    Four modes are supported:

    A.  mode = 'uniform'
        Every (country, percentile) grows at `uniform_rate`.
        Within-country shape preserved.

    B.  mode = 'by_aggregate'
        Every country gets the rate of its aggregate (region_wb or
        incomegroup).  Rates are supplied in `rates_by_group`.
        Within-country shape preserved.

    C.  mode = 'by_popgroup'
        Three rates, applied to every country, one per population
        block:  `block_rates = {'b40', 'm50', 't10'}`.
        Within-country shape CHANGES (blocks grow at different speeds).

    D.  mode = 'by_aggregate_popgroup'
        Three rates per aggregate:
            block_rates_by_group[group] = {'b40': ..., 'm50': ..., 't10': ...}
        Within-country shape changes, differently by aggregate.

    Giants override (CHN, IND): when `split_china` / `split_india` is
    True, those countries ignore the pattern above.
        - In modes A and B CHN/IND get a single scalar rate
          (`china_rate`, `india_rate`) across all percentiles.
        - In modes C and D they get their own 3 block rates
          (`china_block_rates`, `india_block_rates`).  `china_rate` /
          `india_rate` are used as the fallback for any unset block.

    Population blocks follow the convention in the spec:
        percentiles  1..40   -> 'b40'  (bottom 40)
        percentiles 41..90   -> 'm50'  (middle 50)
        percentiles 91..100  -> 't10'  (top 10)
    """

    mode: str                               # 'uniform' | 'by_aggregate' |
                                            # 'by_popgroup' | 'by_aggregate_popgroup'
    target_year: int = 2050

    uniform_rate: float = 0.03              # used in 'uniform' and as the global fallback

    aggregate: str = "region_wb"            # 'region_wb' or 'incomegroup' (B and D)
    rates_by_group: dict[str, float] = field(default_factory=dict)   # mode B

    # Mode C: one rate per block, same across all countries.
    block_rates: dict[str, float] = field(default_factory=dict)

    # Mode D: nested {group -> {block -> rate}}.
    block_rates_by_group: dict[str, dict[str, float]] = field(default_factory=dict)

    # Giants.  Scalar rates used in modes A/B; block-rate dicts used in modes C/D.
    split_china: bool = False
    split_india: bool = False
    china_rate: float = 0.03
    india_rate: float = 0.03
    china_block_rates: dict[str, float] = field(default_factory=dict)
    india_block_rates: dict[str, float] = field(default_factory=dict)


# ======================================================================
# Step 1: build a growth rate per (country, percentile)
# ======================================================================

# Fixed mapping percentile -> block label (1..40 = b40, 41..90 = m50,
# 91..100 = t10).  Built once at import time.
BLOCK_OF_PERCENTILE = np.array(
    ["b40"] * 40 + ["m50"] * 50 + ["t10"] * 10, dtype=object
)


def _percentile_growth_rates(regions: pd.DataFrame,
                             spec: GrowthSpec) -> pd.DataFrame:
    """
    Return a wide DataFrame (index = c3, columns = 1..100) giving the
    annual growth rate for every country-percentile cell under `spec`.

    In modes A and B the rate is constant across percentiles within a
    country.  In modes C and D it varies by population block.
    """
    c3_arr = regions["c3"].to_numpy()
    n      = len(c3_arr)
    # default everywhere: the global fallback rate
    rate = np.full((n, 100), spec.uniform_rate, dtype=float)

    # blocks: length-100 array telling each percentile's block key
    blocks = BLOCK_OF_PERCENTILE

    if spec.mode == "uniform":
        # Nothing to change; every cell is uniform_rate.
        pass

    elif spec.mode == "by_aggregate":
        if spec.aggregate not in ("region_wb", "incomegroup"):
            raise ValueError(
                f"Unknown aggregate: {spec.aggregate!r}; "
                "expected 'region_wb' or 'incomegroup'."
            )
        groups = regions[spec.aggregate].astype(str).to_numpy()
        # One rate per country (same across its 100 percentiles)
        country_rate = np.array(
            [spec.rates_by_group.get(g, spec.uniform_rate) for g in groups]
        )
        rate = np.broadcast_to(country_rate[:, None], (n, 100)).copy()

    elif spec.mode == "by_popgroup":
        # Same three block rates applied to every country.
        block_row = np.array(
            [spec.block_rates.get(b, spec.uniform_rate) for b in blocks]
        )
        rate = np.broadcast_to(block_row[None, :], (n, 100)).copy()

    elif spec.mode == "by_aggregate_popgroup":
        if spec.aggregate not in ("region_wb", "incomegroup"):
            raise ValueError(
                f"Unknown aggregate: {spec.aggregate!r}; "
                "expected 'region_wb' or 'incomegroup'."
            )
        groups = regions[spec.aggregate].astype(str).to_numpy()
        for i, g in enumerate(groups):
            sub = spec.block_rates_by_group.get(g, {})
            rate[i, :] = [sub.get(b, spec.uniform_rate) for b in blocks]

    else:
        raise ValueError(f"Unknown mode: {spec.mode!r}")

    # Giants override -----------------------------------------------------
    modes_scalar_giant = ("uniform", "by_aggregate")
    modes_block_giant  = ("by_popgroup", "by_aggregate_popgroup")

    def _apply_giant(c3_code: str, scalar_rate: float,
                     block_rates: dict[str, float]) -> None:
        idxs = np.where(c3_arr == c3_code)[0]
        if not len(idxs):
            return
        i = idxs[0]
        if spec.mode in modes_scalar_giant:
            rate[i, :] = scalar_rate
        elif spec.mode in modes_block_giant:
            rate[i, :] = [block_rates.get(b, scalar_rate) for b in blocks]

    if spec.split_china:
        _apply_giant("CHN", spec.china_rate, spec.china_block_rates)
    if spec.split_india:
        _apply_giant("IND", spec.india_rate, spec.india_block_rates)

    out = pd.DataFrame(rate, index=c3_arr, columns=range(1, 101))
    out.index.name = "c3"
    out.columns.name = "percentile"
    return out


# ======================================================================
# Step 2: project the income panel
# ======================================================================

BASE_YEAR             = 2022      # WIID cross-section year
BACKFILL_END_YEAR     = 2026      # last year covered by WDI / WEO backfill
PROJECTION_START_YEAR = 2027      # first year of USER-driven growth


# ======================================================================
# Benchmark ("historical continuation") growth
# ======================================================================

def benchmark_growth(historical: pd.DataFrame,
                     n_years: int = 10,
                     base_year: int = BASE_YEAR) -> pd.DataFrame:
    """
    Compute a per-(country, percentile) compound annual growth rate from
    the last `n_years` of the WIID historical panel.

    For each (c3, percentile) pair the rate is

        CAGR = (income[base_year] / income[base_year - n_years]) ** (1/n_years) - 1

    The WIID Companion panel is balanced (survey-year gaps interpolated
    and post-last-survey years extrapolated upstream), so no fallback is
    needed — every cell has a value.  If income at either endpoint is
    non-positive the CAGR is set to 0.0 (degenerate case, shouldn't occur
    with the current dataset).

    Parameters
    ----------
    historical : DataFrame with columns c3, year, percentile, income_level
                 (as produced by scripts/build_historical.py).
    n_years    : Window length, in years (e.g. 10 = CAGR over 2012..2022
                 when base_year=2022).
    base_year  : End of the window (default 2022 = WIID latest).

    Returns
    -------
    DataFrame indexed by c3 with 100 columns (percentile 1..100) giving
    the annualised real growth rate (decimal, e.g. 0.025 for 2.5%).
    Suitable for passing as `post2026_rate` to `project()`.
    """
    if n_years < 1:
        raise ValueError("n_years must be >= 1")
    t0 = base_year - n_years
    t1 = base_year

    df = historical[historical["year"].isin([t0, t1])].copy()
    # Pivot to (c3, percentile) x year.
    wide = df.pivot_table(index=["c3", "percentile"],
                          columns="year",
                          values="income_level")
    if t0 not in wide.columns or t1 not in wide.columns:
        raise ValueError(
            f"historical panel lacks data for year {t0} or {t1}; "
            f"available years: {sorted(wide.columns.tolist())}"
        )

    y0 = wide[t0].to_numpy()
    y1 = wide[t1].to_numpy()
    # Safe CAGR: where either endpoint is <= 0, set to 0.
    good = (y0 > 0) & (y1 > 0)
    rate = np.zeros_like(y0, dtype=float)
    rate[good] = (y1[good] / y0[good]) ** (1.0 / n_years) - 1.0

    out = pd.DataFrame({"growth_rate": rate}, index=wide.index)
    out = out.reset_index().pivot(index="c3",
                                  columns="percentile",
                                  values="growth_rate")
    # Ensure columns are exactly 1..100 in order.
    out = out.reindex(columns=range(1, 101))
    out.index.name = "c3"
    out.columns.name = "percentile"
    return out


def project(
    baseline: pd.DataFrame,
    regions:  pd.DataFrame,
    wpp:      pd.DataFrame,
    spec:     GrowthSpec,
    backfill: pd.DataFrame,
    *,
    post2026_rate: Optional[pd.DataFrame] = None,
    post2026_scalar_by_year: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Produce a long panel with one row per (c3, year, percentile) covering
    years BASE_YEAR..spec.target_year.  Columns:

        c3, year, percentile, income_level, weight

    Two phases:
        2023..2026  "backfill": per-country annual growth rate taken from
                    `backfill` (WDI realised + IMF WEO April 2026, see
                    scripts/build_backfill.py). Applied UNIFORMLY across
                    the 100 percentiles of each country, so within-country
                    distribution shape is held constant during backfill
                    years (consistent with Gradín 2024 and Kanbur,
                    Ortiz-Juarez & Sumner 2024).
        2027..target "projection": user-chosen growth pattern from `spec`.

    `post2026_rate` (optional): if given, a DataFrame indexed by c3 with
    columns = percentile 1..100 and values = annualised growth rate
    (decimal).  It overrides `spec` for years >= PROJECTION_START_YEAR
    and is used to implement the historical-continuation benchmark
    (see `benchmark_growth`).  The backfill phase (2023..2026) is never
    overridden; only the user-driven phase is.

    `post2026_scalar_by_year` (optional): if given, a DataFrame indexed by
    c3 with columns = integer years and values = annualised growth rate
    (decimal).  Unlike `post2026_rate`, this table varies year-by-year
    but is applied uniformly across the 100 percentiles of each country
    (no within-country shape change).  For any projection year t ≥ 2027
    beyond the last year in the table, the rate of that last year is held
    constant ("flat extrapolation").  Used to implement the IMF WEO
    benchmark (scripts/build_imf_benchmark.py).  Mutually exclusive with
    `post2026_rate`.

    `weight` is the population of the percentile in that year (people),
    i.e. country_population * 0.01 .
    """
    if spec.target_year <= BASE_YEAR:
        raise ValueError(f"target_year must be > {BASE_YEAR}")
    if post2026_rate is not None and post2026_scalar_by_year is not None:
        raise ValueError(
            "post2026_rate and post2026_scalar_by_year are mutually exclusive"
        )

    # Reshape baseline to wide: rows = c3, cols = percentile 1..100.
    base_wide = (
        baseline.pivot(index="c3", columns="percentile", values="income_level")
        .sort_index()
    )

    # Rate array used from PROJECTION_START_YEAR onwards.  Either
    # user-driven (default) or the override passed in.
    if post2026_rate is None:
        post_rate = _percentile_growth_rates(regions, spec)         # c3 × 1..100
    else:
        post_rate = post2026_rate

    # Align everything on the same countries & percentile columns.
    common    = base_wide.index.intersection(post_rate.index)
    base_wide = base_wide.loc[common]
    post_rate = post_rate.loc[common, base_wide.columns]

    # Align the IMF-style scalar table if provided.
    if post2026_scalar_by_year is not None:
        sy_wide = (
            post2026_scalar_by_year
            .reindex(base_wide.index)
            .fillna(0.0)
        )
        # Ensure columns are ints and sorted.
        sy_wide.columns = [int(c) for c in sy_wide.columns]
        sy_wide = sy_wide.reindex(sorted(sy_wide.columns), axis=1)
        sy_last_year = int(max(sy_wide.columns))
    else:
        sy_wide = None
        sy_last_year = None

    # Backfill lookup: c3 × year matrix of per-country scalar rates.
    # Any country missing from backfill will get 0% growth for those years
    # (defensive — build_backfill.py should leave no gaps).
    bf_wide = (
        backfill.pivot(index="c3", columns="year", values="growth_rate")
        .reindex(base_wide.index)
        .fillna(0.0)
    )

    c3_idx     = base_wide.index
    perc_cols  = base_wide.columns
    n          = len(c3_idx)
    post_arr   = post_rate.to_numpy()                               # n × 100

    # --- iterative income-level build -------------------------------------
    # We loop year-by-year because growth rates now vary across years in
    # the backfill phase (no closed-form exponent).
    inc_curr = base_wide.to_numpy().astype(float)                   # 2022 levels
    frames   = []

    def _frame(arr: np.ndarray, year: int) -> pd.DataFrame:
        df = pd.DataFrame(arr, index=c3_idx, columns=perc_cols)
        f  = df.stack().rename("income_level").reset_index()
        f["year"] = int(year)
        return f

    # Emit 2022 (baseline) first.
    frames.append(_frame(inc_curr, BASE_YEAR))

    for t in range(BASE_YEAR + 1, spec.target_year + 1):
        if t <= BACKFILL_END_YEAR:
            # Backfill: country-scalar rate broadcast across all 100 percentiles.
            country_rate = (bf_wide[t].to_numpy() if t in bf_wide.columns
                            else np.zeros(n))
            rate_arr = np.broadcast_to(country_rate[:, None], (n, 100))
        elif sy_wide is not None:
            # IMF-style per-(country, year) scalar; hold the last year's
            # rate constant for any year beyond the table's horizon.
            lookup_year = min(t, sy_last_year)
            country_rate = sy_wide[lookup_year].to_numpy()
            rate_arr = np.broadcast_to(country_rate[:, None], (n, 100))
        else:
            # Post-2026: user-driven spec OR per-percentile benchmark override.
            rate_arr = post_arr
        inc_curr = inc_curr * (1.0 + rate_arr)
        frames.append(_frame(inc_curr, t))

    panel = pd.concat(frames, ignore_index=True)

    # --- attach population weights ----------------------------------------
    # UN WPP gives total population per country per year; a percentile
    # is 1 % of that.
    pop = wpp[["c3", "year", "population"]]
    panel = panel.merge(pop, on=["c3", "year"], how="left")
    panel["weight"] = panel["population"] * 0.01
    panel = panel.drop(columns=["population"])

    # Drop any rows missing population (shouldn't happen with current data
    # since coverage is complete, but be defensive).
    panel = panel.dropna(subset=["weight"])

    # Small, ordered column set.
    return panel[["c3", "year", "percentile", "income_level", "weight"]]


# ======================================================================
# Step 3: inequality indices on a weighted sample
# ======================================================================

def _weighted_gini(y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Gini via the trapezoidal Lorenz-curve formula."""
    order = np.argsort(y)
    y = y[order]
    w = w[order]
    W = w.sum()
    # cumulative population share (at the TOP of each bin)
    P = np.cumsum(w) / W
    # cumulative income share (at the TOP of each bin)
    cum_income = np.cumsum(w * y)
    T = cum_income[-1]
    if T <= 0:
        return np.nan
    L = cum_income / T
    # Trapezoidal rule: Gini = 1 - sum (p_i - p_{i-1})*(L_i + L_{i-1})
    P_prev = np.concatenate(([0.0], P[:-1]))
    L_prev = np.concatenate(([0.0], L[:-1]))
    return 1.0 - np.sum((P - P_prev) * (L + L_prev))


def _weighted_ge(y: np.ndarray, w: np.ndarray, alpha: float) -> float:
    """Generalized Entropy GE(alpha) on a weighted sample."""
    W = w.sum()
    mu = (w * y).sum() / W
    if mu <= 0:
        return np.nan
    # Normalised weights summing to 1.
    f = w / W
    if alpha == 0:                # MLD
        return (f * np.log(mu / y)).sum()
    if alpha == 1:                # Theil-T
        return (f * (y / mu) * np.log(y / mu)).sum()
    # General case
    return (1.0 / (alpha * (alpha - 1.0))) * ((f * (y / mu) ** alpha).sum() - 1.0)


def _weighted_atkinson(y: np.ndarray, w: np.ndarray, eps: float) -> float:
    W = w.sum()
    mu = (w * y).sum() / W
    if mu <= 0:
        return np.nan
    f = w / W
    if eps == 1.0:
        # Geometric mean form
        return 1.0 - np.exp((f * np.log(y)).sum()) / mu
    e = 1.0 - eps
    return 1.0 - (((f * (y / mu) ** e).sum()) ** (1.0 / e))


def _share_by_pop_quantile(y: np.ndarray, w: np.ndarray,
                           p_lo: float, p_hi: float) -> float:
    """Income share of the population slice between cumulative population
       fractions p_lo and p_hi (0..1), measured on the sorted distribution."""
    order = np.argsort(y)
    y = y[order]
    w = w[order]
    W = w.sum()
    T = (w * y).sum()
    if T <= 0:
        return np.nan
    # cumulative weight share at top of each cell
    P = np.cumsum(w) / W
    # cumulative income share at top of each cell
    L = np.cumsum(w * y) / T
    # Interpolate L at p_lo and p_hi.
    L_lo = np.interp(p_lo, P, L, left=0.0, right=1.0)
    L_hi = np.interp(p_hi, P, L, left=0.0, right=1.0)
    return float(L_hi - L_lo)


def _year_indices(df_year: pd.DataFrame) -> dict:
    """Return a dict of inequality indices for one year."""
    y = df_year["income_level"].to_numpy(dtype=float)
    w = df_year["weight"].to_numpy(dtype=float)

    # Basic: mean
    W = w.sum()
    mu = (w * y).sum() / W

    out = {
        "mean_income": mu,
        "gini":   _weighted_gini(y, w),
        "ge_m1":  _weighted_ge(y, w, -1.0),
        "ge0":    _weighted_ge(y, w,  0.0),   # MLD
        "ge1":    _weighted_ge(y, w,  1.0),   # Theil
        "ge2":    _weighted_ge(y, w,  2.0),
        "atk_025": _weighted_atkinson(y, w, 0.25),
        "atk_050": _weighted_atkinson(y, w, 0.50),
        "atk_075": _weighted_atkinson(y, w, 0.75),
        "atk_1":   _weighted_atkinson(y, w, 1.00),
        "atk_2":   _weighted_atkinson(y, w, 2.00),
        "bottom40": _share_by_pop_quantile(y, w, 0.0, 0.40),
        "middle50": _share_by_pop_quantile(y, w, 0.40, 0.90),
        "top10":    _share_by_pop_quantile(y, w, 0.90, 1.00),
        "bottom20": _share_by_pop_quantile(y, w, 0.0, 0.20),
        "top20":    _share_by_pop_quantile(y, w, 0.80, 1.00),
    }
    # Derived ratios
    out["palma"]   = (out["top10"]    / out["bottom40"]) if out["bottom40"] > 0 else np.nan
    out["s80s20"]  = (out["top20"]    / out["bottom20"]) if out["bottom20"] > 0 else np.nan
    return out


def indices(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute inequality indices for every year in the panel."""
    rows = []
    for yr, df_y in panel.groupby("year", sort=True):
        r = _year_indices(df_y)
        r["year"] = int(yr)
        rows.append(r)
    out = pd.DataFrame(rows).set_index("year").sort_index().reset_index()
    # Keep a sensible column order
    cols = ["year", "mean_income",
            "gini", "ge_m1", "ge0", "ge1", "ge2",
            "atk_025", "atk_050", "atk_075", "atk_1", "atk_2",
            "bottom20", "bottom40", "middle50", "top10", "top20",
            "palma", "s80s20"]
    return out[cols]


# ----------------------------------------------------------------------
# Per-country (within-country) inequality indices
# ----------------------------------------------------------------------
#
# For one (country, year) the panel stores 100 rows, one per percentile,
# each with equal implicit weight (one percentile = 1 % of country pop).
# Inequality measures used here are scale-invariant in the weights, so
# uniform weights give exactly the same number as the country population
# weights — we use w = 1 for simplicity.
#
# The output row also records the country's total population in that year
# (sum of the 100 percentile weights), so the app can later compute
# population-weighted cross-country means.

_COUNTRY_INDEX_COLS = [
    "mean_income",
    "gini", "ge_m1", "ge0", "ge1", "ge2",
    "atk_050", "atk_1", "atk_2",
    "bottom20", "bottom40", "middle50", "top10", "top20",
    "palma", "s80s20",
]


def _country_year_unweighted(y: np.ndarray) -> dict:
    """Inequality indices for one country-year, on 100 percentile values
    with equal weights (weights cancel in every measure used here)."""
    w = np.ones_like(y, dtype=float)
    out = {
        "mean_income": float(np.mean(y)),
        "gini":   _weighted_gini(y, w),
        "ge_m1":  _weighted_ge(y, w, -1.0),
        "ge0":    _weighted_ge(y, w,  0.0),
        "ge1":    _weighted_ge(y, w,  1.0),
        "ge2":    _weighted_ge(y, w,  2.0),
        "atk_050": _weighted_atkinson(y, w, 0.50),
        "atk_1":   _weighted_atkinson(y, w, 1.00),
        "atk_2":   _weighted_atkinson(y, w, 2.00),
        "bottom20": _share_by_pop_quantile(y, w, 0.00, 0.20),
        "bottom40": _share_by_pop_quantile(y, w, 0.00, 0.40),
        "middle50": _share_by_pop_quantile(y, w, 0.40, 0.90),
        "top10":    _share_by_pop_quantile(y, w, 0.90, 1.00),
        "top20":    _share_by_pop_quantile(y, w, 0.80, 1.00),
    }
    out["palma"]  = (out["top10"]    / out["bottom40"]) if out["bottom40"] > 0 else np.nan
    out["s80s20"] = (out["top20"]    / out["bottom20"]) if out["bottom20"] > 0 else np.nan
    return out


def country_indices(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-country-year inequality indices.  One row per (c3, year).

    Input: a long panel with columns c3, year, percentile, income_level,
    weight (weight = country_pop * 0.01 per percentile).

    Returns columns: c3, year, population, and every measure listed in
    `_COUNTRY_INDEX_COLS`.
    """
    # Sort once so the groupby loops are cheap and deterministic.
    pn = panel.sort_values(["c3", "year", "percentile"])
    rows = []
    for (c, yr), df in pn.groupby(["c3", "year"], sort=False):
        y = df["income_level"].to_numpy(dtype=float)
        r = _country_year_unweighted(y)
        r["c3"]         = c
        r["year"]       = int(yr)
        # Country total population = sum of percentile weights.
        r["population"] = float(df["weight"].sum())
        rows.append(r)
    out = pd.DataFrame(rows)
    cols = ["c3", "year", "population"] + _COUNTRY_INDEX_COLS
    return out[cols]


# ======================================================================
# Step 4: within/between decomposition for GE(0) and GE(1)
# ======================================================================

def _decompose_one_year(df_year: pd.DataFrame, group_key: pd.Series) -> dict:
    """
    MLD(GE0) and Theil(GE1) decomposition.
    group_key is a Series aligned with df_year giving each row's group label.
    """
    y = df_year["income_level"].to_numpy(dtype=float)
    w = df_year["weight"].to_numpy(dtype=float)
    g = group_key.to_numpy()

    W = w.sum()
    mu = (w * y).sum() / W
    if mu <= 0:
        return {"ge0_within": np.nan, "ge0_between": np.nan,
                "ge1_within": np.nan, "ge1_between": np.nan}

    # Per-group sums
    df = pd.DataFrame({"y": y, "w": w, "g": g})
    grp = df.groupby("g", observed=True)
    W_g  = grp["w"].sum()
    T_g  = grp.apply(lambda x: (x["w"] * x["y"]).sum(), include_groups=False)
    mu_g = T_g / W_g
    share_g = W_g / W

    # Between-group terms
    # MLD between: sum_g (W_g/W) * ln(mu / mu_g)
    ge0_b = (share_g * np.log(mu / mu_g)).sum()
    # Theil between: sum_g (W_g/W) * (mu_g/mu) * ln(mu_g/mu)
    ge1_b = (share_g * (mu_g / mu) * np.log(mu_g / mu)).sum()

    # Within-group terms
    ge0_w = 0.0
    ge1_w = 0.0
    for gname, sub in grp:
        yg = sub["y"].to_numpy()
        wg = sub["w"].to_numpy()
        I0 = _weighted_ge(yg, wg, 0.0)
        I1 = _weighted_ge(yg, wg, 1.0)
        s_g = W_g.loc[gname] / W
        mug = mu_g.loc[gname]
        # MLD within weight = s_g
        ge0_w += s_g * I0
        # Theil within weight = s_g * (mu_g/mu)
        ge1_w += s_g * (mug / mu) * I1

    return {
        "ge0_within":  ge0_w,
        "ge0_between": ge0_b,
        "ge1_within":  ge1_w,
        "ge1_between": ge1_b,
    }


def indices_by_group(panel: pd.DataFrame, regions: pd.DataFrame,
                     group: str = "region_wb",
                     years: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Inequality indices computed SEPARATELY for each group in column
    `group` of `regions`.  Useful for the summary table comparing
    inequality inside each region / income group.

    If `years` is given, only those years are computed (saves time when
    the summary table needs just start and end year).

    Returns a long DataFrame with one row per (group, year).
    """
    if group not in regions.columns:
        raise ValueError(f"group must be a column of regions; got {group!r}")

    key_map = regions.set_index("c3")[group]
    panel2 = panel.copy()
    panel2["_g"] = panel2["c3"].map(key_map)
    panel2 = panel2.dropna(subset=["_g"])

    if years is not None:
        panel2 = panel2[panel2["year"].isin(years)]

    rows = []
    for (gname, yr), df_gy in panel2.groupby(["_g", "year"], sort=True):
        r = _year_indices(df_gy)
        r["group"] = gname
        r["year"] = int(yr)
        rows.append(r)
    out = pd.DataFrame(rows)
    # Put group and year first for readability.
    cols = ["group", "year"] + [c for c in out.columns if c not in ("group", "year")]
    return out[cols]


# ======================================================================
# Step 4b: country-level between/within decomposition (Shapley-style)
# ======================================================================
#
# For each year we build two counterfactual distributions on the SAME
# sample as `panel`:
#
#   y^b  "smoothed":      every individual in country c gets y^b = mu_c
#                         (country mean income).  No inequality WITHIN
#                         countries is left; only BETWEEN-country means.
#
#   y^w  "standardized":  every individual's income is rescaled so that
#                         each country has the same mean (= global mean).
#                         No inequality BETWEEN countries is left; only
#                         WITHIN-country shapes.
#
# Three inequality numbers per measure: I(y), I(y^b), I(y^w).
# - Naive between = I(y^b), naive within = I(y^w).
#     I(y^b) + I(y^w) = I(y)  exactly ONLY for GE(0) (MLD).
# - Shapley between = [I(y^b) + I(y) - I(y^w)] / 2
#   Shapley within  = [I(y^w) + I(y) - I(y^b)] / 2
#     These ALWAYS sum to I(y), for any measure (Gini, any GE(alpha), ...).
#
# Weights stay as the observed cell weights throughout.
# Scale-invariance of Gini and GE means the choice of "standardize to
# global mean" is equivalent to "standardize to any constant" for
# inequality purposes; we pick global mean so the mean is preserved.

MEASURES = ("gini", "ge0", "ge1", "ge2")


def _measure(y: np.ndarray, w: np.ndarray, name: str) -> float:
    if name == "gini":
        return _weighted_gini(y, w)
    if name == "ge0":
        return _weighted_ge(y, w, 0.0)
    if name == "ge1":
        return _weighted_ge(y, w, 1.0)
    if name == "ge2":
        return _weighted_ge(y, w, 2.0)
    raise ValueError(f"Unknown measure {name!r}")


def _between_within_one_year(df_year: pd.DataFrame) -> dict:
    """Country-level between/within decomposition for one year of the panel.

    Returns a dict with, for each measure m in {gini, ge0, ge1, ge2}:
        m         -> I(y)           total
        m_b       -> I(y^b)         inequality of the country-smoothed distr.
        m_w       -> I(y^w)         inequality of the country-standardized distr.
        m_sh_b    -> Shapley between = (I(y^b) + I(y) - I(y^w)) / 2
        m_sh_w    -> Shapley within  = (I(y^w) + I(y) - I(y^b)) / 2
    """
    y = df_year["income_level"].to_numpy(dtype=float)
    w = df_year["weight"].to_numpy(dtype=float)
    c = df_year["c3"].to_numpy()

    W = w.sum()
    mu = (w * y).sum() / W

    # Weighted country means.
    tmp = pd.DataFrame({"y": y, "w": w, "c": c})
    grp = tmp.groupby("c", observed=True)
    mu_c = (grp.apply(lambda x: (x["w"] * x["y"]).sum() / x["w"].sum(),
                      include_groups=False)).to_dict()

    # y^b: replace each cell's income with its country mean.
    y_b = np.fromiter((mu_c[cc] for cc in c), dtype=float, count=len(c))
    # y^w: scale each country so that its mean = global mean.
    scale = np.fromiter((mu / mu_c[cc] for cc in c), dtype=float, count=len(c))
    y_w = y * scale

    out: dict = {}
    for m in MEASURES:
        I  = _measure(y,   w, m)
        Ib = _measure(y_b, w, m)
        Iw = _measure(y_w, w, m)
        out[m]         = I
        out[f"{m}_b"]  = Ib
        out[f"{m}_w"]  = Iw
        out[f"{m}_sh_b"] = (Ib + I - Iw) / 2.0
        out[f"{m}_sh_w"] = (Iw + I - Ib) / 2.0
    return out


def between_within_country(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-year country-level between/within decomposition for every
    measure in `MEASURES`.  Columns:

        year,
        gini, gini_b, gini_w, gini_sh_b, gini_sh_w,
        ge0,  ge0_b,  ge0_w,  ge0_sh_b,  ge0_sh_w,
        ge1,  ge1_b,  ge1_w,  ge1_sh_b,  ge1_sh_w,
        ge2,  ge2_b,  ge2_w,  ge2_sh_b,  ge2_sh_w
    """
    rows = []
    for yr, df_y in panel.groupby("year", sort=True):
        r = _between_within_one_year(df_y)
        r["year"] = int(yr)
        rows.append(r)
    out = pd.DataFrame(rows)
    cols = ["year"]
    for m in MEASURES:
        cols += [m, f"{m}_b", f"{m}_w", f"{m}_sh_b", f"{m}_sh_w"]
    return out[cols]


def decompose_ge(panel: pd.DataFrame, regions: pd.DataFrame,
                 group: str = "region_wb") -> pd.DataFrame:
    """
    Per-year within/between decomposition of GE(0) and GE(1) by `group`
    (column in `regions`).  Returns columns:
        year, ge0_within, ge0_between, ge1_within, ge1_between
    """
    if group not in regions.columns:
        raise ValueError(f"group must be a column of regions; got {group!r}")

    # Attach group label to every panel row.
    key_map = regions.set_index("c3")[group]
    panel2 = panel.copy()
    panel2["_g"] = panel2["c3"].map(key_map)

    # Drop rows with missing group (shouldn't happen).
    panel2 = panel2.dropna(subset=["_g"])

    rows = []
    for yr, df_y in panel2.groupby("year", sort=True):
        r = _decompose_one_year(df_y, df_y["_g"])
        r["year"] = int(yr)
        rows.append(r)
    return pd.DataFrame(rows)[["year", "ge0_within", "ge0_between",
                               "ge1_within", "ge1_between"]]
