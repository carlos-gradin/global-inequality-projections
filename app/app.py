"""
app.py  —  Global Inequality Projections
-----------------------------------------

Streamlit web app that projects the 2022 global income distribution
(country x percentile, from WIID) forward to a user-chosen target year
using a user-chosen growth pattern, then reports inequality statistics
year by year.

This first iteration implements modes A (uniform world growth) and
B (by country aggregate: region or World Bank income group, with an
optional override for China and India).  Modes C and D (distinct
growth by population block) will be added next.

To run locally from the project root:
    streamlit run app/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import (GrowthSpec, project, indices, indices_by_group,
                    between_within_country, benchmark_growth,
                    country_indices,
                    PRO_POOR_PRESETS, PRO_POOR_PRESET_NAMES,
                    EXTREME_PRESET_NAME, apply_pro_poor_preset,
                    historical_default_rates, imf_default_rates)

# ----------------------------------------------------------------------
# Paths & data loading (cached — only re-reads on file change).
# ----------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_parquet(DATA_DIR / "baseline_2022.parquet")
    regions  = pd.read_parquet(DATA_DIR / "regions.parquet")
    wpp      = pd.read_parquet(DATA_DIR / "wpp_population.parquet")
    backfill = pd.read_parquet(DATA_DIR / "backfill_2023_2026.parquet")
    return baseline, regions, wpp, backfill


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    """Historical (1950..2021) global inequality indices, precomputed from
    wiidglobal_long.dta by scripts/build_history.py.  Fixed data — never
    depends on user inputs, so we cache once and reuse across runs."""
    return pd.read_parquet(DATA_DIR / "history_indices.parquet")


@st.cache_data(show_spinner=False)
def load_history_bw() -> pd.DataFrame:
    """Historical country-level between/within decomposition (naive +
    Shapley) for every year 1950..2021.  Precomputed and cached."""
    return pd.read_parquet(DATA_DIR / "history_between_within.parquet")


@st.cache_data(show_spinner=False)
def load_historical_panel() -> pd.DataFrame:
    """Historical WIID panel (c3, year, percentile, income_level) for
    2000..2022.  Used to compute the benchmark 'historical continuation'
    growth rates for the scenario-comparison tab."""
    return pd.read_parquet(DATA_DIR / "historical_2000_2022.parquet")


@st.cache_data(show_spinner=False)
def load_history_country_indices() -> pd.DataFrame:
    """Historical per-country-year within-country inequality indices
    (1950..2021), precomputed by scripts/build_history_within_country.py.
    Returns an empty DataFrame if the file is not yet built — the tab
    will then show projection years only."""
    path = DATA_DIR / "history_country_indices.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_imf_benchmark() -> pd.DataFrame:
    """IMF WEO April 2026 per-country annual real-GDP growth rates for
    years 2027..2031 (precomputed by scripts/build_imf_benchmark.py).
    Returns an empty DataFrame if the file is not yet built."""
    path = DATA_DIR / "imf_benchmark_2027_2031.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_historical_defaults() -> dict:
    """Population-weighted average CAGR (2012-2022) for the world, by
    region, and by income group."""
    hist = load_historical_panel()
    _, regions, _, _ = load_data()
    return historical_default_rates(hist, regions, n_years=10, base_year=2022)


@st.cache_data(show_spinner=False)
def load_imf_defaults() -> dict:
    """Population-weighted average IMF WEO forecast rate (2027-2031) for
    the world, by region, and by income group."""
    imf = load_imf_benchmark()
    _, regions, _, _ = load_data()
    return imf_default_rates(imf, regions)


# ----------------------------------------------------------------------
# Cached projection helpers for the "Compare scenarios" tab.
#
# These return ONLY the indices DataFrame (no decomposition) because the
# overlay plot needs just the time series of a single measure.  Much
# cheaper than run_pipeline, which also computes country-level
# decomposition.
# ----------------------------------------------------------------------

def _freeze_spec(spec_dict: dict) -> tuple:
    """Turn a spec dict (possibly with nested dicts) into a hashable
    representation so Streamlit's cache can key on it."""
    def _freeze(v):
        if isinstance(v, dict):
            return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
        return v
    return tuple(sorted((k, _freeze(v)) for k, v in spec_dict.items()))


@st.cache_data(show_spinner="Projecting scenario…")
def run_indices_only(_spec_key: tuple, spec_dict: dict) -> pd.DataFrame:
    """Project with a user spec and return the per-year inequality
    indices.  _spec_key (unused inside) is passed only so Streamlit's
    cache keys on a hashable tuple."""
    baseline, regions, wpp, backfill = load_data()
    spec = GrowthSpec(**spec_dict)
    panel = project(baseline, regions, wpp, spec, backfill)
    return indices(panel)


@st.cache_data(show_spinner="Projecting benchmark (historical continuation)…")
def run_indices_benchmark(n_years: int, target_year: int) -> pd.DataFrame:
    """Project using the benchmark 'historical continuation': per
    (country, percentile) CAGR over the last `n_years` of WIID applied
    from 2027 onwards (backfill 2023..2026 unchanged).  Returns the
    per-year inequality indices."""
    baseline, regions, wpp, backfill = load_data()
    hist = load_historical_panel()
    bench = benchmark_growth(hist, n_years=n_years)
    # Align percentile columns to int (baseline uses ints).
    bench.columns = bench.columns.astype(int)
    # Dummy spec (rates ignored because of the override); target_year matters.
    dummy = GrowthSpec(mode="uniform", target_year=int(target_year),
                       uniform_rate=0.0)
    panel = project(baseline, regions, wpp, dummy, backfill,
                    post2026_rate=bench)
    return indices(panel)


@st.cache_data(show_spinner="Projecting IMF benchmark…")
def run_indices_imf_benchmark(target_year: int) -> pd.DataFrame:
    """Project using the IMF WEO benchmark: per-country scalar growth
    rates for 2027..2031 (held constant afterwards), broadcast across
    all 100 percentiles (no change in within-country inequality).
    Returns the per-year inequality indices."""
    baseline, regions, wpp, backfill = load_data()
    imf = load_imf_benchmark()
    if imf.empty:
        return pd.DataFrame()
    # Pivot long → wide: rows = c3, cols = year, values = growth_rate.
    sy_wide = imf.pivot(index="c3", columns="year", values="growth_rate")
    dummy = GrowthSpec(mode="uniform", target_year=int(target_year),
                       uniform_rate=0.0)
    panel = project(baseline, regions, wpp, dummy, backfill,
                    post2026_scalar_by_year=sy_wide)
    return indices(panel)


@st.cache_data(show_spinner="Projecting distribution and computing indices…")
def run_pipeline(spec_dict: dict) -> dict:
    """Run project + indices + decomposition + by-group summary.
    We pass a plain dict so Streamlit's cache can hash the inputs."""
    baseline, regions, wpp, backfill = load_data()
    spec = GrowthSpec(**spec_dict)

    panel = project(baseline, regions, wpp, spec, backfill)
    idx   = indices(panel)
    # Country-level between/within decomposition (naive + Shapley) per
    # year.  Used by Tab 3.
    bw    = between_within_country(panel)

    # Per-region / per-incomegroup inequality at start & end only
    # (full time series would be heavier).
    yrs = [2022, spec.target_year]
    grp_reg = indices_by_group(panel, regions, "region_wb",   years=yrs)
    grp_inc = indices_by_group(panel, regions, "incomegroup", years=yrs)

    # Per-country-year within-country indices (used by the
    # "Within-country distributions" tab).
    c_idx = country_indices(panel)

    return {"panel": panel, "indices": idx,
            "between_within": bw,
            "group_region": grp_reg, "group_incgrp": grp_inc,
            "country_indices": c_idx,
            "target_year": spec.target_year}


# ======================================================================
# Sidebar — user inputs
# ======================================================================

st.set_page_config(page_title="Global inequality projections",
                   layout="wide", page_icon="📈")

st.title("Global inequality projections")
st.caption(
    "Project the 2022 world income distribution (WIID) forward under a "
    "chosen growth pattern, using UN WPP 2024 population projections."
)

baseline, regions, wpp, backfill = load_data()
REGIONS      = sorted(regions["region_wb"].dropna().unique())
INCOMEGROUPS = ["Low income", "Lower middle income",
                "Upper middle income", "High income"]   # fixed order

# Benchmark growth rates for preset base-rate sources.
HIST_DEFAULTS = load_historical_defaults()   # pop-weighted CAGR 2012-2022
IMF_DEFAULTS  = load_imf_defaults()          # pop-weighted IMF WEO 2027-2031

with st.sidebar:
    # Technical-note download link (PDF shipped with the app).
    _tn_path = DATA_DIR / "technical_note.pdf"
    if _tn_path.exists():
        with open(_tn_path, "rb") as _f:
            st.download_button(
                label="📄 Download technical note (PDF)",
                data=_f.read(),
                file_name="Technical note - Global inequality projections.pdf",
                mime="application/pdf",
                help="Methodology, data sources, and assumptions behind the "
                     "projections.",
                use_container_width=True,
            )

    st.header("Chart range")
    # History start year — only controls how much historical context is
    # shown on the charts; it does NOT affect the projection, which
    # always starts from 2022.
    history_from_year = st.slider(
        "Show history from", 1950, 2022, 1950, step=1,
        help="Earliest year shown on the time-series charts. Historical "
             "values (1950–2021) come from WIID and are precomputed; "
             "the projection always starts in 2022.",
    )

    st.header("Growth pattern")

    # User-chosen growth pattern applies from 2027 onwards.
    # 2023–2026 is backfilled with realised data (WB WDI) and short-term
    # forecasts (IMF WEO April 2026) — documented in the footer caption
    # at the bottom of the main page and in build_backfill.py.
    target_year = st.slider("Target year", 2027, 2100, 2050, step=1)

    mode_label = st.radio(
        "Mode",
        ["A. Uniform world growth",
         "B. By country aggregate",
         "C. By population group (B40 / M50 / T10)",
         "D. By country aggregate × population group"],
        index=0,
        help=(
            "A — same rate for every (country, percentile). "
            "B — per-aggregate rate (region or income group), same across "
            "percentiles within a country. "
            "C — three rates (B40, M50, T10) applied to every country. "
            "D — three rates per aggregate (region or income group). "
            "Modes C and D change within-country inequality."
        ),
    )

    # Aggregation dimension shared by Modes B and D.  A single radio, shown
    # only when the chosen mode uses it.  Default = region.
    if mode_label.startswith(("B.", "D.")):
        agg_choice = st.radio(
            "Aggregation",
            ["By World Bank region", "By World Bank income group"],
            index=0, key="sidebar_agg",
            help="Defines the 'country aggregate' used by Modes B and D.",
        )
    else:
        agg_choice = "By World Bank region"
    aggregate_sidebar = ("region_wb" if agg_choice == "By World Bank region"
                         else "incomegroup")

    st.markdown("---")
    # Default fallback rate, shown in all modes (used as baseline and as
    # fallback for any group or block without an explicit rate).
    base_rate_pct = st.number_input(
        "Baseline annual growth rate (%)",
        min_value=-99.0, value=3.0, step=0.1,
        help="Applied uniformly in Mode A, and as the default for "
             "group / block rates in the other modes.",
    )

    # ------------------------------------------------------------------
    # Per-mode inputs.
    # ------------------------------------------------------------------
    rates_by_group: dict[str, float] = {}
    block_rates: dict[str, float]    = {}
    block_rates_by_group: dict[str, dict[str, float]] = {}
    aggregate = "region_wb"

    # --- Mode B ---
    if mode_label.startswith("B."):
        aggregate = aggregate_sidebar
        # Base-rate source selector: lets the user start from historical
        # or IMF benchmark rates instead of the flat baseline.
        base_src_b = st.selectbox(
            "Base rate source",
            ["Custom (flat baseline)", "Historical trend (CAGR 2012–2022)",
             "IMF forecast (WEO 2027–2031)"],
            index=0, key="base_src_b",
            help="Pre-fill group rates from a benchmark. You can still "
                 "edit individual rates afterwards.",
        )
        if base_src_b.startswith("Historical"):
            _src = HIST_DEFAULTS[aggregate]
        elif base_src_b.startswith("IMF"):
            _src = IMF_DEFAULTS[aggregate]
        else:
            _src = {}
        if aggregate == "region_wb":
            st.subheader("Rates by region (%)")
            for r in REGIONS:
                rates_by_group[r] = st.number_input(
                    r, min_value=-99.0,
                    value=_src.get(r, base_rate_pct),
                    step=0.1, key=f"reg_{r}_{base_src_b}"
                ) / 100.0
        else:
            st.subheader("Rates by income group (%)")
            for g in INCOMEGROUPS:
                rates_by_group[g] = st.number_input(
                    g, min_value=-99.0,
                    value=_src.get(g, base_rate_pct),
                    step=0.1, key=f"ig_{g}_{base_src_b}"
                ) / 100.0

    # --- Mode C ---
    elif mode_label.startswith("C."):
        st.subheader("Rates by population group (%)")
        st.caption(
            "Rates apply to each block **within every country** "
            "(e.g. the bottom 40 % of each country, not the bottom 40 % "
            "of the world pooled)."
        )
        # Pro-poor preset selector
        preset_options_c = ["Custom"] + PRO_POOR_PRESET_NAMES
        preset_c = st.selectbox(
            "Scenario preset",
            preset_options_c,
            index=0,
            key="preset_c",
            help="Load empirically-calibrated pro-poor differentials "
                 "relative to the baseline rate. You can still edit "
                 "individual rates after loading a preset.",
        )
        if preset_c != "Custom":
            pp = apply_pro_poor_preset(base_rate_pct, preset_c)
            c_b40_val, c_m50_val, c_t10_val = pp["b40"], pp["m50"], pp["t10"]
            if preset_c == EXTREME_PRESET_NAME:
                st.caption(
                    f"**Extreme redistribution** (base {base_rate_pct:.1f} %): "
                    f"T10 = 0 %, M50 = base, B40 = {c_b40_val:.2f} % "
                    f"(absorbs residual so weighted average = base rate)."
                )
            else:
                diffs_c = PRO_POOR_PRESETS[preset_c]
                st.caption(
                    f"Preset applies to the baseline rate ({base_rate_pct:.1f} %): "
                    f"B40 {diffs_c['b40_diff']:+.1f} pp, "
                    f"M50 {diffs_c['m50_diff']:+.1f} pp, "
                    f"T10 {diffs_c['t10_diff']:+.1f} pp. "
                    f"Calibrated from historical pro-poor growth episodes "
                    f"(see technical note)."
                )
        else:
            c_b40_val = c_m50_val = c_t10_val = base_rate_pct
        # Include base_rate_pct in the key so that changing the baseline
        # resets the widgets to the correct preset-derived values.
        _ck = f"{preset_c}_{base_rate_pct:.2f}"
        block_rates["b40"] = st.number_input(
            "Bottom 40 %",  value=c_b40_val, step=0.1,
            min_value=-99.0, key=f"c_b40_{_ck}") / 100.0
        block_rates["m50"] = st.number_input(
            "Middle 50 %",  value=c_m50_val, step=0.1,
            min_value=-99.0, key=f"c_m50_{_ck}") / 100.0
        block_rates["t10"] = st.number_input(
            "Top 10 %",     value=c_t10_val, step=0.1,
            min_value=-99.0, key=f"c_t10_{_ck}") / 100.0

    # --- Mode D ---
    # Inputs for Mode D are a (groups × 3 blocks) table, which does not
    # fit in the narrow sidebar.  We set the aggregate here, but the
    # editable table is rendered in the main pane below.
    elif mode_label.startswith("D."):
        aggregate = aggregate_sidebar
        st.info("Edit the group × block rates table in the main pane.")

    # ------------------------------------------------------------------
    # Giants override.  Scalar fields in modes A/B, block fields in C/D.
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Giants override")

    is_block_mode = mode_label.startswith(("C.", "D."))

    split_china = st.checkbox("Use separate rate(s) for China", value=False)
    split_india = st.checkbox("Use separate rate(s) for India", value=False)

    china_rate = base_rate_pct / 100.0
    india_rate = base_rate_pct / 100.0
    china_block_rates: dict[str, float] = {}
    india_block_rates: dict[str, float] = {}

    if not is_block_mode:
        # Modes A / B: a single scalar per giant.
        china_rate = st.number_input(
            "China rate (%)", value=base_rate_pct, step=0.1,
            min_value=-99.0,
            disabled=not split_china, key="cn") / 100.0
        india_rate = st.number_input(
            "India rate (%)", value=base_rate_pct, step=0.1,
            min_value=-99.0,
            disabled=not split_india, key="in") / 100.0
    else:
        # Modes C / D: 3 block rates per giant (shown only when enabled).
        if split_china:
            st.caption("China block rates (%)")
            china_block_rates["b40"] = st.number_input(
                "China B40", value=base_rate_pct, step=0.1,
                min_value=-99.0, key="cn_b40") / 100.0
            china_block_rates["m50"] = st.number_input(
                "China M50", value=base_rate_pct, step=0.1,
                min_value=-99.0, key="cn_m50") / 100.0
            china_block_rates["t10"] = st.number_input(
                "China T10", value=base_rate_pct, step=0.1,
                min_value=-99.0, key="cn_t10") / 100.0
        if split_india:
            st.caption("India block rates (%)")
            india_block_rates["b40"] = st.number_input(
                "India B40", value=base_rate_pct, step=0.1,
                min_value=-99.0, key="in_b40") / 100.0
            india_block_rates["m50"] = st.number_input(
                "India M50", value=base_rate_pct, step=0.1,
                min_value=-99.0, key="in_m50") / 100.0
            india_block_rates["t10"] = st.number_input(
                "India T10", value=base_rate_pct, step=0.1,
                min_value=-99.0, key="in_t10") / 100.0

# ------------------------------------------------------------------
# Mode D main-pane table (needs more width than the sidebar offers).
# Rendered here, BEFORE we build the spec dict, so its values feed the
# pipeline on this same run.
# ------------------------------------------------------------------
if mode_label.startswith("D."):
    groups_D = REGIONS if aggregate == "region_wb" else INCOMEGROUPS
    with st.expander("Growth rates by group × population block (%)",
                     expanded=True):
        # Clarify what the rates actually apply to.  A rate of e.g. 3% in
        # "SSA × Bottom 40" grows the bottom 40 % WITHIN EACH SSA country,
        # not the bottom 40 % of the regionally-pooled SSA distribution.
        st.info(
            "**How these rates are applied.** The rate you type in a "
            "(group × block) cell is applied to that population block "
            "**within each country of the group**, not to the regionally-"
            "pooled distribution. For example, 3 % in *SSA × Bottom 40* "
            "means the bottom 40 % of each SSA country (Nigeria, Kenya, …) "
            "grows at 3 % — not the poorest 40 % of SSA as a whole."
        )
        # --- Base-rate source (sets per-group base rates) ---
        base_src_d = st.selectbox(
            "Base rate source",
            ["Custom (flat baseline)",
             "Historical trend (CAGR 2012–2022)",
             "IMF forecast (WEO 2027–2031)"],
            index=0, key="base_src_d",
            help="Pre-fill the Base % column from a benchmark. "
                 "You can still edit individual cells.",
        )
        if base_src_d.startswith("Historical"):
            _src_d = HIST_DEFAULTS[aggregate]
        elif base_src_d.startswith("IMF"):
            _src_d = IMF_DEFAULTS[aggregate]
        else:
            _src_d = {}
        init_base = pd.Series(
            [_src_d.get(g, base_rate_pct) for g in groups_D],
            index=groups_D,
        )

        # --- Pro-poor differential preset ---
        preset_options_d = ["Custom"] + PRO_POOR_PRESET_NAMES
        preset_d = st.selectbox(
            "Pro-poor scenario",
            preset_options_d,
            index=0,
            key="preset_d",
            help="Apply empirically-calibrated pro-poor differentials "
                 "on top of the base rates. Block columns adjust "
                 "automatically.",
        )
        editor_key = f"d_table_{aggregate}_{base_src_d}_{preset_d}"
        if preset_d != "Custom":
            base_vals = init_base.copy()
            # Incorporate any base-column edits the user already made
            # (stored in session state as deltas from the previous
            # default).  This keeps the disabled block columns in sync
            # with the edited base on re-runs.
            prev = st.session_state.get(editor_key)
            if prev and isinstance(prev, dict) and prev.get("edited_rows"):
                for row_idx_str, changes in prev["edited_rows"].items():
                    row_idx = int(row_idx_str)
                    if "base" in changes and 0 <= row_idx < len(groups_D):
                        base_vals.iloc[row_idx] = changes["base"]
            # Compute block rates from the preset for each group's base.
            b40_vals = np.array([apply_pro_poor_preset(b, preset_d)["b40"]
                                 for b in base_vals.values])
            m50_vals = np.array([apply_pro_poor_preset(b, preset_d)["m50"]
                                 for b in base_vals.values])
            t10_vals = np.array([apply_pro_poor_preset(b, preset_d)["t10"]
                                 for b in base_vals.values])
            default_table = pd.DataFrame({
                "base": base_vals.values,
                "b40":  b40_vals,
                "m50":  m50_vals,
                "t10":  t10_vals,
            }, index=pd.Index(groups_D, name="group"))
            if preset_d == EXTREME_PRESET_NAME:
                st.caption(
                    "**Extreme redistribution**: T10 = 0 %, M50 = base, "
                    "B40 absorbs the residual (= 1.25 × base) so the "
                    "population-weighted average equals the base rate."
                )
            else:
                diffs = PRO_POOR_PRESETS[preset_d]
                st.caption(
                    f"Edit **Base %** per group — block rates adjust "
                    f"automatically (B40 {diffs['b40_diff']:+.1f} pp, "
                    f"M50 {diffs['m50_diff']:+.1f} pp, "
                    f"T10 {diffs['t10_diff']:+.1f} pp). "
                    f"Calibrated from historical pro-poor growth episodes "
                    f"(see technical note)."
                )
        else:
            default_table = pd.DataFrame({
                "base": init_base.values,
                "b40":  init_base.values,
                "m50":  init_base.values,
                "t10":  init_base.values,
            }, index=pd.Index(groups_D, name="group"))
        # When a preset is active, block columns are derived from base;
        # disable them so the user edits only the base column.
        disabled_cols = (["b40", "m50", "t10"]
                         if preset_d != "Custom" else [])
        edited = st.data_editor(
            default_table,
            key=editor_key,
            num_rows="fixed",
            disabled=disabled_cols,
            column_config={
                "base": st.column_config.NumberColumn(
                    "Base %", step=0.1, min_value=-99.0, format="%.2f",
                    help="Average growth rate for this group"),
                "b40": st.column_config.NumberColumn("Bottom 40 %", step=0.1,
                                                    min_value=-99.0,
                                                    format="%.2f"),
                "m50": st.column_config.NumberColumn("Middle 50 %", step=0.1,
                                                    min_value=-99.0,
                                                    format="%.2f"),
                "t10": st.column_config.NumberColumn("Top 10 %",    step=0.1,
                                                    min_value=-99.0,
                                                    format="%.2f"),
            },
            use_container_width=True,
        )
        # Compute effective block rates from the edited table.
        for g in groups_D:
            if g in edited.index:
                row = edited.loc[g]
                if preset_d != "Custom":
                    # Derive block rates from the (possibly edited) base
                    # column via the preset (works for both differential
                    # and extreme presets).
                    b = float(row["base"])
                    pp = apply_pro_poor_preset(b, preset_d)
                    block_rates_by_group[g] = {
                        "b40": pp["b40"] / 100.0,
                        "m50": pp["m50"] / 100.0,
                        "t10": pp["t10"] / 100.0,
                    }
                else:
                    # Custom: use the block columns directly.
                    block_rates_by_group[g] = {
                        "b40": float(row["b40"]) / 100.0,
                        "m50": float(row["m50"]) / 100.0,
                        "t10": float(row["t10"]) / 100.0,
                    }

# Build the spec dict.  Mode letter -> engine mode string:
#   A -> 'uniform'
#   B -> 'by_aggregate'
#   C -> 'by_popgroup'
#   D -> 'by_aggregate_popgroup'
if mode_label.startswith("A."):
    mode = "uniform"
elif mode_label.startswith("B."):
    mode = "by_aggregate"
elif mode_label.startswith("C."):
    mode = "by_popgroup"
else:
    mode = "by_aggregate_popgroup"

spec_dict = dict(
    mode=mode,
    target_year=int(target_year),
    uniform_rate=base_rate_pct / 100.0,
    aggregate=aggregate,
    rates_by_group=rates_by_group,
    block_rates=block_rates,
    block_rates_by_group=block_rates_by_group,
    split_china=split_china, china_rate=china_rate,
    split_india=split_india, india_rate=india_rate,
    china_block_rates=china_block_rates,
    india_block_rates=india_block_rates,
)

# Streamlit's cache hashes a dict's items, but mutable dicts inside
# need frozen representation.  A tuple of sorted items does the trick.
result = run_pipeline(spec_dict)
idx    = result["indices"]
bw_proj = result["between_within"]
grp_r  = result["group_region"]
grp_i  = result["group_incgrp"]
panel  = result["panel"]

# ----------------------------------------------------------------------
# Combine historical (1950..2021) and projected (2022..target) indices
# into a single time series for the charts.  History is loaded from
# precomputed parquets — never depends on user inputs.
# ----------------------------------------------------------------------
history    = load_history()
history_bw = load_history_bw()

# Columns the history parquet has in common with `idx`.
_common_cols = [c for c in idx.columns if c in history.columns]
idx_plot = pd.concat(
    [history.loc[history["year"] >= history_from_year, _common_cols],
     idx[_common_cols]],
    ignore_index=True,
).sort_values("year").reset_index(drop=True)

# Same concat pattern for the between/within decomposition.
_bw_cols = [c for c in bw_proj.columns if c in history_bw.columns]
bw_plot = pd.concat(
    [history_bw.loc[history_bw["year"] >= history_from_year, _bw_cols],
     bw_proj[_bw_cols]],
    ignore_index=True,
).sort_values("year").reset_index(drop=True)

# Constant reused by the charts — marks where the projection begins.
PROJECTION_START = 2027   # first year of user-driven growth
BACKFILL_START   = 2023   # first year of WDI/WEO backfill


# ======================================================================
# Main pane — charts and tables
# ======================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Inequality measures", "Group shares & Palma",
     "Within / between decomposition",
     "Within-country distributions",
     "Summary table",
     "Compare scenarios"]
)

# ---- Tab 1: inequality measures over time ----
# Gini / MLD / Theil on the LEFT axis; GE(2) on the RIGHT axis because
# it generally takes larger values than the other three.  Both axes
# start at 0 so the series can be visually compared to that floor.
with tab1:
    fig = go.Figure()
    # Left-axis series
    for col, name in [("gini", "Gini"), ("ge0", "MLD (GE0)"),
                      ("ge1", "Theil (GE1)")]:
        fig.add_trace(go.Scatter(x=idx_plot["year"], y=idx_plot[col],
                                 mode="lines", name=name))
    # GE(2) on secondary axis — dashed line to signal the different scale.
    fig.add_trace(go.Scatter(x=idx_plot["year"], y=idx_plot["ge2"],
                             mode="lines", name="GE(2) (right axis)",
                             yaxis="y2", line=dict(dash="dot")))
    # Vertical line marking where the projection starts.
    # Backfill band (2023–2026): WDI+WEO fill, not user-driven.
    fig.add_vrect(
        x0=BACKFILL_START - 0.5, x1=PROJECTION_START - 0.5,
        fillcolor="lightgrey", opacity=0.25, line_width=0,
        annotation_text="Backfill (WDI/WEO)", annotation_position="top left",
    )
    # Vertical line where the USER-chosen growth pattern begins.
    fig.add_vline(
        x=PROJECTION_START, line_dash="dash", line_color="grey",
        annotation_text="User projection →", annotation_position="top",
    )
    fig.update_layout(
        title=f"Global inequality, {history_from_year} → {target_year}",
        # dtick=5 -> a tick label every 5 years (1950, 1955, ...).
        xaxis=dict(title="Year", dtick=5),
        yaxis=dict(title="Gini / MLD / Theil", rangemode="tozero"),
        yaxis2=dict(title="GE(2)", overlaying="y", side="right",
                    rangemode="tozero"),
        legend=dict(orientation="h", y=-0.2),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2: group shares (B40, M50, T10) + Palma on right axis ----
with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx_plot["year"], y=idx_plot["bottom40"]*100,
                             mode="lines", name="Bottom 40 %"))
    fig.add_trace(go.Scatter(x=idx_plot["year"], y=idx_plot["middle50"]*100,
                             mode="lines", name="Middle 50 %"))
    fig.add_trace(go.Scatter(x=idx_plot["year"], y=idx_plot["top10"]*100,
                             mode="lines", name="Top 10 %"))
    fig.add_trace(go.Scatter(x=idx_plot["year"], y=idx_plot["palma"],
                             mode="lines", name="Palma (right axis)",
                             yaxis="y2", line=dict(dash="dot")))
    # Backfill band (2023–2026): WDI+WEO fill, not user-driven.
    fig.add_vrect(
        x0=BACKFILL_START - 0.5, x1=PROJECTION_START - 0.5,
        fillcolor="lightgrey", opacity=0.25, line_width=0,
        annotation_text="Backfill (WDI/WEO)", annotation_position="top left",
    )
    # Vertical line where the USER-chosen growth pattern begins.
    fig.add_vline(
        x=PROJECTION_START, line_dash="dash", line_color="grey",
        annotation_text="User projection →", annotation_position="top",
    )
    fig.update_layout(
        title="Population-group income shares and Palma ratio",
        xaxis=dict(title="Year", dtick=5),          # tick every 5 years
        # rangemode="tozero" forces the y-axis to start at 0, even if
        # the data minimum is well above zero (e.g. Middle 50 % ~ 40 %).
        yaxis=dict(title="Income share (%)", rangemode="tozero"),
        yaxis2=dict(title="Palma ratio (Top10 / Bottom40)",
                    overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h", y=-0.2),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: country-level between/within decomposition ----
# For each measure we plot TWO charts side by side.
#
#   LEFT  — I(y), I(y^b), I(y^w):
#           y^b  = each individual gets their country mean
#                  (no WITHIN-country inequality)
#           y^w  = each country scaled to global mean
#                  (no BETWEEN-country inequality)
#           I(y^b) + I(y^w) = I(y) ONLY for MLD (GE0); for other
#           measures the two pieces do not sum to the total.
#
#   RIGHT — Shapley between / within:
#           sh_b = (I(y^b) + I(y) - I(y^w)) / 2
#           sh_w = (I(y^w) + I(y) - I(y^b)) / 2
#           sh_b + sh_w = I(y) for every measure, by construction.
with tab3:
    st.caption(
        "Country-level between/within decomposition. "
        "**Left**: naive components — I(y) total, I(y^b) on the "
        "country-smoothed distribution (no within-country inequality), "
        "I(y^w) on the country-standardized distribution (no between-"
        "country inequality). "
        "**Right**: Shapley decomposition — the two terms always sum to "
        "the total, for every measure."
    )

    MEASURE_LABELS = [
        ("gini", "Gini"),
        ("ge0",  "MLD (GE0)"),
        ("ge1",  "Theil (GE1)"),
        ("ge2",  "GE(2)"),
    ]

    def _bw_fig_naive(df: pd.DataFrame, m: str, label: str) -> go.Figure:
        """Left-hand figure: I(y), I(y^b) and I(y^w) as lines over time."""
        f = go.Figure()
        f.add_trace(go.Scatter(x=df["year"], y=df[m],
                               mode="lines", name=f"I(y) — total",
                               line=dict(color="black", width=2)))
        f.add_trace(go.Scatter(x=df["year"], y=df[f"{m}_b"],
                               mode="lines",
                               name="I(y^b) — smoothed (between)",
                               line=dict(color="#1f77b4")))
        f.add_trace(go.Scatter(x=df["year"], y=df[f"{m}_w"],
                               mode="lines",
                               name="I(y^w) — standardized (within)",
                               line=dict(color="#d62728")))
        f.add_vrect(x0=BACKFILL_START - 0.5, x1=PROJECTION_START - 0.5,
                    fillcolor="lightgrey", opacity=0.25, line_width=0)
        f.add_vline(x=PROJECTION_START, line_dash="dash",
                    line_color="grey",
                    annotation_text="User projection →",
                    annotation_position="top")
        f.update_layout(
            title=f"{label}: naive between/within",
            xaxis=dict(title="Year", dtick=5),      # tick every 5 years
            yaxis=dict(title=label, rangemode="tozero"),
            legend=dict(orientation="h", y=-0.2),
            height=380,
        )
        return f

    def _bw_fig_shapley(df: pd.DataFrame, m: str, label: str) -> go.Figure:
        """Right-hand figure: Shapley between + Shapley within stacked
        against I(y).  Because the two pieces sum to the total by
        construction, a stacked area gives a clean visual."""
        f = go.Figure()
        f.add_trace(go.Scatter(x=df["year"], y=df[f"{m}_sh_b"],
                               mode="lines",
                               name="Shapley between",
                               stackgroup="one",
                               line=dict(color="#1f77b4")))
        f.add_trace(go.Scatter(x=df["year"], y=df[f"{m}_sh_w"],
                               mode="lines",
                               name="Shapley within",
                               stackgroup="one",
                               line=dict(color="#d62728")))
        # Overlay the total as a solid line to visually confirm the stack.
        f.add_trace(go.Scatter(x=df["year"], y=df[m],
                               mode="lines", name="I(y) — total",
                               line=dict(color="black", width=2, dash="dot")))
        f.add_vrect(x0=BACKFILL_START - 0.5, x1=PROJECTION_START - 0.5,
                    fillcolor="lightgrey", opacity=0.25, line_width=0)
        f.add_vline(x=PROJECTION_START, line_dash="dash",
                    line_color="grey",
                    annotation_text="User projection →",
                    annotation_position="top")
        f.update_layout(
            title=f"{label}: Shapley between/within (sums to total)",
            xaxis=dict(title="Year", dtick=5),      # tick every 5 years
            yaxis=dict(title=label, rangemode="tozero"),
            legend=dict(orientation="h", y=-0.2),
            height=380,
        )
        return f

    for m, label in MEASURE_LABELS:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_bw_fig_naive(bw_plot, m, label),
                            use_container_width=True)
        with c2:
            st.plotly_chart(_bw_fig_shapley(bw_plot, m, label),
                            use_container_width=True)

# ---- Tab 4: within-country distributions ----
# Average (across countries) of a within-country inequality indicator,
# plotted for the World + the 7 WB regions (or 4 WB income groups).
# The indicator is computed separately for each country from its 100
# percentile incomes (within-country inequality only — no between
# component), then averaged across countries using either country
# populations as weights (default) or a simple unweighted mean.
with tab4:
    st.caption(
        "Average **within-country** inequality across countries. "
        "For each country we compute the indicator on its 100 percentile "
        "incomes (so this measures inequality *inside* countries only, "
        "ignoring between-country differences), then we average across "
        "countries — by default weighting each country by its total "
        "population. Unweighted shows each country counting equally."
    )

    MEASURE_CHOICES_WITHIN = [
        ("gini",      "Gini"),
        ("ge0",       "MLD (GE0)"),
        ("ge1",       "Theil (GE1)"),
        ("ge_m1",     "GE(-1)"),
        ("ge2",       "GE(2)"),
        ("atk_050",   "Atkinson(0.5)"),
        ("atk_1",     "Atkinson(1)"),
        ("atk_2",     "Atkinson(2)"),
        ("palma",     "Palma ratio (T10/B40)"),
        ("s80s20",    "S80/S20"),
        ("bottom40",  "Bottom 40 share"),
        ("top10",     "Top 10 share"),
        ("mean_income", "Mean income (PPP 2021 USD)"),
    ]
    MEASURE_KEY_TO_LABEL_W = dict(MEASURE_CHOICES_WITHIN)

    col_w1, col_w2, col_w3 = st.columns([2, 1, 1])
    with col_w1:
        within_measure_label = st.selectbox(
            "Indicator",
            [lbl for _, lbl in MEASURE_CHOICES_WITHIN],
            index=0, key="within_measure",
        )
        within_measure = next(k for k, lbl in MEASURE_CHOICES_WITHIN
                              if lbl == within_measure_label)
    with col_w2:
        within_weighted = st.checkbox(
            "Population-weighted", value=True, key="within_weighted",
            help="ON (default): each country's contribution is scaled by "
                 "its total population. OFF: simple mean, every country "
                 "counts equally.",
        )
    with col_w3:
        within_agg_label = st.selectbox(
            "Aggregation",
            ["By World Bank region", "By World Bank income group"],
            index=0, key="within_agg",
        )
    within_agg = ("region_wb" if within_agg_label == "By World Bank region"
                  else "incomegroup")

    # ---- Build a combined (historical + projected) per-country table ----
    c_idx_proj = result["country_indices"]      # 2022..target, from pipeline
    c_idx_hist = load_history_country_indices()  # 1950..2021, precomputed

    # Keep only years >= history_from_year on the history side (and drop
    # 2022 if it's there — it belongs to projection).
    if len(c_idx_hist):
        c_idx_hist = c_idx_hist[
            (c_idx_hist["year"] >= history_from_year) &
            (c_idx_hist["year"] < 2022)
        ]

    # Attach region / incomegroup label.
    reg_map = regions.set_index("c3")[within_agg]
    c_all = pd.concat([c_idx_hist, c_idx_proj], ignore_index=True)
    c_all["_g"] = c_all["c3"].map(reg_map)
    c_all = c_all.dropna(subset=[within_measure, "_g", "population"])

    # ---- Aggregate by group x year (World, and per-group). ----------------
    def _aggregate(df: pd.DataFrame, measure: str, weighted: bool,
                   group_col: str | None) -> pd.DataFrame:
        """Return a long DataFrame (year, group, value) with the
        cross-country mean of `measure`.  If `group_col` is None the
        aggregation is over all countries (World)."""
        keys = ["year"] + ([group_col] if group_col else [])
        grouped = df.groupby(keys, sort=True)
        if weighted:
            # Weighted mean: sum(pop * x) / sum(pop)
            num = grouped.apply(
                lambda s: (s[measure] * s["population"]).sum()
                          / s["population"].sum(),
                include_groups=False,
            )
        else:
            num = grouped[measure].mean()
        out = num.reset_index(name="value")
        if group_col is None:
            out["group"] = "World"
        else:
            out = out.rename(columns={group_col: "group"})
        return out[["year", "group", "value"]]

    world_df = _aggregate(c_all, within_measure, within_weighted, None)
    grp_df   = _aggregate(c_all, within_measure, within_weighted, "_g")

    plot_df = pd.concat([world_df, grp_df], ignore_index=True)

    # ---- Build the figure -------------------------------------------------
    # Scale income-shares to % for a friendlier display.
    def _scale(v: pd.Series) -> pd.Series:
        if within_measure in ("bottom40", "top10"):
            return v * 100.0
        return v

    fig = go.Figure()

    # Fixed color palette for the groups.
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
               "#ff7f0e", "#17becf", "#e377c2", "#bcbd22"]
    if within_agg == "region_wb":
        group_order = REGIONS
        group_kind  = "regions"
    else:
        group_order = INCOMEGROUPS
        group_kind  = "income groups"

    # Let the user add / remove groups on the chart (default: all +
    # World). "World" is a cross-country average over all countries.
    options_with_world = ["World"] + list(group_order)
    selected_groups = st.multiselect(
        f"Show World / {group_kind} on chart",
        options=options_with_world,
        default=options_with_world,
        key=f"within_groups_{within_agg}",
    )

    # Draw World first (as a thicker black line) so it sits on top.
    if "World" in selected_groups:
        w = plot_df[plot_df["group"] == "World"].sort_values("year")
        fig.add_trace(go.Scatter(
            x=w["year"], y=_scale(w["value"]),
            mode="lines", name="World",
            line=dict(color="black", width=3),
        ))

    for i, g in enumerate(group_order):
        if g not in selected_groups:
            continue
        sub = plot_df[plot_df["group"] == g].sort_values("year")
        if not len(sub):
            continue
        fig.add_trace(go.Scatter(
            x=sub["year"], y=_scale(sub["value"]),
            mode="lines", name=g,
            line=dict(color=palette[i % len(palette)], width=2),
        ))

    # Backfill band + user-projection start line (same as other tabs).
    fig.add_vrect(
        x0=BACKFILL_START - 0.5, x1=PROJECTION_START - 0.5,
        fillcolor="lightgrey", opacity=0.25, line_width=0,
        annotation_text="Backfill (WDI/WEO)",
        annotation_position="top left",
    )
    fig.add_vline(
        x=PROJECTION_START, line_dash="dash", line_color="grey",
        annotation_text="User projection →", annotation_position="top",
    )

    y_title = MEASURE_KEY_TO_LABEL_W[within_measure]
    if within_measure in ("bottom40", "top10"):
        y_title += " (%)"

    wt_tag = "population-weighted" if within_weighted else "unweighted"
    fig.update_layout(
        title=f"Average within-country {MEASURE_KEY_TO_LABEL_W[within_measure]} "
              f"({wt_tag}), {history_from_year} → {target_year}",
        xaxis=dict(title="Year", dtick=5),
        yaxis=dict(title=y_title, rangemode="tozero"),
        legend=dict(orientation="h", y=-0.2),
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    if not len(c_idx_hist):
        st.info(
            "Historical per-country indices not yet built — showing "
            "projection years only.  Run "
            "`python scripts/build_history_within_country.py` from the "
            "project root to precompute 1950–2021."
        )

    # ------------------------------------------------------------------
    # Second chart: per-country evolution of the same indicator.
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Evolution by country")
    st.caption(
        "Same indicator, but shown for individual countries instead of "
        "regional / income-group averages. Defaults to India, China, "
        "Brazil, and the United States; add or remove any country below."
    )

    # Country-picker options: sorted by country name, ISO3 shown in label.
    c3_to_name = regions.set_index("c3")["country"].to_dict()
    country_options = sorted(
        [c for c in c_all["c3"].unique() if c in c3_to_name],
        key=lambda c: c3_to_name[c],
    )
    country_labels = {c: f"{c3_to_name[c]} ({c})" for c in country_options}

    DEFAULT_COUNTRIES = [c for c in ["IND", "CHN", "BRA", "USA"]
                         if c in country_options]
    selected_c3 = st.multiselect(
        "Countries",
        options=country_options,
        default=DEFAULT_COUNTRIES,
        format_func=lambda c: country_labels[c],
        key="within_countries",
    )

    if selected_c3:
        fig_c = go.Figure()
        palette_c = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                     "#17becf", "#e377c2", "#bcbd22", "#8c564b", "#7f7f7f"]
        for i, c3 in enumerate(selected_c3):
            sub = (c_all[c_all["c3"] == c3]
                   .sort_values("year")[["year", within_measure]])
            if not len(sub):
                continue
            fig_c.add_trace(go.Scatter(
                x=sub["year"], y=_scale(sub[within_measure]),
                mode="lines", name=country_labels[c3],
                line=dict(color=palette_c[i % len(palette_c)], width=2),
            ))

        fig_c.add_vrect(
            x0=BACKFILL_START - 0.5, x1=PROJECTION_START - 0.5,
            fillcolor="lightgrey", opacity=0.25, line_width=0,
            annotation_text="Backfill (WDI/WEO)",
            annotation_position="top left",
        )
        fig_c.add_vline(
            x=PROJECTION_START, line_dash="dash", line_color="grey",
            annotation_text="User projection →", annotation_position="top",
        )
        fig_c.update_layout(
            title=f"Within-country {MEASURE_KEY_TO_LABEL_W[within_measure]} "
                  f"by country, {history_from_year} → {target_year}",
            xaxis=dict(title="Year", dtick=5),
            yaxis=dict(title=y_title, rangemode="tozero"),
            legend=dict(orientation="h", y=-0.2),
            height=520,
        )
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.caption("Pick at least one country to show the chart.")

# ---- Tab 5: summary table (2022 → target year) ----
with tab5:
    tgt = result["target_year"]
    st.subheader(f"Change in inequality: 2022 → {tgt}")

    def _pivot(df_long: pd.DataFrame, cols=("gini", "ge0", "ge1",
                                           "bottom40", "top10", "palma")) -> pd.DataFrame:
        """Wide table: one row per group, columns for each measure at
        2022 and target year, plus the percentage-point (or %) change."""
        w = df_long.pivot(index="group", columns="year", values=list(cols))
        # Flatten MultiIndex columns -> '<measure>_<year>'
        w.columns = [f"{m}_{y}" for m, y in w.columns]
        # Add change columns
        for m in cols:
            w[f"{m}_change"] = w[f"{m}_{tgt}"] - w[f"{m}_2022"]
        return w.round(4)

    # World row (from the global indices table).
    world_row = {
        "group": "World",
        **{f"{m}_2022": idx.loc[idx["year"] == 2022, m].iloc[0]
           for m in ("gini", "ge0", "ge1", "bottom40", "top10", "palma")},
        **{f"{m}_{tgt}": idx.loc[idx["year"] == tgt, m].iloc[0]
           for m in ("gini", "ge0", "ge1", "bottom40", "top10", "palma")},
    }
    for m in ("gini", "ge0", "ge1", "bottom40", "top10", "palma"):
        world_row[f"{m}_change"] = world_row[f"{m}_{tgt}"] - world_row[f"{m}_2022"]
    world_df = pd.DataFrame([world_row]).set_index("group")

    w_reg = _pivot(grp_r)
    w_inc = _pivot(grp_i)

    st.markdown("**World**")
    st.dataframe(world_df, use_container_width=True)

    st.markdown("**By region (within-group inequality)**")
    st.dataframe(w_reg, use_container_width=True)

    st.markdown("**By income group (within-group inequality)**")
    st.dataframe(w_inc, use_container_width=True)

# ---- Tab 6: scenario comparison ----
# Not available for Mode A (a single world rate has nothing meaningful
# to compare — saving a second 'scenario' would differ only in that rate,
# which is easier to eyeball in Tab 1).
#
# Available scenarios to overlay on a single-indicator plot:
#   (a) Current scenario   — whatever the user has chosen in the sidebar.
#   (b) Benchmark           — historical continuation: per-(c3, percentile)
#                              CAGR over the last N years of WIID applied
#                              from 2027 onwards (backfill unchanged).
#                              Optional (checkbox); default ON.
#   (c) Saved scenarios     — the user can save the current spec under a
#                              name; saved scenarios persist for the
#                              browser session and are per-mode (a
#                              scenario saved in Mode C is only listed in
#                              Mode C's comparison).
MEASURE_CHOICES = [
    ("gini",      "Gini"),
    ("ge0",       "MLD (GE0)"),
    ("ge1",       "Theil (GE1)"),
    ("ge_m1",     "GE(-1)"),
    ("ge2",       "GE(2)"),
    ("atk_050",   "Atkinson(0.5)"),
    ("atk_1",     "Atkinson(1)"),
    ("atk_2",     "Atkinson(2)"),
    ("palma",     "Palma ratio (T10/B40)"),
    ("s80s20",    "S80/S20"),
    ("bottom40",  "Bottom 40 share"),
    ("top10",     "Top 10 share"),
    ("mean_income", "Mean income (PPP 2021 USD)"),
]
MEASURE_KEY_TO_LABEL = dict(MEASURE_CHOICES)

with tab6:
    if mode == "uniform":
        st.info(
            "Scenario comparison is only meaningful in Modes B, C, and D "
            "(where growth varies across countries, groups, or population "
            "blocks). Pick a different mode in the sidebar to use this tab."
        )
    else:
        st.caption(
            "Compare the evolution of a single inequality indicator under "
            "the current scenario, a historical-continuation benchmark, "
            "and any scenarios you save from the sidebar. Saved scenarios "
            "persist for this browser session and are kept separately per "
            "mode."
        )

        # Session-state keys — per-mode so saved scenarios don't bleed
        # across incompatible specs (a Mode-C spec has 3 block rates,
        # Mode-D has a matrix, etc.).
        scen_key = f"scenarios_{mode}"
        if scen_key not in st.session_state:
            st.session_state[scen_key] = []   # list of (name, spec_dict)

        col_top1, col_top2, col_top3, col_top4 = st.columns([2, 1, 1, 1])
        with col_top1:
            measure_label = st.selectbox(
                "Indicator",
                [label for _, label in MEASURE_CHOICES],
                index=0,    # default Gini
            )
            measure = next(k for k, lbl in MEASURE_CHOICES
                           if lbl == measure_label)
        with col_top2:
            include_benchmark = st.checkbox(
                "Include benchmark",
                value=True,
                help="Historical continuation: each (country, percentile) "
                     "is projected at its own compound annual growth rate "
                     "over the last N years of WIID (default N=10).",
            )
        with col_top3:
            bench_n_years = st.slider(
                "Benchmark window (years)", 5, 20, 10, step=1,
                help="Length of the WIID window used to estimate the "
                     "per-(country, percentile) CAGR for the benchmark.",
            )
        with col_top4:
            include_imf = st.checkbox(
                "Include IMF benchmark",
                value=False,
                help="IMF WEO April 2026 per-country growth rates for "
                     "2027..2031 (held constant afterwards), applied "
                     "uniformly across all percentiles (no change in "
                     "within-country inequality).",
            )

        # ---- Save / clear controls ---------------------------------------
        st.markdown("**Saved scenarios**")
        col_s1, col_s2, col_s3 = st.columns([3, 1, 1])
        with col_s1:
            new_name = st.text_input(
                "Name", value="",
                placeholder=f"e.g. 'convergence' or 'SSA high growth'",
                label_visibility="collapsed",
            )
        with col_s2:
            save_clicked = st.button("Save current scenario",
                                     use_container_width=True)
        with col_s3:
            clear_clicked = st.button("Clear all", type="secondary",
                                      use_container_width=True)

        if save_clicked:
            if not new_name.strip():
                st.warning("Please type a name before saving.")
            else:
                existing_names = {n for n, _ in st.session_state[scen_key]}
                if new_name.strip() in existing_names:
                    st.warning(f"A scenario named {new_name.strip()!r} "
                               "already exists in this mode.")
                else:
                    # Copy the spec dict so later sidebar edits don't mutate
                    # the saved scenario.
                    import copy
                    st.session_state[scen_key].append(
                        (new_name.strip(), copy.deepcopy(spec_dict))
                    )
                    st.rerun()
        if clear_clicked:
            st.session_state[scen_key] = []
            st.rerun()

        saved = st.session_state[scen_key]
        if saved:
            saved_names = [n for n, _ in saved]
            selected_names = st.multiselect(
                "Show on chart:",
                options=saved_names,
                default=saved_names,
            )
        else:
            selected_names = []
            st.caption("No saved scenarios yet. Configure the sidebar and "
                       "click *Save current scenario* to add one.")

        # ---- Build the overlay figure ------------------------------------
        fig = go.Figure()

        # Scale the series for a couple of measures where a friendlier
        # display makes sense (percentage shares).
        def _yvals(df, m):
            v = df[m]
            if m in ("bottom40", "top10"):
                return v * 100.0
            return v

        # Historical segment (history_from_year .. 2021) drawn once as a
        # thin grey line.  Every scenario shares the same history by
        # construction, so we do not repeat it for each curve.
        if measure in history.columns:
            hist_seg = history[
                (history["year"] >= history_from_year) &
                (history["year"] <= 2021)
            ]
            if len(hist_seg):
                fig.add_trace(go.Scatter(
                    x=hist_seg["year"], y=_yvals(hist_seg, measure),
                    mode="lines", name="Historical (WIID)",
                    line=dict(color="#7f7f7f", width=1.5),
                ))

        # (a) Current scenario (already computed in run_pipeline).
        fig.add_trace(go.Scatter(
            x=idx["year"], y=_yvals(idx, measure),
            mode="lines", name="Current scenario",
            line=dict(color="black", width=3),
        ))

        # (b) Benchmark.
        if include_benchmark:
            bench_idx = run_indices_benchmark(int(bench_n_years),
                                              int(target_year))
            fig.add_trace(go.Scatter(
                x=bench_idx["year"], y=_yvals(bench_idx, measure),
                mode="lines",
                name=f"Benchmark (hist. continuation, N={bench_n_years})",
                line=dict(color="#7f7f7f", width=2, dash="dash"),
            ))

        # (b') IMF WEO benchmark.
        if include_imf:
            imf_idx = run_indices_imf_benchmark(int(target_year))
            if not imf_idx.empty:
                fig.add_trace(go.Scatter(
                    x=imf_idx["year"], y=_yvals(imf_idx, measure),
                    mode="lines",
                    name="IMF benchmark (WEO 2027–2031, then held)",
                    line=dict(color="#8c564b", width=2, dash="dot"),
                ))
            else:
                st.warning(
                    "IMF benchmark parquet not found. Run "
                    "`python scripts/build_imf_benchmark.py` to build it."
                )

        # (c) Saved scenarios the user picked.
        palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                   "#ff7f0e", "#17becf", "#e377c2", "#bcbd22"]
        for i, name in enumerate(selected_names):
            spec_d = dict(next(s for n, s in saved if n == name))
            # Force all saved scenarios to use the CURRENT target year so
            # all curves cover the same horizon on the chart.
            spec_d["target_year"] = int(target_year)
            scen_idx = run_indices_only(_freeze_spec(spec_d), spec_d)
            fig.add_trace(go.Scatter(
                x=scen_idx["year"], y=_yvals(scen_idx, measure),
                mode="lines", name=name,
                line=dict(color=palette[i % len(palette)], width=2),
            ))

        # Backfill band + projection start line — same look as other tabs.
        fig.add_vrect(
            x0=BACKFILL_START - 0.5, x1=PROJECTION_START - 0.5,
            fillcolor="lightgrey", opacity=0.25, line_width=0,
            annotation_text="Backfill (WDI/WEO)",
            annotation_position="top left",
        )
        fig.add_vline(
            x=PROJECTION_START, line_dash="dash", line_color="grey",
            annotation_text="User projection →", annotation_position="top",
        )

        y_axis_title = MEASURE_KEY_TO_LABEL[measure]
        if measure in ("bottom40", "top10"):
            y_axis_title += " (%)"

        fig.update_layout(
            title=f"{MEASURE_KEY_TO_LABEL[measure]}: scenario comparison",
            xaxis=dict(title="Year", dtick=5),
            yaxis=dict(title=y_axis_title, rangemode="tozero"),
            legend=dict(orientation="h", y=-0.2),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Small table with the saved-scenario specs, so users can remember
        # what each saved name stood for.
        if saved:
            with st.expander("Saved-scenario details", expanded=True):
                # Human-readable summary of each saved scenario.  One line
                # per scenario, listing the rates that actually drive the
                # projection (base rate + mode-specific rates + giants).
                MODE_LABELS = {
                    "uniform": "A — uniform",
                    "by_aggregate": "B — by aggregate",
                    "by_popgroup":  "C — by population group",
                    "by_aggregate_popgroup": "D — by aggregate × block",
                }

                def _fmt_pct(x: float) -> str:
                    return f"{x * 100:.2f}%"

                def _describe(s: dict) -> str:
                    parts: list[str] = []
                    m = s.get("mode")
                    parts.append(f"Base rate {_fmt_pct(s.get('uniform_rate', 0.0))}")
                    if m == "by_aggregate":
                        agg = s.get("aggregate", "region_wb")
                        rbg = s.get("rates_by_group") or {}
                        if rbg:
                            label = "region" if agg == "region_wb" else "income group"
                            per_grp = ", ".join(f"{g}={_fmt_pct(r)}"
                                                for g, r in rbg.items())
                            parts.append(f"by {label}: {per_grp}")
                    elif m == "by_popgroup":
                        br = s.get("block_rates") or {}
                        if br:
                            parts.append("blocks: "
                                         + ", ".join(f"{b}={_fmt_pct(r)}"
                                                     for b, r in br.items()))
                    elif m == "by_aggregate_popgroup":
                        agg = s.get("aggregate", "region_wb")
                        brg = s.get("block_rates_by_group") or {}
                        if brg:
                            label = "region" if agg == "region_wb" else "income group"
                            # Compact: "SSA (b40=2%, m50=3%, t10=4%); …"
                            groups_desc = []
                            for g, blocks in brg.items():
                                bd = ", ".join(f"{b}={_fmt_pct(r)}"
                                               for b, r in blocks.items())
                                groups_desc.append(f"{g} ({bd})")
                            parts.append(f"by {label} × block: "
                                         + "; ".join(groups_desc))
                    # Giants
                    if s.get("split_china"):
                        if s.get("china_block_rates"):
                            bd = ", ".join(f"{b}={_fmt_pct(r)}"
                                           for b, r in s["china_block_rates"].items())
                            parts.append(f"China ({bd})")
                        else:
                            parts.append(f"China={_fmt_pct(s.get('china_rate', 0.0))}")
                    if s.get("split_india"):
                        if s.get("india_block_rates"):
                            bd = ", ".join(f"{b}={_fmt_pct(r)}"
                                           for b, r in s["india_block_rates"].items())
                            parts.append(f"India ({bd})")
                        else:
                            parts.append(f"India={_fmt_pct(s.get('india_rate', 0.0))}")
                    return " · ".join(parts)

                rows = [{"Name":  n,
                         "Mode":  MODE_LABELS.get(s.get("mode"), s.get("mode")),
                         "Rates": _describe(s)}
                        for n, s in saved]
                st.dataframe(pd.DataFrame(rows),
                             hide_index=True,
                             use_container_width=True)


# ======================================================================
# Downloads
# ======================================================================

st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download global indices (CSV)",
        data=idx.to_csv(index=False).encode("utf-8"),
        file_name=f"indices_2022_{result['target_year']}.csv",
        mime="text/csv",
    )
with c2:
    st.download_button(
        "Download projected panel (CSV)",
        data=panel.to_csv(index=False).encode("utf-8"),
        file_name=f"panel_2022_{result['target_year']}.csv",
        mime="text/csv",
    )

st.caption(
    "Baseline: WIID (wiidglobal_long.dta), 2022 cross-section, 211 "
    "countries × 100 percentiles. "
    "Backfill 2023–2026: WB WDI realised growth (2023–2024) + IMF WEO "
    "April 2026 estimates/forecasts (2025–2026), applied uniformly "
    "across each country's percentiles (within-country shape held "
    "constant). The 22 countries with neither WDI nor WEO coverage in "
    "at least one year (small territories plus CUB, PRK, XKX, PSE, "
    "AFG, LBN, LKA, SYR) are imputed with the population-weighted "
    "mean growth of their World Bank region. "
    "Population projections: UN WPP 2024, Medium variant. "
    "Inequality computed on pooled country-percentile cells weighted "
    "by population. In Modes A and B within-country shape is invariant; "
    "global inequality changes through differential growth across "
    "country aggregates and population-weight shifts over time."
)
