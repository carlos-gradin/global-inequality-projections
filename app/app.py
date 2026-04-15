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
                    between_within_country)

# ----------------------------------------------------------------------
# Paths & data loading (cached — only re-reads on file change).
# ----------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_parquet(DATA_DIR / "baseline_2022.parquet")
    regions  = pd.read_parquet(DATA_DIR / "regions.parquet")
    wpp      = pd.read_parquet(DATA_DIR / "wpp_population.parquet")
    return baseline, regions, wpp


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


@st.cache_data(show_spinner="Projecting distribution and computing indices…")
def run_pipeline(spec_dict: dict) -> dict:
    """Run project + indices + decomposition + by-group summary.
    We pass a plain dict so Streamlit's cache can hash the inputs."""
    baseline, regions, wpp = load_data()
    spec = GrowthSpec(**spec_dict)

    panel = project(baseline, regions, wpp, spec)
    idx   = indices(panel)
    # Country-level between/within decomposition (naive + Shapley) per
    # year.  Used by Tab 3.
    bw    = between_within_country(panel)

    # Per-region / per-incomegroup inequality at start & end only
    # (full time series would be heavier).
    yrs = [2022, spec.target_year]
    grp_reg = indices_by_group(panel, regions, "region_wb",   years=yrs)
    grp_inc = indices_by_group(panel, regions, "incomegroup", years=yrs)

    return {"panel": panel, "indices": idx,
            "between_within": bw,
            "group_region": grp_reg, "group_incgrp": grp_inc,
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

baseline, regions, wpp = load_data()
REGIONS      = sorted(regions["region_wb"].dropna().unique())
INCOMEGROUPS = ["Low income", "Lower middle income",
                "Upper middle income", "High income"]   # fixed order

with st.sidebar:
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

    target_year = st.slider("Target year", 2023, 2100, 2050, step=1)

    mode_label = st.radio(
        "Mode",
        ["A. Uniform world growth",
         "B. By World Bank region",
         "B. By World Bank income group",
         "C. By population group (B40 / M50 / T10)",
         "D. By region × population group",
         "D. By income group × population group"],
        index=0,
        help=(
            "A — same rate for every (country, percentile). "
            "B — per-region or per-income-group rate, same across "
            "percentiles within a country. "
            "C — three rates (B40, M50, T10) applied to every country. "
            "D — three rates per aggregate (region or income group). "
            "Modes C and D change within-country inequality."
        ),
    )

    st.markdown("---")
    # Default fallback rate, shown in all modes (used as baseline and as
    # fallback for any group or block without an explicit rate).
    base_rate_pct = st.number_input(
        "Baseline annual growth rate (%)",
        min_value=-5.0, max_value=15.0, value=3.0, step=0.1,
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
    if mode_label == "B. By World Bank region":
        aggregate = "region_wb"
        st.subheader("Rates by region (%)")
        for r in REGIONS:
            rates_by_group[r] = st.number_input(
                r, min_value=-5.0, max_value=15.0,
                value=base_rate_pct, step=0.1, key=f"reg_{r}"
            ) / 100.0
    elif mode_label == "B. By World Bank income group":
        aggregate = "incomegroup"
        st.subheader("Rates by income group (%)")
        for g in INCOMEGROUPS:
            rates_by_group[g] = st.number_input(
                g, min_value=-5.0, max_value=15.0,
                value=base_rate_pct, step=0.1, key=f"ig_{g}"
            ) / 100.0

    # --- Mode C ---
    elif mode_label.startswith("C."):
        st.subheader("Rates by population group (%)")
        st.caption(
            "Rates apply to each block **within every country** "
            "(e.g. the bottom 40 % of each country, not the bottom 40 % "
            "of the world pooled)."
        )
        block_rates["b40"] = st.number_input(
            "Bottom 40 %",  value=base_rate_pct, step=0.1,
            min_value=-5.0, max_value=15.0, key="c_b40") / 100.0
        block_rates["m50"] = st.number_input(
            "Middle 50 %",  value=base_rate_pct, step=0.1,
            min_value=-5.0, max_value=15.0, key="c_m50") / 100.0
        block_rates["t10"] = st.number_input(
            "Top 10 %",     value=base_rate_pct, step=0.1,
            min_value=-5.0, max_value=15.0, key="c_t10") / 100.0

    # --- Mode D ---
    # Inputs for Mode D are a (groups × 3 blocks) table, which does not
    # fit in the narrow sidebar.  We set the aggregate here, but the
    # editable table is rendered in the main pane below.
    elif mode_label.startswith("D."):
        aggregate = "region_wb" if "region" in mode_label else "incomegroup"
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
            min_value=-5.0, max_value=15.0,
            disabled=not split_china, key="cn") / 100.0
        india_rate = st.number_input(
            "India rate (%)", value=base_rate_pct, step=0.1,
            min_value=-5.0, max_value=15.0,
            disabled=not split_india, key="in") / 100.0
    else:
        # Modes C / D: 3 block rates per giant (shown only when enabled).
        if split_china:
            st.caption("China block rates (%)")
            china_block_rates["b40"] = st.number_input(
                "China B40", value=base_rate_pct, step=0.1,
                min_value=-5.0, max_value=15.0, key="cn_b40") / 100.0
            china_block_rates["m50"] = st.number_input(
                "China M50", value=base_rate_pct, step=0.1,
                min_value=-5.0, max_value=15.0, key="cn_m50") / 100.0
            china_block_rates["t10"] = st.number_input(
                "China T10", value=base_rate_pct, step=0.1,
                min_value=-5.0, max_value=15.0, key="cn_t10") / 100.0
        if split_india:
            st.caption("India block rates (%)")
            india_block_rates["b40"] = st.number_input(
                "India B40", value=base_rate_pct, step=0.1,
                min_value=-5.0, max_value=15.0, key="in_b40") / 100.0
            india_block_rates["m50"] = st.number_input(
                "India M50", value=base_rate_pct, step=0.1,
                min_value=-5.0, max_value=15.0, key="in_m50") / 100.0
            india_block_rates["t10"] = st.number_input(
                "India T10", value=base_rate_pct, step=0.1,
                min_value=-5.0, max_value=15.0, key="in_t10") / 100.0

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
        default_table = pd.DataFrame(
            {"b40": base_rate_pct, "m50": base_rate_pct, "t10": base_rate_pct},
            index=pd.Index(groups_D, name="group"),
        )
        edited = st.data_editor(
            default_table,
            key=f"d_table_{aggregate}",
            num_rows="fixed",
            column_config={
                "b40": st.column_config.NumberColumn("Bottom 40 %", step=0.1,
                                                    min_value=-5.0, max_value=15.0,
                                                    format="%.2f"),
                "m50": st.column_config.NumberColumn("Middle 50 %", step=0.1,
                                                    min_value=-5.0, max_value=15.0,
                                                    format="%.2f"),
                "t10": st.column_config.NumberColumn("Top 10 %",    step=0.1,
                                                    min_value=-5.0, max_value=15.0,
                                                    format="%.2f"),
            },
            use_container_width=True,
        )
        for g in groups_D:
            if g in edited.index:
                row = edited.loc[g]
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
PROJECTION_START = 2022


# ======================================================================
# Main pane — charts and tables
# ======================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Inequality measures", "Group shares & Palma",
     "Within / between decomposition", "Summary table"]
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
    fig.add_vline(
        x=PROJECTION_START, line_dash="dash", line_color="grey",
        annotation_text="Projection →", annotation_position="top",
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
    fig.add_vline(
        x=PROJECTION_START, line_dash="dash", line_color="grey",
        annotation_text="Projection →", annotation_position="top",
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
        f.add_vline(x=PROJECTION_START, line_dash="dash",
                    line_color="grey",
                    annotation_text="Projection →",
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
        f.add_vline(x=PROJECTION_START, line_dash="dash",
                    line_color="grey",
                    annotation_text="Projection →",
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

# ---- Tab 4: summary table (2022 → target year) ----
with tab4:
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
    "Population projections: UN WPP 2024, Medium variant. "
    "Inequality computed on pooled country-percentile cells weighted "
    "by population. In Modes A and B within-country shape is invariant; "
    "global inequality changes through differential growth across "
    "country aggregates and population-weight shifts over time."
)
