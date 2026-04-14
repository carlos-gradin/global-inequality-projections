# Global Inequality Projections

A Streamlit web app that projects the 2022 world income distribution (from the
WIID cross-country country-percentile database) forward to a user-chosen year
under different growth scenarios, and reports global inequality indices — with
1950–2021 historical context drawn from WIID for reference.

## What it does

- Start from the 2022 WIID distribution (211 countries × 100 percentiles).
- Apply annual real-growth rates chosen by the user, with six scenario modes:
  - **A.** Uniform world growth
  - **B.** By World Bank region
  - **B.** By World Bank income group
  - **C.** By population block (Bottom 40 / Middle 50 / Top 10)
  - **D.** By region × population block
  - **D.** By income group × population block
  - Optional per-country override for China and India.
- Project forward year by year to the chosen target year using UN WPP 2024
  medium-variant country populations.
- Report Gini, GE(-1), MLD (GE0), Theil (GE1), GE(2), Atkinson (four values),
  income shares (Bottom 20/40, Middle 50, Top 10/20), Palma, S80/S20.
- **Between/within decomposition** at the country level, both naive
  (I(y^b), I(y^w)) and Shapley-style (which always sums to the total).
- Historical context (1950–2021) is precomputed and stacked seamlessly in
  front of the projection, with a vertical line marking where the projection
  starts.

## Run locally

```bash
python -m pip install -r app/requirements.txt
python -m streamlit run app/app.py
```

A browser tab opens at `http://localhost:8501`.

## Project layout

```
app/
├── app.py                ← Streamlit UI (4 tabs)
├── engine.py             ← projection + inequality formulas
├── requirements.txt
└── data/
    ├── baseline_2022.parquet         (311 KB — 2022 country × percentile)
    ├── regions.parquet               (12 KB — country → region / income group)
    ├── wpp_population.parquet        (142 KB — UN WPP 2024 2022–2100)
    ├── history_indices.parquet       (22 KB — WIID 1950–2021 global indices)
    └── history_between_within.parquet (25 KB — country between/within 1950–2021)

scripts/
├── build_baseline.py     ← (re)build baseline + regions from wiidglobal_long.dta
├── build_history.py      ← (re)build historical parquets from wiidglobal_long.dta
└── fetch_wpp.py          ← refresh UN WPP 2024 population projections
```

The `scripts/` helpers are only needed when the source data changes. The
app itself runs entirely from the five parquets in `app/data/`.

## Data sources

- **WIID Companion (WIID Global)** — UNU-WIDER, country × percentile income
  distribution. <https://www.wider.unu.edu/database/wiid>
- **UN World Population Prospects 2024** — medium variant country populations.
  <https://population.un.org/wpp/>

## Method notes

- Weighting: each country-percentile cell counts as 1 % of the country's
  population in that year.
- Gini: weighted trapezoidal Lorenz formula.
- GE(α): standard closed form, with special cases for α = 0 (MLD) and
  α = 1 (Theil).
- Between/within: two counterfactual distributions on the same sample —
  `y^b` replaces every cell with its country mean (no within-country
  inequality); `y^w` scales each country to the global mean (no
  between-country inequality). Shapley components are
  `[I(y^b) + I(y) - I(y^w)] / 2` and `[I(y^w) + I(y) - I(y^b)] / 2`
  and always sum to `I(y)` for every measure.
- Caveats: inequality is measured on WIID-reported incomes at the
  country-percentile level; the "global" pool ignores PPP/CPI updates
  post-2022. Comparisons across years are relative, not absolute.
