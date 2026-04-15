"""
build_technical_note_pdf.py
---------------------------
Generates a PDF copy of the technical note (same content as
'Technical note - Global inequality projections.docx', built by
scripts/build_technical_note.py) using reportlab, so the PDF can be
shipped with the Streamlit app and served via a download button.

Output: app/data/technical_note.pdf (committed to the deploy repo).

Run whenever the docx changes:
    python scripts/build_technical_note_pdf.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                PageBreak, Table, TableStyle, KeepTogether)


ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "app" / "data" / "technical_note.pdf"

URL_WIID = "https://www.wider.unu.edu/database/world-income-inequality-database-wiid"
URL_WIID_COMPANION = ("https://www.wider.unu.edu/database/"
                      "wiid-companion-world-income-inequality-database")
URL_WDI  = "https://databank.worldbank.org/source/world-development-indicators"
URL_WEO  = "https://www.imf.org/en/Publications/WEO"
URL_WPP  = "https://population.un.org/wpp/"
URL_GEP  = ("https://www.worldbank.org/en/publication/"
            "global-economic-prospects")


def _link(text: str, url: str) -> str:
    """Return a reportlab-style hyperlink tag."""
    return f'<link href="{url}" color="#0563C1"><u>{text}</u></link>'


# ---------------------------------------------------------------------
# Style sheet
# ---------------------------------------------------------------------
styles = getSampleStyleSheet()

ST_H0 = ParagraphStyle("H0", parent=styles["Heading1"], fontSize=20,
                       leading=24, spaceAfter=6, textColor=colors.black)
ST_SUB = ParagraphStyle("Sub", parent=styles["BodyText"], fontSize=12,
                        leading=15, fontName="Helvetica-Oblique",
                        spaceAfter=2)
ST_META = ParagraphStyle("Meta", parent=styles["BodyText"], fontSize=10,
                         leading=13, fontName="Helvetica-Oblique",
                         spaceAfter=18)
ST_H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14,
                       leading=18, spaceBefore=14, spaceAfter=6)
ST_H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12,
                       leading=15, spaceBefore=10, spaceAfter=4)
ST_P  = ParagraphStyle("P",  parent=styles["BodyText"], fontSize=10,
                       leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
ST_BUL = ParagraphStyle("Bul", parent=ST_P, leftIndent=18,
                        bulletIndent=6, spaceAfter=3)
ST_TH  = ParagraphStyle("TH", parent=styles["BodyText"], fontSize=9,
                        leading=11, textColor=colors.white,
                        fontName="Helvetica-Bold")
ST_TD  = ParagraphStyle("TD", parent=styles["BodyText"], fontSize=8,
                        leading=10)
ST_TDB = ParagraphStyle("TDB", parent=ST_TD, fontName="Helvetica-Bold")


def P(text: str, style=ST_P):
    return Paragraph(text, style)


def H1(text): return Paragraph(text, ST_H1)
def H2(text): return Paragraph(text, ST_H2)


def BUL(text):
    return Paragraph(f"• {text}", ST_BUL)


# ---------------------------------------------------------------------
# Build story
# ---------------------------------------------------------------------
story = []

story.append(Paragraph("Technical note: Global inequality projections", ST_H0))
story.append(Paragraph(
    "Methodology, data, and assumptions behind the Streamlit app", ST_SUB))
story.append(Paragraph("Carlos Gradin  |  April 2026", ST_META))

# -------- 1. Overview --------
story.append(H1("1. Overview"))
story.append(P(
    "This note documents the data, methodology, and modelling assumptions "
    "behind the 'Global Inequality Projections' Streamlit app. The app "
    "takes the 2022 cross-country income distribution from the WIID "
    "Companion database, backfills 2023–2026 with realised and short-term "
    "forecast growth from external sources, and then projects the full "
    "global distribution forward under user-chosen annual real-growth "
    "patterns until a target year of up to 2100. Outputs include a "
    "battery of global inequality measures and their within/between "
    "decomposition, the average within-country inequality across "
    "countries (for the world and for each region or income group), and "
    "a scenario-comparison view where the user can overlay named "
    "scenarios against a historical-continuation benchmark and an IMF "
    "WEO benchmark."
))
story.append(P(
    "The design is deliberately simple and transparent. Within-country "
    "distributional shape is held fixed throughout (income at every "
    "percentile of a country grows by the same country-specific rate in "
    "a given year), so the only sources of change in global inequality "
    "over time are (i) differential growth across countries or "
    "country-aggregate groups, (ii) differential growth across "
    "population blocks when the user chooses a Mode-C or Mode-D "
    "scenario, and (iii) shifts in country population weights from UN "
    "WPP 2024 medium-variant projections."
))

# -------- 2. Data sources --------
story.append(H1("2. Data sources"))

story.append(H2("2.1 Baseline distribution (2022)"))
story.append(P(
    f"The baseline is built from the {_link('WIID Companion', URL_WIID_COMPANION)} "
    "dataset (wiidglobal_long.dta), a harmonised 1950–2022 extension of "
    f"the UNU-WIDER {_link('World Income Inequality Database (WIID)', URL_WIID)}, "
    "restricted to the 2022 country cross-section. Each country is "
    "represented by 100 percentile observations of per capita household "
    "income (PPP-adjusted, 2021 USD). Only genuine countries are kept "
    "(ISO3 code of length 3); pre-aggregated regional rows in WIID are "
    "discarded. Region (World Bank) and income-group codes are decoded "
    "via the Stata value labels exposed by pyreadstat. The resulting "
    "parquet file contains 211 countries × 100 percentiles = 21,100 "
    "rows."
))

story.append(H2("2.2 Population projections"))
story.append(P(
    f"Country population counts for 2022–2100 come from the "
    f"{_link('UN World Population Prospects 2024, Medium Variant', URL_WPP)} "
    "(file 'WPP2024_TotalPopulationBySex.csv.gz'). The app filters "
    "LocTypeName = 'Country/Area' and converts the UN 'thousands' unit "
    "to individuals. Each country-percentile cell is assigned a weight "
    "equal to one one-hundredth of the country population for the year "
    "in question."
))

story.append(H2("2.3 Growth data for 2023–2026"))
story.append(P(
    "Two external sources of country-level real GDP growth are used to "
    "fill the gap between the 2022 baseline and the first user-driven "
    "projection year (2027):"
))
story.append(BUL(
    f"World Bank — {_link('World Development Indicators (WDI)', URL_WDI)}, "
    "indicator NY.GDP.MKTP.KD.ZG (real GDP growth, annual %). Retrieved "
    "via the WDI API (April 2026 vintage)."
))
story.append(BUL(
    f"International Monetary Fund — {_link('World Economic Outlook (WEO)', URL_WEO)} "
    "April 2026 database, indicator NGDP_RPCH (real GDP growth, annual "
    "%). Each WEO row carries a LATEST_ACTUAL_ANNUAL_DATA tag that "
    "distinguishes realised from forecast years for that country."
))
story.append(P(
    f"The World Bank {_link('Global Economic Prospects (GEP)', URL_GEP)} "
    "(January 2026) was also inspected but its country-level tables "
    "were not granular enough for this exercise; it is retained for "
    "reference only."
))

story.append(H2("2.4 IMF benchmark growth 2027–2031"))
story.append(P(
    f"For the 'IMF benchmark' overlay in the Compare-scenarios tab the "
    f"app uses the same {_link('IMF WEO', URL_WEO)} April 2026 NGDP_RPCH "
    "series extended five more years (2027–2031 forecast horizon). When "
    "a country is missing in WEO for a given year, the app imputes the "
    "population-weighted regional mean of WEO growth in that year, "
    "using World Bank regions; if no regional value is available "
    "either, the fallback is zero growth. The file "
    "app/data/imf_benchmark_2027_2031.parquet stores 211 × 5 = 1,055 "
    "rows (941 WEO forecasts + 114 regional-mean imputations across 22 "
    "small or non-WEO countries)."
))

# -------- 3. Projection methodology --------
story.append(H1("3. Projection methodology"))
story.append(P(
    "The projection runs year by year from 2023 to the user-chosen "
    "target year (between 2027 and 2100). Two phases are distinguished:"
))

story.append(H2("3.1 Backfill phase: 2023–2026"))
story.append(P(
    "For each (country c, year t) in 2023–2026 a scalar real-growth "
    "rate is assigned following a strict waterfall:"
))
story.append(BUL("(1) WB WDI realised value for the country-year, if available."))
story.append(BUL(
    "(2) Otherwise, the WEO April 2026 value, tagged 'WEO_actual' when "
    "t is ≤ LATEST_ACTUAL_ANNUAL_DATA for that country, 'WEO_forecast' "
    "otherwise."
))
story.append(BUL(
    "(3) Otherwise, the population-weighted mean of WEO growth in the "
    "same World Bank region for the same year, tagged "
    "'WEO_regional_mean'. This handles the 22 WIID countries for which "
    "neither source offers coverage in at least one year (small "
    "territories, plus AFG, LBN, LKA, SYR, CUB, PRK, XKX, PSE, ERI)."
))
story.append(P(
    "The resulting file 'backfill_2023_2026.parquet' has 844 rows "
    "(211 countries × 4 years) with the following tally of sources over "
    "the four years combined: 391 WDI, 82 WEO_actual, 314 WEO_forecast, "
    "and 57 WEO_regional_mean (covering the 22 countries listed above, "
    "most of them in only one or two years)."
))
story.append(P(
    "During the backfill phase the country-level scalar rate is "
    "broadcast across all 100 percentiles, so within-country shape is "
    "unchanged. This is consistent with the treatment in Gradin (2024) "
    "and Kanbur, Ortiz-Juarez &amp; Sumner (2024): the authors fill the "
    "short gap between the latest observed survey data and their "
    "projection horizon with country-level growth rates from IMF/WB "
    "sources while holding the within-country distribution constant. "
    "See the appendix for details."
))

story.append(H2("3.2 User-driven phase: 2027 onwards"))
story.append(P(
    "From 2027 the user selects one of four growth modes (A, B, C, D). "
    "Modes B and D offer a further 'Aggregation' toggle that switches "
    "the country-aggregate dimension between World Bank regions "
    "(default) and World Bank income groups. In each case the rates "
    "chosen by the user are applied year after year until the target "
    "year. Population blocks used in Modes C and D are fixed: "
    "b40 = percentiles 1–40, m50 = percentiles 41–90, t10 = percentiles "
    "91–100 of each country."
))

# Mode table
mode_rows = [
    [Paragraph("Mode", ST_TH),
     Paragraph("Internal spec.mode", ST_TH),
     Paragraph("Inputs edited by the user", ST_TH)],
    [Paragraph("A", ST_TD), Paragraph("uniform", ST_TD),
     Paragraph("One world-wide growth rate.", ST_TD)],
    [Paragraph("B", ST_TD), Paragraph("by_aggregate", ST_TD),
     Paragraph(
         "One rate per country aggregate (World Bank region or income "
         "group, selected via the Aggregation toggle); optional "
         "China/India override.", ST_TD)],
    [Paragraph("C", ST_TD), Paragraph("by_popgroup", ST_TD),
     Paragraph(
         "One rate each for b40 / m50 / t10 (applied the same in every "
         "country).", ST_TD)],
    [Paragraph("D", ST_TD), Paragraph("by_aggregate_popgroup", ST_TD),
     Paragraph(
         "Three rates (b40 / m50 / t10) per country aggregate (region "
         "or income group, selected via the Aggregation toggle); "
         "optional China/India override.", ST_TD)],
]
mode_tbl = Table(mode_rows, colWidths=[1.5*cm, 4.0*cm, 11.5*cm])
mode_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(Spacer(1, 4))
story.append(mode_tbl)
story.append(Spacer(1, 8))

story.append(P(
    "A clarification that affects the interpretation of Modes C and D: "
    "when the user specifies, for example, '3% for the bottom 40% in "
    "Sub-Saharan Africa', the app applies 3% to the bottom 40% of every "
    "country in Sub-Saharan Africa (not to the pooled bottom 40% of the "
    "region). This has been flagged in the user interface."
))
story.append(P("Two internal consistency checks are reported by the engine:"))
story.append(BUL(
    "Mode C with identical b40 / m50 / t10 rates reproduces Mode A "
    "with the same world-wide rate exactly (maximum absolute difference "
    "across all indices = 0)."
))
story.append(BUL("Mode D with all equal rates reproduces Mode A likewise."))

story.append(H2("3.3 Engine mechanics"))
story.append(P(
    "The engine runs an explicit year-by-year loop on a 211 × 100 "
    "matrix of country-percentile incomes (replacing the earlier "
    "closed-form (1+r)^(t-base) shortcut, which is no longer valid "
    "because backfill rates differ across years). At each iteration t:"
))
story.append(BUL(
    "For t ≤ 2026 the country-year growth rate is looked up in the "
    "backfill parquet and broadcast across the 100 percentiles of the "
    "country."
))
story.append(BUL(
    "For t ≥ 2027 the rate comes from the user's growth spec: a single "
    "scalar for Mode A, a vector of country-aggregate rates for "
    "Modes B, a (100,) vector of percentile rates for Mode C, or a "
    "(country, percentile) matrix for Mode D."
))
story.append(BUL(
    "Incomes are multiplied by (1 + rate), population weights for year "
    "t are attached from the WPP parquet, and the resulting panel for "
    "year t is appended to the output."
))

# -------- 4. Inequality measures --------
story.append(H1("4. Inequality measures"))
story.append(P(
    "For each year in the projection the engine computes the following "
    "inequality indicators on the pooled country-percentile cells, "
    "weighted by the product of country population and 0.01 "
    "(one percentile = 1% of the country's population):"
))
story.append(BUL("Gini coefficient (weighted trapezoidal Lorenz formula)."))
story.append(BUL(
    "Generalised Entropy class: GE(-1), GE(0) — also known as the Mean "
    "Log Deviation, GE(1) — the Theil index, and GE(2). GE(0) and "
    "GE(1) use special-case branches to avoid numerical issues near "
    "zero or very small incomes."
))
story.append(BUL("Atkinson indices A(0.5), A(1), A(2)."))
story.append(BUL(
    "Income-share summaries: bottom 20%, bottom 40%, middle 50%, "
    "top 20%, top 10%."
))
story.append(BUL("Ratios: Palma (top 10% / bottom 40%) and S80/S20."))

story.append(H2("4.1 Within/between decomposition"))
story.append(P(
    "The engine provides an additive within/between decomposition of "
    "GE(0) and GE(1) by country-aggregate group (World Bank region or "
    "income group). The identity GE_total = GE_within + GE_between is "
    "enforced numerically to ~1e-16."
))
story.append(P(
    "The app additionally computes a Shapley-style country-level "
    "between/within decomposition for the Gini and for GE(0), GE(1), "
    "GE(2). Two counterfactual distributions are built on the same "
    "sample: y^b replaces every individual's income with that "
    "individual's country mean (eliminating within-country inequality), "
    "and y^w rescales every country to the global mean (eliminating "
    "between-country inequality). The naive components I(y^b) and "
    "I(y^w) sum to I(y) only for MLD (GE0); the Shapley components "
    "(I(y^b)+I(y)−I(y^w))/2 and (I(y^w)+I(y)−I(y^b))/2 sum to I(y) "
    "exactly for every measure."
))

story.append(H2("4.2 Within-country inequality across countries"))
story.append(P(
    "A complementary view plots the cross-country average of an "
    "indicator computed on each country's 100 percentile incomes (so "
    "this measures inequality inside countries only, ignoring "
    "between-country differences). The average is reported for the "
    "World and for each World Bank region — or, at the user's choice, "
    "each World Bank income group — and can be population-weighted "
    "(default, each country's contribution proportional to its total "
    "population) or unweighted (each country counts equally). "
    "Historical values 1950–2021 come from a precomputed "
    "per-country-year file "
    "(scripts/build_history_within_country.py), concatenated with the "
    "projected panel."
))
story.append(P(
    "Users can toggle on or off which aggregates are drawn — the World "
    "line and every region (or income group) appear by default and can "
    "be added or removed individually. A second chart below the "
    "aggregate view shows the same indicator for individual countries. "
    "The country picker defaults to India, China, Brazil, and the "
    "United States and can be freely edited from the full 211-country "
    "list."
))

story.append(H2("4.3 Scenario comparison"))
story.append(P(
    "Users can save multiple growth specifications under named labels "
    "and overlay their trajectories for a single indicator on the same "
    "chart. Two optional reference benchmarks can be drawn alongside "
    "the saved scenarios:"
))
story.append(BUL(
    "Historical-continuation benchmark (on by default): projects each "
    "country's 100 percentiles forward from 2027 at the compound "
    "annual growth rate observed over the last N years of WIID "
    "(N adjustable between 5 and 20, default 10; base year 2022). "
    "This preserves country-specific growth incidence, so "
    "within-country shape can change."
))
story.append(BUL(
    "IMF benchmark (off by default): uses IMF WEO April 2026 "
    "per-country growth rates for 2027–2031 (see §2.4) held constant "
    "for every year beyond 2031 until the target year. Rates are "
    "broadcast across all 100 percentiles, so within-country "
    "inequality is frozen; only cross-country dynamics and population "
    "reweighting move the global distribution."
))
story.append(P(
    "Both benchmarks leave the 2023–2026 backfill phase untouched and "
    "only replace the user-driven phase. Saved scenarios are kept "
    "per-mode for the duration of the browser session."
))

# -------- 5. Assumptions and caveats --------
story.append(H1("5. Assumptions and caveats"))
story.append(BUL(
    "Within-country distributional shape is held constant throughout "
    "the projection (in both the 2023–2026 backfill and the "
    "user-driven phase). This is distribution-neutral growth at the "
    "country level. In Modes C and D within-country shape changes "
    "only through differential growth across the three population "
    "blocks (b40, m50, t10), but each block is still treated "
    "uniformly inside every country."
))
story.append(BUL(
    "Income units: PPP-adjusted 2021 USD per capita, consistent with "
    "the WIID baseline. Growth rates applied to the distribution are "
    "real (inflation already removed in the sources)."
))
story.append(BUL(
    "Country coverage: 211 countries from WIID. Countries without WDI "
    "or WEO coverage in the backfill period are imputed with the "
    "population-weighted regional mean of WEO growth; see §3.1."
))
story.append(BUL(
    "Population: UN WPP 2024 Medium Variant is the only demographic "
    "scenario considered; fertility and migration uncertainty is not "
    "propagated into the inequality estimates."
))
story.append(BUL(
    "No climate feedback, redistribution channel, or behavioural "
    "response is modelled. The app is a descriptive tool for "
    "distribution-neutral 'what-if' scenarios, not a structural model. "
    "Papers that embed redistribution and climate damage "
    "(Bothe et al. 2025) are discussed in the appendix."
))

# -------- 6. Reproducibility --------
story.append(H1("6. Reproducibility and files"))
story.append(P("All code and data live in a single project folder:"))
story.append(BUL(
    "app/app.py — Streamlit UI (six tabs: inequality measures; group "
    "shares &amp; Palma; within/between decomposition; within-country "
    "distributions; summary table; compare scenarios — and four growth "
    "modes A, B, C, D, with a region/income-group Aggregation toggle "
    "in Modes B and D)."
))
story.append(BUL("app/engine.py — projection loop and inequality measures."))
story.append(BUL(
    "app/data/baseline_2022.parquet, regions.parquet, "
    "wpp_population.parquet, backfill_2023_2026.parquet — the four "
    "inputs read at app startup for the projection."
))
story.append(BUL(
    "app/data/imf_benchmark_2027_2031.parquet — per-country IMF WEO "
    "growth rates 2027–2031 used by the 'IMF benchmark' overlay (see "
    "§2.4 and §4.3)."
))
story.append(BUL(
    "app/data/history_indices.parquet, history_between_within.parquet — "
    "precomputed global and country-level between/within historical "
    "series (1950–2021), used to show historical context on the charts."
))
story.append(BUL(
    "app/data/history_country_indices.parquet — precomputed "
    "per-country-year within-country inequality indices (1950–2021), "
    "used by the 'Within-country distributions' tab."
))
story.append(BUL(
    "app/data/historical_2000_2022.parquet — per-country-percentile "
    "WIID panel (2000–2022), used to compute the 'historical "
    "continuation' benchmark for the scenario-comparison tab."
))
story.append(BUL(
    "scripts/build_baseline.py — rebuilds the 2022 baseline from the "
    "raw Stata file."
))
story.append(BUL(
    "scripts/build_backfill.py — rebuilds the 2023–2026 backfill from "
    "WDI and WEO inputs; re-run when new vintages are released."
))
story.append(BUL(
    "scripts/build_imf_benchmark.py — rebuilds the 2027–2031 IMF WEO "
    "per-country benchmark growth rates (same waterfall as backfill)."
))
story.append(BUL(
    "scripts/build_history.py — rebuilds the global and country-level "
    "between/within historical series (1950–2021)."
))
story.append(BUL(
    "scripts/build_history_within_country.py — rebuilds the historical "
    "per-country-year inequality indices (1950–2021)."
))
story.append(BUL(
    "scripts/build_historical.py — rebuilds the historical WIID "
    "per-country-percentile panel (2000–2022) used by the benchmark."
))
story.append(BUL("scripts/fetch_wpp.py — re-downloads UN WPP 2024 populations."))
story.append(P(
    "A deployed version of the app is available on Streamlit Community "
    "Cloud (see README for the URL)."
))

# =====================================================================
# APPENDIX
# =====================================================================
story.append(PageBreak())
story.append(H1("Appendix A. Summary of three recent projection papers"))
story.append(P(
    "The approach adopted here draws on three recent papers that also "
    "project the global income distribution forward from a recent "
    "baseline. The short summaries below are reproduced from our "
    "internal review; a detailed side-by-side comparison table follows "
    "in Appendix B."
))

story.append(H2("A.1 Gradin (2024)"))
story.append(P(
    "<i>'Revisiting the trends in global inequality' — "
    "World Development.</i>"
))
story.append(P(
    "Gradin constructs a new WIID Companion dataset spanning 1950–2020 "
    "that deliberately integrates multiple sources (WDI, Maddison, "
    "PWT, WID.world) to maximise temporal and cross-sectional "
    "consistency. A particularly valuable feature is a hybrid dataset "
    "that replaces the top 1% income share in each country with "
    "estimates from WID.world (1980–2020) while keeping average "
    "incomes unchanged, allowing him to isolate the effect of survey "
    "underestimation of top incomes. Projections to 2028 assume "
    "constant within-country distributions and use IMF WEO October "
    "2023 growth forecasts, so they are highly data-dependent on "
    "IMF's pandemic-recovery assumptions. The paper reports "
    "extensively in absolute inequality terms (absolute Gini, standard "
    "deviation in constant 2017 PPP USD), revealing that absolute "
    "inequality has unambiguously increased across all decades since "
    "1950. A Shapley decomposition quantifies country contributions "
    "and shows that China's role in driving between-country inequality "
    "has shrunk to near-zero by 2020. The trade-off is that "
    "projections are limited to 2028 and remain silent on medium-term "
    "redistribution scenarios."
))

story.append(H2("A.2 Kanbur, Ortiz-Juarez &amp; Sumner (2024)"))
story.append(P(
    "<i>'Is the era of declining global income inequality over?' — "
    "Structural Change and Economic Dynamics.</i>"
))
story.append(P(
    "The paper develops a parsimonious three-country theoretical model "
    "(Africa / China / US) that elegantly formalises the intuition "
    "that rapidly-growing middle-income countries will eventually "
    "cause global inequality to rise as they converge to rich-country "
    "levels. Empirically, the authors reconstruct the full "
    "distribution from World Bank PovcalNet using an ingenious "
    "10-cent binning algorithm (6,230+ country-year distributions), "
    "assigning each $0.10 bin its midpoint value rather than assuming "
    "uniform income within deciles — more flexible than Darvas (2019)'s "
    "'identical quantile income' method. They incorporate the COVID-19 "
    "shock via WB nowcasting for 2020 and project forward using HFCE "
    "per capita growth (not GDP), which is theoretically closer to "
    "household income. Two stylised post-pandemic scenarios (weak and "
    "strong recovery) make uncertainty explicit. The methodology "
    "assumes distribution-neutral growth, so within-country shapes are "
    "frozen. Projections extend to 2040, offering a medium-term "
    "horizon between Gradin and Bothe et al."
))

story.append(H2("A.3 Bothe, Chancel, Gethin &amp; Mohren (2025)"))
story.append(P(
    "<i>'Global Income Inequality by 2050: Convergence, "
    "Redistribution, and Climate Change' — World Inequality Lab "
    "Working Paper 2025/09.</i>"
))
story.append(P(
    "This is the most comprehensive of the three papers. It projects "
    "to 2050 using the SSP2 ('Middle of the Road') scenario, enabling "
    "alignment with climate and development literatures. It leverages "
    "the most extensive pre-tax and post-tax inequality dataset "
    "(WID.world, 1980–2023, 146 countries, 127 percentiles). The BAU "
    "scenario is sophisticated: each percentile captures growth "
    "according to its observed 2000–2023 growth incidence, allowing "
    "for realistic within-country inequality dynamics rather than "
    "pure distribution-neutrality. Four policy scenarios progressively "
    "benchmark countries to redistribution leaders in their income "
    "group, showing that convergence to the most progressive "
    "redistributors within LIC/LMIC/UMIC/HIC would double the "
    "bottom-50% income share by 2050. The paper incorporates RCP8.5 "
    "climate damages with an income elasticity parameter (η = 0.64, "
    "from Gilli et al. 2024), modelling climate change as regressive — "
    "damages concentrated on lower-income populations. Results show "
    "climate could fully offset 40 years of inequality improvements. "
    "The trade-off is added complexity: pre-tax/post-tax distinction, "
    "multiple scenario branches, and climate-damage distributions "
    "require careful interpretation."
))

# -------- Appendix B: comparison table --------
story.append(PageBreak())
story.append(H1("Appendix B. Side-by-side comparison"))
story.append(P(
    "Detailed methodology comparison across the three papers. Table "
    "and short commentary reproduced from the internal review "
    "PROJECTION_METHODS_REVIEW.md."
))

DIMENSIONS = [
    ("Base year (distribution)",
     "1950–2020 (WIID Companion) with WID.world top-1% corrections from 1980.",
     "1981–2019; reconstructed from WB PovcalNet via the 'Ten Cents Database' 10-cent-bin algorithm.",
     "1980–2023; WID.world covering 146 countries (surveys + tax records + fiscal incidence)."),
    ("Target year", "2028.", "2040.", "2050."),
    ("Short-gap backfill",
     "WDI (2021–22) + IMF WEO April 2023 growth through 2028; within-country distributions unchanged.",
     "WB WDI growth rates for 2020 (COVID shock via WB nowcasting); HFCE or GDP per capita growth to 2040; within-country distributions unchanged.",
     "None; projections start in 2024 from the 2023 baseline."),
    ("Growth data source",
     "IMF WEO October 2023; WDI, Maddison, PWT for sensitivity; per capita income in 2017 PPP USD.",
     "WDI (July 2022); WB nowcasting for 2020; UN WPP 2022 medium variant for population; per capita HFCE or income in 2011 PPP USD.",
     "SSP2 ('Middle of the Road') GDP/NI growth from Crespo Cuaresma (2017); IIASA Release 3.1 population (KC et al. 2024)."),
    ("Distributional assumption",
     "Constant within-country distributions (distribution-neutral growth); sensitivity checks on this assumption.",
     "Distribution-neutral growth: full pass-through of country growth to every income/consumption bin.",
     "BAU: each percentile keeps its 2000–2023 growth incidence; post-tax redistribution follows the same trend. Policy scenarios: convergence to redistribution leaders within LIC/LMIC/UMIC/HIC."),
    ("Population projections",
     "UN WPP 2019 through 2028.",
     "UN WPP 2022 medium variant through 2040.",
     "IIASA SSP2 (KC et al. 2024, Release 3.1) through 2050; 127 generalised percentiles."),
    ("Scenarios",
     "Single main scenario; sensitivity to alternative GDP measures and to WID.world top-1% corrections.",
     "Two post-pandemic recovery scenarios: weaker recovery (half of 1990–2019 growth, consistent with IMF 2023) and stronger recovery (full 1990–2019 growth).",
     "BAU + four policy scenarios (pre-tax and post-tax redistribution leaders) + climate scenarios (low- and high-impact RCP8.5 with η=0.64 to 1.0)."),
    ("Inequality measures",
     "Gini, MLD (GE0), Theil (GE1), GE(-1), GE(2), Palma; also absolute Gini and standard deviation; income shares by percentile.",
     "MLD as the primary measure; decomposed into within- and between-country components; global distributions visualised as density plots (1981, 2019).",
     "Gini; Theil with between/within decomposition; income shares (top 1%, top 10%, bottom 50%, middle 40%); both pre-tax and post-tax inequality."),
    ("Unit",
     "Per capita household income (net of taxes/transfers), PPP-adjusted 2017 USD.",
     "Per capita household income or consumption, PPP-adjusted 2011 USD; ranked at individual level.",
     "Per capita (and per-adult) pre-tax and post-tax income from WID.world; PPP-adjusted, dataset-harmonised."),
    ("Unique features",
     "Hybrid WIID Companion dataset; Shapley decomposition of country contributions; robustness to alternative GDP measures and top-income corrections; absolute vs relative inequality throughout; short horizon (2028).",
     "Theoretical 3-country model formalising the 'turning point' in global inequality; 10-cent-bin distributions (6,230 country-years); two stylised recovery scenarios; MLD focus suits between-country decomposition.",
     "Joint pre-tax / post-tax projection through 2050; SSP2 alignment with climate literatures; dedicated RCP8.5 climate-damage modelling with income-elastic damages; convergence-to-leader policy scenarios; largest sample (146 countries) with tax-record data."),
]

header = [Paragraph(h, ST_TH) for h in [
    "Dimension", "Gradin (2024)",
    "Kanbur et al. (2024)", "Bothe et al. (2025)"]]
body = []
for row in DIMENSIONS:
    body.append([Paragraph(row[0], ST_TDB)] + [Paragraph(c, ST_TD) for c in row[1:]])
cmp_tbl = Table([header] + body,
                colWidths=[3.2*cm, 4.5*cm, 4.5*cm, 4.5*cm],
                repeatRows=1)
cmp_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.white, colors.HexColor("#F2F2F2")]),
]))
story.append(cmp_tbl)

# -------- Appendix C: key divergences --------
story.append(PageBreak())
story.append(H1("Appendix C. Key divergences across papers and implications"))

story.append(BUL(
    "Within-country distribution evolution. Gradin (2024) and Kanbur "
    "et al. (2024) assume constant shapes; Bothe et al. (2025) use "
    "historical growth incidence (realistic but requires 2000–2023 "
    "data reliability). This is material: if inequality within "
    "countries continues rising (as in China and India), BAU "
    "inequality will worsen despite between-country convergence."
))
story.append(BUL(
    "Growth assumptions. Gradin relies on IMF October 2023 forecasts; "
    "Bothe et al. use SSP2 long-run scenarios (smoother, consistent "
    "with IPCC frameworks); Kanbur et al. test two extremes. Results "
    "are sensitive to these — IMF typically forecasts slower recovery "
    "than historical averages."
))
story.append(BUL(
    "Climate integration. Only Bothe et al. include climate. Their "
    "high-impact scenario reduces the bottom-50% share to pre-1980 "
    "levels by 2050, offsetting 40 years of convergence gains. "
    "Crucial for long-term 2050 projections but absent from Gradin "
    "and Kanbur et al."
))
story.append(BUL(
    "Inequality measures. Gradin emphasises absolute inequality "
    "(often ignored, but shows an unambiguous rise); Kanbur et al. "
    "focus on MLD decomposition; Bothe et al. split pre-tax vs "
    "post-tax. Gini is reported by all three but means different "
    "things depending on what else is measured."
))
story.append(BUL(
    "Policy scenarios. Only Bothe et al. model redistribution "
    "changes; Gradin and Kanbur et al. are descriptive of baseline "
    "trajectories."
))
story.append(Spacer(1, 6))
story.append(P(
    "For practitioners, Bothe et al. (2025) is the most appropriate "
    "reference for policy-relevant long-run projections; Kanbur et "
    "al. (2024) for understanding transition dynamics and structural "
    "forces; Gradin (2024) for establishing a robust historical "
    "consensus on what actually happened."
))


# ---------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
    title="Technical note - Global inequality projections",
    author="Carlos Gradin",
)
doc.build(story)
print(f"Wrote: {OUT}")
