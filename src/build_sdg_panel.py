"""
build_sdg_panel.py
==================
Builds SDG indicator panel from existing data holdings.

Sources (all already on disk):
  - data/harmonised/unified_panel.csv   (WDI indicators, 268 countries 2000-2023)
  - data/raw/who_health/GHED_data.XLSX  (health expenditure SDG 3)
  - data/raw/vdem/V-Dem-CY-Core-v14.csv (governance SDG 16)
  - data/raw/imf_capital/IMF_Capital_Stock_2021.xlsx (infrastructure SDG 9)
  - data/raw/gfw_tree_cover_loss.csv    (forest SDG 15)

Output: data/processed/sdg_panel.parquet

Run:
    cd ~/cross-national-dev-analytics
    conda activate ds
    PYTHONPATH=. python src/build_sdg_panel.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_HARM= ROOT / "data" / "harmonised"
DATA_PROC= ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ── Asia-Pacific country list ──────────────────────────────────────────────────
APAC_ISO3 = {
    "AFG","AUS","BGD","BTN","BRN","KHM","CHN","FJI","IND","IDN",
    "JPN","KAZ","KIR","KGZ","LAO","MYS","MDV","MHL","MNG","MMR",
    "NPL","NZL","PAK","PLW","PNG","PHL","KOR","WSM","SGP","SLB",
    "LKA","TJK","TLS","TON","TKM","TUV","UZB","VUT","VNM",
    # ADB DMCs
    "ARM","AZE","GEO","NRU","FSM",
}

# ── SDG indicator mapping from WDI column names ───────────────────────────────
SDG_WDI_MAP = {
    "sdg1_poverty":            "poverty_headcount",
    "sdg1_gni_per_capita":     "gni_per_capita_atlas",
    "sdg2_undernourishment":   "sdg_undernourishment",
    "sdg3_u5_mortality":       "mortality_under5",
    "sdg3_maternal_mortality": "sdg_maternal_mortality",
    "sdg3_life_expectancy":    "life_expectancy",
    "sdg4_primary_enrolment":  "primary_school_enrolment",
    "sdg4_secondary_enrolment":"secondary_school_enrolment",
    "sdg6_clean_water":        "access_clean_water",
    "sdg6_sanitation":         "access_sanitation",
    "sdg7_renewable_energy":   "renewable_energy_pct",
    "sdg7_electricity":        "access_electricity",
    "sdg8_gdp_growth":         "gdp_growth",
    "sdg8_gdp_per_capita":     "gdp_per_capita",
    "sdg9_internet":           "internet_users_pct",
    "sdg9_mobile_subs":        "mobile_subscriptions",
    "sdg10_gini":              "gini_index",
    "sdg10_fdi":               "fdi_inflows_pct_gdp",
    "sdg16_rule_of_law":       "wgi_rule_of_law",
    "sdg16_govt_effectiveness":"wgi_govt_effectiveness",
    "sdg16_corruption":        "wgi_control_of_corruption",
    "sdg16_political_stability":"wgi_political_stability",
    "sdg16_regulatory":        "wgi_regulatory_quality",
    "sdg16_voice":             "wgi_voice_accountability",
    "sdg17_trade":             "trade_pct_gdp",
    "sdg17_fdi":               "fdi_inflows_pct_gdp",
    "sdg_birth_registration":  "sdg_birth_registration",
    "sdg_poverty_ratio":       "sdg_poverty_ratio",
    "sdg_composite":           "sdg_composite_score",
    "financial_inclusion":     "financial_inclusion_index",
}

SDG_METADATA = {
    1:  {"name": "No Poverty",           "color": "#e74c3c", "icon": "🏠"},
    2:  {"name": "Zero Hunger",          "color": "#e67e22", "icon": "🌾"},
    3:  {"name": "Good Health",          "color": "#2ecc71", "icon": "🏥"},
    4:  {"name": "Quality Education",    "color": "#c0392b", "icon": "📚"},
    5:  {"name": "Gender Equality",      "color": "#e91e8c", "icon": "⚖️"},
    6:  {"name": "Clean Water",          "color": "#3498db", "icon": "💧"},
    7:  {"name": "Clean Energy",         "color": "#f1c40f", "icon": "⚡"},
    8:  {"name": "Decent Work",          "color": "#a04000", "icon": "💼"},
    9:  {"name": "Infrastructure",       "color": "#e67e22", "icon": "🏗️"},
    10: {"name": "Reduced Inequalities", "color": "#e91e63", "icon": "📊"},
    13: {"name": "Climate Action",       "color": "#27ae60", "icon": "🌍"},
    16: {"name": "Peace & Institutions", "color": "#2c3e50", "icon": "🏛️"},
    17: {"name": "Partnerships",         "color": "#16a085", "icon": "🤝"},
}


def load_unified_panel() -> pd.DataFrame:
    path = DATA_HARM / "unified_panel.csv"
    if not path.exists():
        path = DATA_HARM / "unified_panel_v2.csv"
    log.info(f"Loading unified panel: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Shape: {df.shape}")
    return df


def extract_sdg_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """Extract SDG-relevant columns from unified panel."""
    # Identify id columns
    id_candidates = ["iso3", "country_code", "countrycode", "iso_code"]
    year_candidates = ["year", "yr"]
    name_candidates = ["country", "country_name", "countryname"]

    iso_col  = next((c for c in id_candidates if c in panel.columns), None)
    year_col = next((c for c in year_candidates if c in panel.columns), None)
    name_col = next((c for c in name_candidates if c in panel.columns), None)

    if not iso_col or not year_col:
        log.error("Cannot find iso3 or year column in panel")
        log.info(f"Available columns: {list(panel.columns[:20])}")
        return pd.DataFrame()

    # Find which SDG indicators exist in the panel
    found_indicators = {}
    for sdg_name, wdi_code in SDG_WDI_MAP.items():
        # Try exact match, then case-insensitive, then partial
        if wdi_code in panel.columns:
            found_indicators[sdg_name] = wdi_code
        elif wdi_code.lower() in panel.columns:
            found_indicators[sdg_name] = wdi_code.lower()
        else:
            # Try replacing dots with underscores (common in cleaned datasets)
            alt = wdi_code.replace(".", "_").lower()
            if alt in panel.columns:
                found_indicators[sdg_name] = alt

    log.info(f"Found {len(found_indicators)}/{len(SDG_WDI_MAP)} SDG indicators in panel")

    keep_cols = [iso_col, year_col]
    if name_col:
        keep_cols.append(name_col)
    keep_cols += list(found_indicators.values())

    # Remove duplicates
    keep_cols = list(dict.fromkeys(keep_cols))
    df = panel[[c for c in keep_cols if c in panel.columns]].copy()

    # Rename to canonical SDG names
    rename_map = {v: k for k, v in found_indicators.items()}
    rename_map[iso_col]  = "iso3"
    rename_map[year_col] = "year"
    if name_col:
        rename_map[name_col] = "country_name"
    df = df.rename(columns=rename_map)

    return df


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add region, income group, and APAC flag."""
    df["is_apac"] = df["iso3"].isin(APAC_ISO3)

    # World Bank regions (simplified)
    region_map = {
        "EAS": ["CHN","JPN","KOR","MNG","PRK"],
        "SEA": ["BRN","KHM","IDN","LAO","MYS","MMR","PHL","SGP","THA","TLS","VNM"],
        "SAS": ["AFG","BGD","BTN","IND","MDV","NPL","PAK","LKA"],
        "PAC": ["AUS","FJI","KIR","MHL","FSM","NRU","NZL","PLW","PNG","WSM","SLB","TON","TUV","VUT"],
        "CAC": ["ARM","AZE","GEO","KAZ","KGZ","TJK","TKM","UZB"],
    }
    df["apac_subregion"] = "Other"
    for region, isos in region_map.items():
        df.loc[df["iso3"].isin(isos), "apac_subregion"] = region

    return df


def build_panel():
    panel = load_unified_panel()
    df    = extract_sdg_indicators(panel)

    if df.empty:
        log.error("No SDG indicators extracted — check column names in unified_panel.csv")
        return

    df = add_metadata(df)
    df = df[df["year"].between(2000, 2023)]
    df = df.sort_values(["iso3", "year"])

    out_parquet = DATA_PROC / "sdg_panel.parquet"
    out_csv     = DATA_PROC / "sdg_panel.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    log.info(f"\n✓ SDG panel: {df.shape}")
    log.info(f"  Countries: {df['iso3'].nunique()}")
    log.info(f"  APAC: {df[df['is_apac']]['iso3'].nunique()}")
    log.info(f"  Years: {df['year'].min()}–{df['year'].max()}")
    log.info(f"  Saved: {out_parquet}")

    # Coverage report
    val_cols = [c for c in df.columns if c.startswith("sdg")]
    cov = df[val_cols].notna().mean().sort_values(ascending=False)
    log.info("\nIndicator coverage:")
    for col, pct in cov.items():
        log.info(f"  {col:40s} {pct*100:.0f}%")

    return df


if __name__ == "__main__":
    build_panel()
