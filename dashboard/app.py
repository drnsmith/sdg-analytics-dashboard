"""
dashboard/app.py — SDG Progress Analytics Dashboard
=====================================================
Tracks Sustainable Development Goal indicators across Asia-Pacific and globally.
Built for ADB DIG (AI & Data Analytics) portfolio demonstration.

Run:
    cd ~/sdg-analytics-dashboard
    conda activate ds
    pip install dash dash-bootstrap-components statsmodels
    PYTHONPATH=. python dashboard/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc

ROOT      = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed"

# ── DESIGN ────────────────────────────────────────────────────────────────────
DARK_BG   = "#060d1f"
CARD_BG   = "#0d1730"
BORDER    = "#1a2744"
ACCENT    = "#00c9a7"
ACCENT2   = "#4361ee"
ACCENT3   = "#f72585"
WARN      = "#ffd166"
TEXT      = "#e2e8f0"
TEXT_DIM  = "#64748b"
FONT_BODY = "'IBM Plex Sans', 'Inter', sans-serif"
FONT_MONO = "'IBM Plex Mono', monospace"

SDG_COLORS = {
    1: "#e5243b", 2: "#dda63a", 3: "#4c9f38", 4: "#c5192d",
    5: "#ff3a21", 6: "#26bde2", 7: "#fcc30b", 8: "#a21942",
    9: "#fd6925", 10: "#dd1367", 13: "#3f7e44", 16: "#00689d",
    17: "#19486a",
}

SDG_NAMES = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health",
    4: "Quality Education", 5: "Gender Equality", 6: "Clean Water",
    7: "Clean Energy", 8: "Decent Work", 9: "Infrastructure",
    10: "Reduced Inequalities", 13: "Climate Action",
    16: "Peace & Institutions", 17: "Partnerships",
}

INDICATOR_LABELS = {
    # Actual panel column names → proper labels
    "sdg1_poverty":             "Poverty headcount (<$1.90/day) %",
    "sdg1_gni_per_capita":      "GNI per capita (USD)",
    "sdg2_undernourishment":    "Undernourishment %",
    "sdg3_u5_mortality":        "Under-5 mortality (per 1,000)",
    "sdg3_maternal_mortality":  "Maternal mortality (per 100,000)",
    "sdg3_life_expectancy":     "Life expectancy (years)",
    "sdg4_primary_enrolment":   "Primary school enrolment %",
    "sdg4_secondary_enrolment": "Secondary school enrolment %",
    "sdg6_clean_water":         "Access to clean water %",
    "sdg6_sanitation":          "Access to sanitation %",
    "sdg7_renewable_energy":    "Renewable energy share %",
    "sdg7_electricity":         "Access to electricity %",
    "sdg8_gdp_growth":          "GDP growth rate %",
    "sdg8_gdp_per_capita":      "GDP per capita (USD)",
    "sdg9_internet":            "Internet users %",
    "sdg9_mobile_subs":         "Mobile subscriptions per 100",
    "sdg10_gini":               "Gini coefficient (inequality)",
    "sdg10_fdi":                "FDI inflows (% GDP)",
    "sdg16_rule_of_law":        "Rule of law (WGI)",
    "sdg16_govt_effectiveness": "Government effectiveness (WGI)",
    "sdg16_corruption":         "Control of corruption (WGI)",
    "sdg16_political_stability":"Political stability (WGI)",
    "sdg16_regulatory":         "Regulatory quality (WGI)",
    "sdg16_voice":              "Voice & accountability (WGI)",
    "sdg17_trade":              "Trade (% GDP)",
    "sdg17_fdi":                "FDI inflows (% GDP)",
    "sdg_birth_registration":   "Birth registration %",
    "sdg_poverty_ratio":        "Poverty ratio %",
    "sdg_composite":            "SDG composite score",
    "financial_inclusion":      "Financial inclusion index",
    # Original labels
    "sdg1_poverty_190":        "Poverty headcount (<$1.90/day) %",
    "sdg1_poverty_320":        "Poverty headcount (<$3.20/day) %",
    "sdg1_gni_per_capita":     "GNI per capita (USD)",
    "sdg2_undernourishment":   "Undernourishment %",
    "sdg3_u5_mortality":       "Under-5 mortality (per 1,000)",
    "sdg3_maternal_mortality": "Maternal mortality (per 100,000)",
    "sdg3_life_expectancy":    "Life expectancy (years)",
    "sdg3_health_expenditure": "Health expenditure (% GDP)",
    "sdg4_primary_completion": "Primary school completion %",
    "sdg4_secondary_enrol":    "Secondary enrolment %",
    "sdg4_literacy_rate":      "Adult literacy rate %",
    "sdg5_female_labour":      "Female labour participation %",
    "sdg5_women_parliament":   "Women in parliament %",
    "sdg6_water_access":       "Basic water access %",
    "sdg6_sanitation":         "Basic sanitation access %",
    "sdg7_electricity_access": "Electricity access %",
    "sdg7_renewable_energy":   "Renewable energy share %",
    "sdg8_gdp_growth":         "GDP growth rate %",
    "sdg8_gdp_per_capita":     "GDP per capita (constant USD)",
    "sdg8_unemployment":       "Unemployment rate %",
    "sdg9_internet_users":     "Internet users %",
    "sdg9_mobile_subs":        "Mobile subscriptions per 100",
    "sdg10_gini":              "Gini coefficient",
    "sdg10_remittances":       "Remittances received (% GDP)",
    "sdg13_co2_per_capita":    "CO₂ emissions per capita (tonnes)",
    "sdg16_rule_of_law":       "Rule of law (WGI estimate)",
    "sdg16_control_corruption":"Control of corruption (WGI)",
    "sdg16_govt_effectiveness":"Government effectiveness (WGI)",
    "sdg17_oda_received":      "ODA received (% GNI)",
    "sdg17_tax_revenue":       "Tax revenue (% GDP)",
}

APAC_ISO3 = {
    "AFG","AUS","BGD","BTN","BRN","KHM","CHN","FJI","IND","IDN",
    "JPN","KAZ","KIR","KGZ","LAO","MYS","MDV","MHL","MNG","MMR",
    "NPL","NZL","PAK","PLW","PNG","PHL","KOR","WSM","SGP","SLB",
    "LKA","TJK","TLS","TON","TKM","TUV","UZB","VUT","VNM",
    "ARM","AZE","GEO","NRU","FSM",
}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_BODY, color=TEXT),
    margin=dict(l=20, r=20, t=45, b=20),
)
AXIS = dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_data():
    panel_path = DATA_PROC / "sdg_panel.parquet"
    fcast_path = DATA_PROC / "sdg_forecasts.parquet"

    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
    else:
        panel = make_demo_panel()

    forecasts = pd.read_parquet(fcast_path) if fcast_path.exists() else pd.DataFrame()
    return panel, forecasts


def make_demo_panel():
    """Generate demo data if real panel not built yet."""
    np.random.seed(42)
    countries = {
        "CHN": "China", "IND": "India", "IDN": "Indonesia",
        "PHL": "Philippines", "VNM": "Viet Nam", "BGD": "Bangladesh",
        "PAK": "Pakistan", "MMR": "Myanmar", "KHM": "Cambodia",
        "LAO": "Lao PDR", "NPL": "Nepal", "LKA": "Sri Lanka",
        "MNG": "Mongolia", "KGZ": "Kyrgyz Republic", "TJK": "Tajikistan",
        "AUS": "Australia", "NZL": "New Zealand", "JPN": "Japan",
        "KOR": "Korea, Rep.", "MYS": "Malaysia", "THA": "Thailand",
        "SGP": "Singapore", "BRN": "Brunei Darussalam",
        "USA": "United States", "GBR": "United Kingdom",
        "DEU": "Germany", "BRA": "Brazil", "NGA": "Nigeria",
        "ETH": "Ethiopia", "KEN": "Kenya",
    }
    rows = []
    for iso3, name in countries.items():
        is_apac = iso3 in APAC_ISO3
        is_dev  = iso3 in {"CHN","IND","IDN","PHL","VNM","BGD","PAK",
                           "MMR","KHM","LAO","NPL","LKA","MNG","KGZ","TJK",
                           "NGA","ETH","KEN","BRA"}
        base_gdp = np.random.uniform(500, 50000)

        for yr in range(2000, 2024):
            t = yr - 2000
            rows.append({
                "iso3": iso3, "year": yr, "country_name": name,
                "is_apac": is_apac,
                "sdg1_poverty_190":        max(0, 40 - t*1.2 + np.random.normal(0,2)) if is_dev else max(0, 3 - t*0.1 + np.random.normal(0,0.3)),
                "sdg3_life_expectancy":    55 + t*0.4 + np.random.normal(0,0.5) if is_dev else 75 + t*0.1 + np.random.normal(0,0.3),
                "sdg3_health_expenditure": np.random.uniform(2, 10),
                "sdg4_primary_completion": min(100, 60 + t*1.5 + np.random.normal(0,2)) if is_dev else min(100, 95 + t*0.1),
                "sdg7_electricity_access": min(100, 50 + t*2 + np.random.normal(0,2)) if is_dev else 99 + np.random.normal(0,0.2),
                "sdg7_renewable_energy":   max(0, 20 + t*0.5 + np.random.normal(0,2)),
                "sdg8_gdp_per_capita":     base_gdp * (1.03 ** t) + np.random.normal(0, base_gdp*0.02),
                "sdg8_gdp_growth":         np.random.normal(4 if is_dev else 2, 2),
                "sdg8_unemployment":       np.random.uniform(2, 15),
                "sdg9_internet_users":     min(100, 2 + t*3 + np.random.normal(0,3)) if is_dev else min(100, 40 + t*2),
                "sdg10_gini":              np.random.uniform(25, 55),
                "sdg13_co2_per_capita":    np.random.uniform(0.2, 15),
                "sdg16_rule_of_law":       np.random.normal(-0.5 if is_dev else 0.5, 0.3),
                "sdg16_control_corruption":np.random.normal(-0.5 if is_dev else 0.5, 0.3),
            })
    return pd.DataFrame(rows)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
PANEL, FORECASTS = load_data()
SDG_INDICATORS   = [c for c in PANEL.columns if c.startswith("sdg")]
ALL_COUNTRIES    = sorted(PANEL["iso3"].unique())
APAC_COUNTRIES   = sorted(PANEL[PANEL["is_apac"]==True]["iso3"].unique())

# Build country name lookup
try:
    import pycountry
    NAME_LOOKUP = {c.alpha_3: c.name for c in pycountry.countries}
    panel_names = PANEL[["iso3","country_name"]].dropna().drop_duplicates().set_index("iso3")["country_name"].to_dict()
    NAME_LOOKUP.update(panel_names)
except ImportError:
    NAME_LOOKUP = {iso: iso for iso in ALL_COUNTRIES}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def sdg_number(indicator: str) -> int:
    try: return int(indicator.split("_")[0].replace("sdg",""))
    except: return 0

def ind_label(indicator: str) -> str:
    if indicator in INDICATOR_LABELS:
        return INDICATOR_LABELS[indicator]
    # Clean up auto-generated names
    name = indicator
    # Remove sdgN_ prefix
    import re
    name = re.sub(r"^sdg\d+_", "", name)
    name = re.sub(r"^sdg_", "", name)
    # Replace underscores, title case
    name = name.replace("_", " ").title()
    # Fix common abbreviations
    name = name.replace("Gdp", "GDP").replace("Gni", "GNI")
    name = name.replace("Wgi", "WGI").replace("Pct", "%")
    name = name.replace("Per Capita", "per capita")
    return name

def country_options(apac_only=True):
    isos = APAC_COUNTRIES if apac_only else ALL_COUNTRIES
    return [{"label": f"{NAME_LOOKUP.get(iso, iso)} ({iso})", "value": iso}
            for iso in isos]

def indicator_options():
    opts = []
    for sdg_num in sorted(SDG_NAMES.keys()):
        inds = [i for i in SDG_INDICATORS if sdg_number(i) == sdg_num]
        for ind in inds:
            opts.append({
                "label": ind_label(ind),
                "value": ind,
            })
    return opts

# ── FIGURE BUILDERS ───────────────────────────────────────────────────────────
def fig_trend(iso3_list, indicator, with_forecast=True):
    if not iso3_list or not indicator:
        return go.Figure()

    fig   = go.Figure()
    # Assign colors by iso3 for consistency across tabs
    iso3_color_map = {
        "CHN": "#00c9a7", "IND": "#4361ee", "USA": "#f72585",
        "GBR": "#ffd166", "BRA": "#ff9f1c", "RUS": "#8338ec",
        "IDN": "#2ec4b6", "PHL": "#e71d36", "VNM": "#fb5607",
        "BGD": "#06d6a0", "PAK": "#118ab2", "MMR": "#ef476f",
        "KHM": "#ffc43d", "LAO": "#1b998b", "NPL": "#e9c46a",
        "LKA": "#f4a261", "MNG": "#a8dadc", "JPN": "#457b9d",
        "KOR": "#e63946", "AUS": "#264653",
    }
    color_list = [
        "#00c9a7", "#4361ee", "#f72585", "#ffd166",
        "#ff9f1c", "#8338ec", "#2ec4b6", "#e71d36",
        "#fb5607", "#06d6a0", "#118ab2", "#ef476f",
        "#ffc43d", "#1b998b", "#e9c46a", "#f4a261",
        "#264653", "#a8dadc", "#457b9d", "#e63946",
    ]

    for i, iso3 in enumerate(iso3_list[:8]):
        color = iso3_color_map.get(iso3, color_list[i % len(color_list)])
        cname = NAME_LOOKUP.get(iso3, iso3)
        cdf   = PANEL[PANEL["iso3"]==iso3].sort_values("year")
        cdf   = cdf[cdf[indicator].notna()] if indicator in cdf.columns else pd.DataFrame()
        if cdf.empty:
            continue

        fig.add_trace(go.Scatter(
            x=cdf["year"], y=cdf[indicator],
            mode="lines+markers", name=cname,
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{cname}</b><br>Year: %{{x}}<br>{ind_label(indicator)}: %{{y:.2f}}<extra></extra>",
        ))

        # Forecast
        if with_forecast and not FORECASTS.empty:
            fc = FORECASTS[(FORECASTS["iso3"]==iso3) & (FORECASTS["indicator"]==indicator)]
            fc = fc[fc["is_forecast"]==True]
            if not fc.empty:
                fig.add_trace(go.Scatter(
                    x=fc["year"], y=fc["value"],
                    mode="lines", name=f"{cname} (forecast)",
                    line=dict(color=color, width=1.5, dash="dot"),
                    showlegend=False,
                    hovertemplate=f"<b>{cname} forecast</b><br>Year: %{{x}}<br>{ind_label(indicator)}: %{{y:.2f}}<extra></extra>",
                ))
                # CI band
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc["year"], fc["year"].iloc[::-1]]),
                    y=pd.concat([fc["upper_95"], fc["lower_95"].iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(0,201,167,0.08)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False, hoverinfo="skip",
                ))

    # Forecast zone shading
    if with_forecast:
        last_year = int(PANEL["year"].max())
        fig.add_vrect(
            x0=last_year, x1=last_year + 5,
            fillcolor="rgba(255,255,255,0.02)",
            line=dict(color=BORDER, width=1, dash="dot"),
            annotation_text="forecast", annotation_position="top left",
            annotation_font=dict(color=TEXT_DIM, size=10),
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=ind_label(indicator), font=dict(size=14, color=TEXT_DIM)),
        xaxis=dict(**AXIS, title="Year"),
        yaxis=dict(**AXIS, title=ind_label(indicator)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
        height=420,
    )
    return fig


def fig_apac_map(indicator, year, apac_only=True):
    df = PANEL[(PANEL["year"]==year) & (PANEL["is_apac"]==True)] if apac_only else PANEL[PANEL["year"]==year]
    df = df[df[indicator].notna()] if indicator in df.columns else pd.DataFrame()
    if df.empty:
        return go.Figure()

    sdg_n = sdg_number(indicator)
    color = SDG_COLORS.get(sdg_n, ACCENT)

    fig = go.Figure(go.Choropleth(
        locations=df["iso3"],
        z=df[indicator],
        text=df["iso3"].map(NAME_LOOKUP),
        colorscale=[
            [0, DARK_BG],
            [0.3, color],
            [1, color],
        ],
        marker_line_color=BORDER,
        marker_line_width=0.5,
        colorbar=dict(
            title=dict(text=ind_label(indicator)[:20], font=dict(color=TEXT_DIM)),
            tickfont=dict(color=TEXT_DIM, family=FONT_MONO),
            bgcolor="rgba(0,0,0,0)",
            outlinecolor=BORDER,
        ),
        hovertemplate="<b>%{text}</b><br>" + ind_label(indicator) + ": %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        geo=dict(
            scope="asia" if apac_only else "world",
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor=BORDER,
            showland=True, landcolor=CARD_BG,
            showocean=True, oceancolor=DARK_BG,
            showlakes=False,
            projection_type="natural earth",
            center=dict(lon=105, lat=20),
        ),
        title=dict(
            text=f"Asia-Pacific: {ind_label(indicator)} ({year})",
            font=dict(size=13, color=TEXT_DIM),
        ),
        height=450,
    )
    return fig


def fig_sdg_scoreboard(iso3):
    """Traffic-light status per SDG goal for a single country."""
    if not iso3:
        return go.Figure()

    cdf = PANEL[PANEL["iso3"]==iso3].sort_values("year")
    latest_year = cdf["year"].max()
    latest = cdf[cdf["year"]==latest_year].iloc[0] if len(cdf) else pd.Series()

    rows = []
    for sdg_num in sorted(SDG_NAMES.keys()):
        inds = [i for i in SDG_INDICATORS if sdg_number(i)==sdg_num and i in latest.index]
        if not inds:
            rows.append({"SDG": f"SDG {sdg_num}", "Goal": SDG_NAMES[sdg_num],
                         "Status": "No data", "Value": "—"})
            continue

        # Use first available indicator as proxy
        ind  = inds[0]
        val  = latest.get(ind, np.nan)
        if pd.isna(val):
            rows.append({"SDG": f"SDG {sdg_num}", "Goal": SDG_NAMES[sdg_num],
                         "Status": "No data", "Value": "—"})
            continue

        # Compute trend over last 5 years
        recent = cdf[cdf["year"] >= latest_year - 5][ind].dropna()
        if len(recent) >= 2:
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            # Heuristic: positive trend = improving for most SDGs
            # (reverse for poverty, mortality, CO2)
            reverse = any(x in ind for x in ["poverty","mortality","co2","gini","unemployment"])
            improving = (trend < 0) if reverse else (trend > 0)
            status = "On track" if improving else "Off track"
        else:
            status = "Insufficient data"

        rows.append({
            "SDG":    f"SDG {sdg_num}",
            "Goal":   SDG_NAMES[sdg_num],
            "Status": status,
            "Value":  f"{val:.2f}",
        })

    df_score = pd.DataFrame(rows)
    return df_score


def fig_bar_comparison(indicator, year, apac_only=True, ranking_mode="top20", selected=[]):
    df = PANEL[PANEL["year"]==year].copy()
    if apac_only:
        df = df[df["is_apac"]==True]
    df = df[df[indicator].notna()] if indicator in df.columns else pd.DataFrame()
    if df.empty:
        return go.Figure()

    if ranking_mode == "selected" and selected:
        df = df[df["iso3"].isin(selected)].sort_values(indicator, ascending=True)
    else:
        df = df.nlargest(20, indicator)
    df["country_label"] = df["iso3"].map(NAME_LOOKUP).fillna(df["iso3"])

    sdg_n = sdg_number(indicator)
    color = SDG_COLORS.get(sdg_n, ACCENT)
    bar_colors = [ACCENT if iso in selected else color for iso in df["iso3"]]

    fig = go.Figure(go.Bar(
        x=df[indicator],
        y=df["country_label"],
        orientation="h",
        marker=dict(
            color=bar_colors,
            showscale=False,
        ),
        text=[f"{v:.1f}" for v in df[indicator]],
        textposition="outside",
        textfont=dict(color=TEXT_DIM, size=10, family=FONT_MONO),
        hovertemplate="<b>%{y}</b><br>" + ind_label(indicator) + ": %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"{ind_label(indicator)} — {year}", font=dict(size=13, color=TEXT_DIM)),
        xaxis=dict(**AXIS, title=ind_label(indicator)),
        yaxis=dict(**AXIS, title=""),
        height=max(350, len(df)*22),
        bargap=0.3,
    )
    return fig


# ── APP LAYOUT ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap",
    ],
    title="SDG Progress Analytics",
    suppress_callback_exceptions=True,
)
server = app.server

# Controls
CONTROLS = dbc.Card([
    html.Div("FILTERS", style={
        "fontFamily": FONT_MONO, "fontSize": "10px",
        "letterSpacing": "2px", "color": ACCENT,
        "marginBottom": "16px", "fontWeight": "600",
    }),

    html.Label("Region", style={"color": TEXT_DIM, "fontSize": "12px"}),
    dcc.RadioItems(
        id="region-toggle",
        options=[
            {"label": " Asia-Pacific only", "value": "apac"},
            {"label": " Global", "value": "global"},
        ],
        value="apac",
        style={"color": TEXT, "fontSize": "13px", "marginBottom": "12px"},
    ),

    html.Label("Countries", style={"color": TEXT_DIM, "fontSize": "12px"}),
    dcc.Dropdown(
        id="country-select",
        options=country_options(apac_only=True),
        value=["CHN", "IND", "IDN", "PHL", "VNM"],
        multi=True,
        style={"marginBottom": "12px"},
        className="dark-dropdown",
    ),

    html.Label("SDG Indicator", style={"color": TEXT_DIM, "fontSize": "12px"}),
    dcc.Dropdown(
        id="indicator-select",
        options=indicator_options(),
        value=SDG_INDICATORS[0] if SDG_INDICATORS else None,
        clearable=False,
        style={"marginBottom": "12px"},
    ),

    html.Label("Year (for map & bar)", style={"color": TEXT_DIM, "fontSize": "12px"}),
    dcc.Slider(
        id="year-slider",
        min=2000, max=2023, step=1, value=2022,
        marks={y: {"label": str(y), "style": {"color": TEXT_DIM, "fontSize": "10px"}}
               for y in range(2000, 2024, 5)},
        tooltip={"placement": "bottom"},
    ),

    html.Div(style={"marginTop": "16px"}),
    html.Label("Rankings view", style={"color": TEXT_DIM, "fontSize": "12px", "marginTop": "12px"}),
    dcc.RadioItems(
        id="ranking-mode",
        options=[
            {"label": " Top 20 (highlight selected)", "value": "top20"},
            {"label": " Selected countries only", "value": "selected"},
        ],
        value="top20",
        style={"color": TEXT, "fontSize": "13px", "marginBottom": "12px"},
    ),

    dcc.Checklist(
        id="forecast-toggle",
        options=[{"label": " Show 5-year forecast", "value": "forecast"}],
        value=["forecast"],
        style={"color": TEXT, "fontSize": "13px"},
    ),

], style={
    "background": CARD_BG, "border": f"1px solid {BORDER}",
    "borderRadius": "12px", "padding": "20px",
})


def _kpi(value, label):
    return html.Div([
        html.Div(value, style={
            "fontFamily": FONT_MONO, "fontSize": "16px",
            "fontWeight": "700", "color": ACCENT,
        }),
        html.Div(label, style={
            "fontSize": "9px", "color": TEXT_DIM,
            "letterSpacing": "1px", "textTransform": "uppercase",
        }),
    ], style={
        "background": CARD_BG, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "10px 14px", "textAlign": "center",
    })

HEADER = html.Div([
    html.Div([
        html.Div([
            html.Span("SDG · GLOBAL ANALYTICS", style={
            "fontFamily": FONT_MONO, "fontSize": "10px",
            "letterSpacing": "3px", "color": ACCENT,
            "fontWeight": "600",
        }),
        html.H1("SDG Progress Analytics", style={
            "fontFamily": FONT_BODY, "fontWeight": "700",
            "fontSize": "clamp(20px, 2.5vw, 30px)",
            "color": TEXT, "margin": "6px 0 4px",
        }),
        html.P("Tracking Sustainable Development Goal indicators across 268 countries · World Bank WDI · WHO GHED · V-Dem", style={
            "color": TEXT_DIM, "fontSize": "12px", "margin": "0",
        }),
        ]),
        html.Div([
            _kpi("268", "Countries"),
            _kpi("2000–2023", "Coverage"),
            _kpi(str(len(SDG_INDICATORS)), "Indicators"),
            _kpi(str(len(APAC_COUNTRIES)), "APAC DMCs"),
        ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
    ], style={
        "display": "flex", "justifyContent": "space-between",
        "alignItems": "center", "flexWrap": "wrap", "gap": "16px",
        "padding": "24px 32px",
        "borderBottom": f"1px solid {BORDER}",
        "background": f"linear-gradient(135deg, {DARK_BG} 0%, #080f24 100%)",
    }),
])



TABS = dbc.Tabs([
    dbc.Tab(label="Trend Analysis",    tab_id="tab-trend"),
    dbc.Tab(label="Regional Map",  tab_id="tab-map"),
    dbc.Tab(label="Country Rankings",  tab_id="tab-bar"),
    dbc.Tab(label="Country Scorecard", tab_id="tab-score"),
], id="sdg-tabs", active_tab="tab-trend", style={
    "padding": "0 32px",
    "borderBottom": f"1px solid {BORDER}",
    "background": DARK_BG,
})




app.layout = html.Div([
    HEADER,
    TABS,
    html.Div([
        dbc.Row([
            dbc.Col(CONTROLS, lg=3, style={"marginBottom": "16px"}),
            dbc.Col(html.Div(id="sdg-tab-content"), lg=9),
        ]),
    ], style={"padding": "24px 32px"}),
], style={"background": DARK_BG, "minHeight": "100vh", "fontFamily": FONT_BODY})


# ── CALLBACKS ─────────────────────────────────────────────────────────────────
@callback(Output("country-select", "options"),
          Output("country-select", "value"),
          Input("region-toggle", "value"))
def update_country_options(region):
    apac = region == "apac"
    opts = country_options(apac_only=apac)
    default = ["CHN","IND","IDN","PHL","VNM"] if apac else ["CHN","IND","USA","GBR","BRA"]
    return opts, default


@callback(Output("sdg-tab-content", "children"),
          Input("sdg-tabs", "active_tab"),
          Input("country-select", "value"),
          Input("indicator-select", "value"),
          Input("year-slider", "value"),
          Input("forecast-toggle", "value"),
          Input("region-toggle", "value"),
          Input("ranking-mode", "value"))
def render_content(tab, countries, indicator, year, forecast_toggle, region, ranking_mode):
    if not indicator:
        return html.P("Select an indicator.", style={"color": TEXT_DIM})

    with_forecast = "forecast" in (forecast_toggle or [])
    apac_only = region == "apac"

    if tab == "tab-trend":
        return dcc.Graph(
            figure=fig_trend(countries or [], indicator, with_forecast),
            config={"displayModeBar": False},
        )

    elif tab == "tab-map":
        return dcc.Graph(
            figure=fig_apac_map(indicator, year, apac_only),
            config={"displayModeBar": False},
        )

    elif tab == "tab-bar":
        return dcc.Graph(
            figure=fig_bar_comparison(indicator, year, apac_only, ranking_mode, countries or []),
            config={"displayModeBar": False},
        )

    elif tab == "tab-score":
        iso3 = (countries or ["CHN"])[0]
        df_score = fig_sdg_scoreboard(iso3)
        cname = NAME_LOOKUP.get(iso3, iso3)

        status_colors = {
            "On track": "#10b981", "Off track": "#ef4444",
            "Insufficient data": "#f59e0b", "No data": "#4b5563",
        }

        return html.Div([
            html.H4(f"SDG Scorecard — {cname}", style={
                "color": TEXT, "fontFamily": FONT_BODY, "marginBottom": "16px",
            }),
            html.P("Based on 5-year trend direction. On track = improving trend.",
                   style={"color": TEXT_DIM, "fontSize": "12px", "marginBottom": "16px"}),
            html.Div([
                html.Div([
                    html.Div(row["SDG"], style={
                        "fontFamily": FONT_MONO, "fontSize": "11px",
                        "color": str(SDG_COLORS.get(int(row["SDG"].split()[1]), ACCENT)),
                        "fontWeight": "600", "marginBottom": "4px",
                    }),
                    html.Div(row["Goal"], style={"fontSize": "12px", "color": TEXT}),
                    html.Div(row["Status"], style={
                        "fontSize": "11px", "fontWeight": "600",
                        "color": status_colors.get(row["Status"], TEXT_DIM),
                        "marginTop": "4px",
                    }),
                    html.Div(row["Value"], style={
                        "fontFamily": FONT_MONO, "fontSize": "13px",
                        "color": TEXT_DIM, "marginTop": "2px",
                    }),
                ], style={
                    "background": CARD_BG,
                    "border": f"1px solid {BORDER}",
                    "borderLeft": f"3px solid {status_colors.get(row['Status'], BORDER)}",
                    "borderRadius": "8px",
                    "padding": "12px 14px",
                })
                for _, row in df_score.iterrows()
            ], style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fill, minmax(180px, 1fr))",
                "gap": "10px",
            }),
        ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8051))
    print(f"\n{'='*50}")
    print(f"  SDG Analytics Dashboard")
    print(f"  http://127.0.0.1:{port}")
    print(f"{'='*50}\n")
    app.run(debug=True, port=port, host="0.0.0.0")
