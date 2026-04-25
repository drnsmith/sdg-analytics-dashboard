# The Sustainable Development Goals Progress Analytics Dashboard

**Live demo:** [https://drnsmith-sdg-analytics-dashboard.hf.space](https://huggingface.co/spaces/drnsmith/sdg-analytics-dashboard)

---

## Why I built this

The Sustainable Development Goals were adopted in 2015 with a 2030 deadline. We are now past the halfway point and the honest answer, for most countries on most goals, is: we don't really know how things are going.

Part of the problem is data. SDG monitoring relies on dozens of indicators drawn from different sources — World Bank, WHO, V-Dem, IMF — each with different coverage, different update frequencies, and different methodologies. Pulling them together into something coherent enough to reason about is non-trivial. Most of the official SDG tracking tools are either too aggregated to be useful at the country level, or too raw to be interpretable by anyone without a statistics background.

I built this dashboard because I wanted to do that work properly — not just aggregate the data, but build a forecasting layer on top of it that gives a genuine answer to the question development analysts actually ask: is this country on track, and where is it likely to be by 2030?

The forecasting method is Holt-Winters exponential smoothing, fitted per country per indicator. It's not a causal model — I'm not claiming that past trends will continue because of any particular mechanism. But trend extrapolation is exactly what you need for SDG monitoring, and doing it with proper uncertainty quantification (95% prediction intervals, clearly labelled as projections) is more honest than either pretending we can't forecast or pretending the forecasts are certain.

The traffic-light scorecard is the output I'm most satisfied with. For each country and each SDG goal, I compute a 5-year OLS trend slope and classify the direction as on-track or off-track, with the direction of improvement defined correctly per indicator (falling poverty is on-track; falling electricity access is off-track). It reduces a complex multivariate picture to something a policy analyst can act on in thirty seconds.

---

## What I built

An ETL pipeline that ingests and harmonises SDG-aligned indicators from five sources into a unified 6,432 × 50 country-year panel covering 268 countries from 2000 to 2023. A forecasting engine that fits Holt-Winters models per country per indicator and generates projections to 2030. A Plotly Dash dashboard with four tabs designed for different analytical purposes.

The pipeline handles missing data gracefully — indicators with under 40% coverage are excluded from the dropdown rather than shown with misleading sparse data. The 28 indicators that remain have between 43% and 99% country-year coverage, which is honest about what the data can and cannot support.

---

## Dashboard tabs

**Trend Analysis** — Time series for any indicator across selected countries, 2000–2023, with Holt-Winters projections to 2030 and 95% confidence bands. Country colours are consistent across tabs. The forecast zone is clearly shaded and labelled.

**Regional Map** — Choropleth of any indicator for any year. Switches between Asia-Pacific scope and global scope based on the region filter.

**Country Rankings** — Two modes: top 20 globally with selected countries highlighted, or selected countries only ranked against each other. Useful for positioning a country within a global or regional peer group.

**Country Scorecard** — Traffic-light SDG status for a single country across all 13 SDG goals covered. Green = improving trend over the last 5 years. Red = deteriorating. Amber = insufficient data to determine.

---

## Coverage

| Indicator | Coverage |
|---|---|
| Life expectancy | 99% |
| Electricity access | 98% |
| GDP per capita | 96% |
| Clean water access | 96% |
| Sanitation access | 95% |
| GDP growth | 95% |
| Mobile subscriptions | 92% |
| GNI per capita | 91% |
| Under-5 mortality | 91% |
| Renewable energy | 89% |
| Internet users | 83% |
| Rule of law (WGI) | 73% |

---

## Data sources

| Source | SDGs | Coverage |
|---|---|---|
| World Bank WDI | 1, 3, 6, 7, 8, 9, 16, 17 | 268 countries · 2000–2023 |
| WHO GHED | SDG 3 (health expenditure) | 195 countries · 2000–2022 |
| V-Dem v14 | SDG 16 (governance) | 180+ countries · 1900–2023 |
| IMF Capital Stock | SDG 9 (infrastructure) | 170 countries · 1960–2019 |
| Global Forest Watch | SDG 15 (forests) | Global · 2001–2024 |

---

## Tech stack

- Python 3.11, statsmodels (Holt-Winters), pandas, numpy, pyarrow
- Plotly Dash 4.1.0, dash-bootstrap-components 2.0.4
- Deployed on Render

---

## Run locally

```bash
git clone https://github.com/drnsmith/sdg-analytics-dashboard.git
cd sdg-analytics-dashboard
pip install -r requirements.txt

PYTHONPATH=. python src/build_sdg_panel.py
PYTHONPATH=. python src/forecasting.py

PYTHONPATH=. python dashboard/app.py
# → http://127.0.0.1:8051
```

---

## Project structure

```
sdg-analytics-dashboard/
├── src/
│   ├── build_sdg_panel.py   # ETL: WDI + WHO + V-Dem + IMF + GFW → parquet
│   └── forecasting.py       # Holt-Winters projections to 2030
├── dashboard/
│   └── app.py               # Plotly Dash, 4 tabs
├── data/
│   └── processed/           # sdg_panel.parquet, sdg_forecasts.parquet
└── requirements.txt
```

---

---

## License

MIT — see [LICENSE](LICENSE).
