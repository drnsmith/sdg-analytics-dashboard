"""
forecasting.py
==============
Trend forecasting for SDG indicators.
Uses statsmodels ExponentialSmoothing (no Prophet dependency).

Run:
    PYTHONPATH=. python src/forecasting.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed"


def forecast_indicator(series: pd.Series, horizon: int = 7) -> pd.DataFrame:
    # Ensure clean integer index
    series = series.dropna()
    series.index = series.index.astype(int)
    """
    Forecast a single time series using Holt linear trend.
    Returns DataFrame with year, value, lower_95, upper_95, is_forecast.
    """
    series = series.dropna().sort_index()
    if len(series) < 4:
        return pd.DataFrame()

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        model  = ExponentialSmoothing(series, trend="add", damped_trend=True)
        fitted = model.fit(optimized=True, remove_bias=True)
        fcast  = fitted.forecast(horizon)

        # Simple prediction interval (±1.96 * residual std)
        resid_std = np.std(fitted.resid)
        se = resid_std * np.sqrt(np.arange(1, horizon + 1))

        hist = pd.DataFrame({
            "year":        series.index,
            "value":       series.values,
            "lower_95":    series.values,
            "upper_95":    series.values,
            "is_forecast": False,
        })
        future_years = np.arange(series.index[-1] + 1,
                                  series.index[-1] + horizon + 1)
        fore = pd.DataFrame({
            "year":        future_years,
            "value":       fcast.values,
            "lower_95":    fcast.values - 1.96 * se,
            "upper_95":    fcast.values + 1.96 * se,
            "is_forecast": True,
        })
        return pd.concat([hist, fore], ignore_index=True)
    except Exception as e:
        log.debug(f"Forecast failed: {e}")
        return pd.DataFrame()


def build_forecast_cache(sdg_panel: pd.DataFrame,
                         indicators: list = None,
                         horizon: int = 7) -> pd.DataFrame:
    """
    Build forecast cache for all country × indicator combinations.
    Saves to data/processed/sdg_forecasts.parquet
    """
    if indicators is None:
        indicators = [c for c in sdg_panel.columns if c.startswith("sdg")]

    records = []
    countries = sdg_panel["iso3"].unique()
    log.info(f"Forecasting {len(indicators)} indicators × {len(countries)} countries...")

    for iso3 in countries:
        cdf = sdg_panel[sdg_panel["iso3"] == iso3].set_index("year")
        for ind in indicators:
            if ind not in cdf.columns:
                continue
            series = cdf[ind].dropna()
            if len(series) < 4:
                continue
            fc = forecast_indicator(series, horizon=horizon)
            if fc.empty:
                continue
            fc["iso3"]      = iso3
            fc["indicator"] = ind
            records.append(fc)

    if not records:
        log.warning("No forecasts generated")
        return pd.DataFrame()

    df = pd.concat(records, ignore_index=True)
    out = DATA_PROC / "sdg_forecasts.parquet"
    df.to_parquet(out, index=False)
    log.info(f"✓ Forecasts saved: {df.shape} → {out}")
    return df


def get_forecast(forecasts: pd.DataFrame,
                 iso3: str,
                 indicator: str) -> pd.DataFrame:
    """Retrieve forecast for a specific country × indicator."""
    mask = (forecasts["iso3"] == iso3) & (forecasts["indicator"] == indicator)
    return forecasts[mask].sort_values("year")


if __name__ == "__main__":
    panel_path = DATA_PROC / "sdg_panel.parquet"
    if not panel_path.exists():
        log.error("Run build_sdg_panel.py first")
    else:
        panel = pd.read_parquet(panel_path)
        build_forecast_cache(panel)
