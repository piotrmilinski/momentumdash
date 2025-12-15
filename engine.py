"""Core calculations for the Streamlit dashboard."""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

import config


DATA_DIR = os.path.join(os.getcwd(), "data")
PRICES_PATH = os.path.join(DATA_DIR, "prices_daily.csv")
META_PATH = os.path.join(DATA_DIR, "meta.json")


def fetch_prices_yahoo(
    tickers: Iterable[str], start=None, end=None, verify: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Fetch daily adjusted close prices from Yahoo Finance ticker-by-ticker.
def fetch_prices_yahoo(tickers: Iterable[str], start=None, end=None) -> pd.DataFrame:
    """Fetch daily adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers: Iterable[str]
        List of ticker symbols.
    start, end: optional
        Date boundaries for the download.
    verify: bool, default True
        Whether to verify SSL certificates for HTTPS requests. Setting to False
        can help in restrictive corporate environments with custom proxies.

    Returns
    -------
    prices: pd.DataFrame
        Adjusted close prices with tickers as columns.
    failures: Dict[str, str]
        Mapping of tickers that failed to download to error messages.
    """

    frames = []
    failures: Dict[str, str] = {}

    session = None
    if not verify:
        session = requests.Session()
        session.verify = False

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                session=session,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                if "Adj Close" in df.columns:
                    series = df["Adj Close"].rename(ticker)
                elif "Close" in df.columns:
                    series = df["Close"].rename(ticker)
                else:
                    failures[ticker] = "Missing close prices in response"
                    continue
                frames.append(series)
            else:
                failures[ticker] = "No data returned"
        except Exception as exc:  # pragma: no cover - defensive
            failures[ticker] = str(exc)

    prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    prices = prices.dropna(how="all")
    return prices, failures
    """

    df = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
        df = df["Adj Close"].copy()
    df = df.dropna(how="all")
    return df


def resample_weekly(prices: pd.DataFrame, rule: str = config.ASOF_WEEKLY_RULE) -> pd.DataFrame:
    """Resample daily prices to weekly closes using the supplied rule."""

    return prices.resample(rule).last().dropna(how="all")


def portfolio_vol_annualized(returns: pd.DataFrame, weights: pd.Series) -> float:
    """Compute annualized portfolio volatility from weekly returns and weights."""

    if returns.empty or weights.empty:
        return float("nan")

    cov = returns.cov()
    vol = np.sqrt(np.dot(weights, np.dot(cov.values, weights.T)))
    return float(vol * np.sqrt(52))


def _compute_breakouts(prices_w: pd.DataFrame) -> pd.DataFrame:
    lookback = 52
    rolling_high = prices_w.shift(1).rolling(lookback).max()
    rolling_low = prices_w.shift(1).rolling(lookback).min()
    breakout_high = prices_w > rolling_high
    breakout_low = prices_w < rolling_low
    out = pd.DataFrame(index=prices_w.index, columns=prices_w.columns)
    out[breakout_high] = "52W High"
    out[breakout_low] = "52W Low"
    return out


def _compute_scores(prices_w: pd.DataFrame, returns_w: pd.DataFrame, model: str, spec: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    horizons = spec["horizons"]
    vol_lookback = spec["vol_lookback"]
    deadband = spec["deadband"]
    tanh_scale = spec["tanh_scale"]

    vol = returns_w.rolling(vol_lookback).std()
    vol = vol.replace(0.0, np.nan)

    composite = pd.DataFrame(index=prices_w.index, columns=prices_w.columns, dtype=float)
    for horizon, weight in horizons.items():
        horizon_ret = prices_w.pct_change(horizon)
        risk_adj = horizon_ret / vol
        composite = composite.add(weight * risk_adj, fill_value=0.0)

    composite = composite.clip(*config.SCORE_CLIP)
    pos = np.tanh(composite * tanh_scale)
    signals = pos.applymap(lambda x: "Flat" if abs(x) < deadband else ("Long" if x > 0 else "Short"))
    return pos, vol, composite.iloc[-1]


def _normalize_inverse_vol_positions(pos: pd.DataFrame, vol: pd.DataFrame, target_vol: float, cov_lookback: int, max_leverage: float, returns_w: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inv_vol = pos.divide(vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = inv_vol.abs().sum(axis=1)
    base_weights = inv_vol.div(gross.where(gross != 0, np.nan), axis=0).fillna(0.0)

    scaled_weights = base_weights.copy()
    for dt, w in base_weights.iterrows():
        lookback_returns = returns_w.loc[:dt].tail(cov_lookback)
        if lookback_returns.shape[0] < 2:
            continue
        port_vol = portfolio_vol_annualized(lookback_returns, w)
        if not port_vol or np.isnan(port_vol) or port_vol == 0:
            continue
        scale = target_vol / port_vol
        scaled = (w * scale).clip(-max_leverage, max_leverage)
        scaled_weights.loc[dt] = scaled

    risk = scaled_weights.abs() * vol
    return scaled_weights, risk


def compute_weekly_trend_model(
    prices_daily: pd.DataFrame | None = None,
    models: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute weekly model outputs.

    Returns
    -------
    prices_w: Weekly resampled prices.
    positions: Dict of per-model position DataFrames.
    notional_weights: Dict of per-model notional weight DataFrames.
    risk_tables: Dict of per-model risk approximation DataFrames.
    latest_snapshot: DataFrame summarizing the latest week.
    """

    if prices_daily is None:
        if not os.path.exists(PRICES_PATH):
            raise FileNotFoundError("Cached prices not found. Run update_data.py first.")
        prices_daily = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    prices_daily = prices_daily.reindex(columns=sorted(config.UNIVERSE.keys())).dropna(how="all")
    if prices_daily.empty:
        raise ValueError("No price data available for the requested universe.")

    prices_w = resample_weekly(prices_daily)
    returns_w = prices_w.pct_change()

    models = models or list(config.MODEL_SPECS.keys())

    positions: Dict[str, pd.DataFrame] = {}
    notional_weights: Dict[str, pd.DataFrame] = {}
    risk_tables: Dict[str, pd.DataFrame] = {}
    score_latest: Dict[str, pd.Series] = {}
    signals_latest: Dict[str, pd.Series] = {}

    for model in models:
        spec = config.MODEL_SPECS.get(model)
        if spec is None:
            continue
        pos, vol, last_scores = _compute_scores(prices_w, returns_w, model, spec)
        positions[model] = pos
        score_latest[model] = last_scores

        notional, risk = _normalize_inverse_vol_positions(
            pos,
            vol,
            spec["target_portfolio_vol"],
            spec["cov_lookback"],
            spec["max_leverage_per_asset"],
            returns_w,
        )
        notional_weights[model] = notional
        risk_tables[model] = risk
        signals_latest[model] = pos.iloc[-1].apply(lambda x: "Flat" if abs(x) < spec["deadband"] else ("Long" if x > 0 else "Short"))

    breakout_flags = _compute_breakouts(prices_w)
    latest_date = prices_w.index.max()
    latest_prices = prices_w.loc[latest_date]
    breakout_latest = breakout_flags.loc[latest_date]

    rows = []
    for ticker, meta in config.UNIVERSE.items():
        row = {
            "Ticker": ticker,
            "Class": meta["class"],
            "Name": meta["name"],
            "Price": latest_prices.get(ticker, np.nan),
            "Breakout": breakout_latest.get(ticker, ""),
        }
        for model in models:
            pos_df = positions.get(model)
            weight_df = notional_weights.get(model)
            risk_df = risk_tables.get(model)
            if pos_df is None:
                continue
            row[f"{model} Position"] = pos_df.iloc[-1].get(ticker, np.nan)
            row[f"{model} Signal"] = signals_latest[model].get(ticker, "")
            row[f"{model} Weight"] = weight_df.iloc[-1].get(ticker, np.nan) if weight_df is not None else np.nan
            row[f"{model} Risk"] = risk_df.iloc[-1].get(ticker, np.nan) if risk_df is not None else np.nan
            row[f"{model} Score"] = score_latest.get(model, pd.Series()).get(ticker, np.nan)
        rows.append(row)

    latest_snapshot = pd.DataFrame(rows)
    return prices_w, positions, notional_weights, risk_tables, latest_snapshot


def load_cached_meta() -> dict:
    """Load metadata file if it exists."""

    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}
