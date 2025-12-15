"""Streamlit dashboard for multi-horizon momentum positioning."""
from __future__ import annotations

import os
import subprocess
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from engine import (
    DATA_DIR,
    META_PATH,
    PRICES_PATH,
    compute_weekly_trend_model,
    load_cached_meta,
)

st.set_page_config(page_title="Momentum Dashboard", layout="wide")


def _load_prices() -> pd.DataFrame:
    if not os.path.exists(PRICES_PATH):
        st.warning("Price cache not found. Click refresh to download data.")
        return pd.DataFrame()
    return pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)


def _refresh_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        subprocess.run(["python", "update_data.py"], check=True)
        st.success("Prices refreshed.")
    except subprocess.CalledProcessError as exc:
        st.error(f"Failed to refresh prices: {exc}")


def _render_snapshot(snapshot: pd.DataFrame, models: List[str]):
    cols = ["Ticker", "Class", "Name", "Price", "Breakout"]
    for model in models:
        cols.extend([
            f"{model} Position",
            f"{model} Signal",
            f"{model} Weight",
            f"{model} Risk",
            f"{model} Score",
        ])
    cols = [c for c in cols if c in snapshot.columns]
    display = snapshot[cols].copy()
    st.subheader("Current positioning snapshot")
    st.dataframe(display.set_index("Ticker"))


def _compute_aggregates(weights: Dict[str, pd.DataFrame], risk: Dict[str, pd.DataFrame]):
    aggregates = {}
    latest = next(iter(weights.values())).index.max() if weights else None
    if latest is None:
        return aggregates
    for model, df in weights.items():
        latest_weights = df.loc[latest] if latest in df.index else pd.Series(dtype=float)
        risk_df = risk.get(model, pd.DataFrame())
        latest_risk = risk_df.loc[latest] if not risk_df.empty and latest in risk_df.index else pd.Series(dtype=float)
        class_rows = []
        for ticker, meta in config.UNIVERSE.items():
            risk_weight = latest_risk.get(ticker, np.nan)
            class_rows.append({
                "class": meta["class"],
                "risk_weight": risk_weight,
                "weight": latest_weights.get(ticker, np.nan),
            })
        class_df = pd.DataFrame(class_rows).fillna(0.0)
        agg = class_df.groupby("class").apply(
            lambda g: pd.Series(
                {
                    "Position": (np.sign(g["weight"]) * g["risk_weight"]).sum() / g["risk_weight"].sum() if g["risk_weight"].sum() != 0 else np.nan,
                    "Net Weight": g["weight"].sum(),
                    "Gross Weight": g["weight"].abs().sum(),
                    "Risk": g["risk_weight"].sum(),
                }
            )
        )
        aggregates[model] = agg
    return aggregates


def _render_heatmap(positions: Dict[str, pd.DataFrame], models: List[str]):
    if not positions:
        return
    latest = next(iter(positions.values())).index.max()
    matrix = pd.DataFrame({model: df.loc[latest] for model, df in positions.items() if model in models and latest in df.index})
    matrix = matrix.reindex(config.UNIVERSE.keys())
    if matrix.empty:
        return
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Position"),
    ))
    fig.update_layout(height=500, title="Positions heatmap (latest week)")
    st.plotly_chart(fig, use_container_width=True)


def _render_breakouts(prices_w: pd.DataFrame):
    lookback = 52
    rolling_high = prices_w.shift(1).rolling(lookback).max()
    rolling_low = prices_w.shift(1).rolling(lookback).min()
    latest = prices_w.index.max()
    latest_px = prices_w.loc[latest]
    highs = latest_px[latest_px > rolling_high.loc[latest]].index.tolist()
    lows = latest_px[latest_px < rolling_low.loc[latest]].index.tolist()
    st.subheader("52-week breakout watchlist")
    st.write({"Highs": highs, "Lows": lows})


def _render_drilldown(prices_w: pd.DataFrame, positions: Dict[str, pd.DataFrame], models: List[str]):
    ticker = st.selectbox("Select instrument", list(config.UNIVERSE.keys()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices_w.index, y=prices_w[ticker], name=f"{ticker} Price", line=dict(color="black")))
    for model in models:
        pos_df = positions.get(model)
        if pos_df is None or ticker not in pos_df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=pos_df.index,
                y=pos_df[ticker],
                name=f"{model} Position",
                yaxis="y2",
            )
        )
    fig.update_layout(
        title=f"{ticker} weekly price and positions",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Position", overlaying="y", side="right", range=[-1.05, 1.05]),
        legend_orientation="h",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Weekly Trend-Following Dashboard")
    st.write("Momentum and breakout overview across ETFs.")

    if st.button("Refresh prices now"):
        _refresh_data()

    prices_daily = _load_prices()
    if prices_daily.empty:
        st.stop()

    model_choices = list(config.MODEL_SPECS.keys())
    models = st.multiselect("Models", model_choices, default=model_choices)

    try:
        prices_w, positions, weights, risk_tables, snapshot = compute_weekly_trend_model(prices_daily, models=models)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to compute models: {exc}")
        st.stop()

    meta = load_cached_meta()
    st.caption(f"Last updated: {meta.get('updated_at', 'unknown')}")

    _render_snapshot(snapshot, models)

    aggregates = _compute_aggregates(weights, risk_tables)
    st.subheader("Asset class aggregates (risk-weighted)")
    for model, df in aggregates.items():
        st.markdown(f"**{model}**")
        st.dataframe(df)

    _render_heatmap(positions, models)
    _render_breakouts(prices_w)
    _render_drilldown(prices_w, positions, models)


if __name__ == "__main__":
    main()
