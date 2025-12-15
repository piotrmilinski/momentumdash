"""Download and cache daily prices for the dashboard."""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

import config
from engine import DATA_DIR, META_PATH, PRICES_PATH, fetch_prices_yahoo


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tickers = list(config.UNIVERSE.keys())
    print(f"Downloading prices for {len(tickers)} tickers starting {config.START_DATE}...")
    prices, failures = fetch_prices_yahoo(
        tickers, start=config.START_DATE, verify=config.YAHOO_VERIFY_SSL
    )
    if failures:
        print("\nSome tickers failed to download:")
        for t, msg in failures.items():
            print(f" - {t}: {msg}")

    if prices.empty:
        if os.path.exists(PRICES_PATH):
            print(
                "No new data downloaded; keeping existing cache at"
                f" {PRICES_PATH}."
            )
            return
        raise RuntimeError("No data downloaded. Check network access or tickers.")

    if os.path.exists(PRICES_PATH):
        existing = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        if not existing.empty:
            prices = existing.combine_first(prices)

    prices = prices.sort_index()
    prices.to_csv(PRICES_PATH)

    meta = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "latest_date": prices.index.max().isoformat(),
        "tickers": tickers,
        "failed_tickers": sorted(failures.keys()),
    }
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved {len(prices)} rows to {PRICES_PATH}")


if __name__ == "__main__":
    main()
