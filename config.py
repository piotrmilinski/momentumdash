"""Configuration for the momentum dashboard."""
import os
from datetime import datetime, timedelta

ASOF_WEEKLY_RULE = "W-FRI"
SCORE_CLIP = (-5.0, 5.0)
YEARS_HISTORY = 10

UNIVERSE = {
    "SPY": {"name": "S&P 500", "class": "US Equity"},
    "QQQ": {"name": "Nasdaq 100", "class": "US Equity"},
    "IWM": {"name": "Russell 2000", "class": "US Equity"},
    "EFA": {"name": "MSCI EAFE", "class": "International Equity"},
    "EEM": {"name": "MSCI Emerging", "class": "International Equity"},
    "VNQ": {"name": "REITs", "class": "Real Estate"},
    "TLT": {"name": "US Treasury 20+", "class": "Rates"},
    "LQD": {"name": "IG Credit", "class": "Credit"},
    "HYG": {"name": "High Yield", "class": "Credit"},
    "GLD": {"name": "Gold", "class": "Commodities"},
    "SLV": {"name": "Silver", "class": "Commodities"},
    "DBC": {"name": "Broad Commodities", "class": "Commodities"},
}

MODEL_SPECS = {
    "Fast": {
        "horizons": {4: 0.5, 12: 0.35, 24: 0.15},
        "vol_lookback": 16,
        "cov_lookback": 26,
        "tanh_scale": 1.25,
        "deadband": 0.1,
        "target_portfolio_vol": 0.12,
        "max_leverage_per_asset": 0.35,
    },
    "Core": {
        "horizons": {8: 0.4, 16: 0.35, 32: 0.25},
        "vol_lookback": 26,
        "cov_lookback": 52,
        "tanh_scale": 1.1,
        "deadband": 0.1,
        "target_portfolio_vol": 0.1,
        "max_leverage_per_asset": 0.3,
    },
    "Slow": {
        "horizons": {12: 0.25, 26: 0.35, 52: 0.4},
        "vol_lookback": 52,
        "cov_lookback": 78,
        "tanh_scale": 1.0,
        "deadband": 0.1,
        "target_portfolio_vol": 0.08,
        "max_leverage_per_asset": 0.25,
    },
}

START_DATE = (datetime.utcnow() - timedelta(days=365 * YEARS_HISTORY)).date()


def _env_flag(name: str, default: str = "true") -> bool:
    """Parse a boolean-like environment variable with a fallback."""

    return os.environ.get(name, default).lower() not in {"0", "false", "no", ""}


# Whether to verify SSL certificates for Yahoo Finance requests.
# Leave enabled by default; users facing corporate MITM/SSL issues can set the
# environment variable ``YAHOO_VERIFY_SSL=false`` to disable verification.
YAHOO_VERIFY_SSL = _env_flag("YAHOO_VERIFY_SSL", "true")
