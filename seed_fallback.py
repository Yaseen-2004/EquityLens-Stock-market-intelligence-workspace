"""Deterministic synthetic market data.

Used only when Yahoo Finance is unreachable, so the dashboard is always
demonstrable. Seeded per-symbol, so the same symbol always produces the same
series - reproducible screenshots and tests.
"""
from __future__ import annotations

import datetime as dt
import random

# Rough real-world anchors so the synthetic ranking still looks sensible.
SEED_PRICES = {
    "AAPL": 250.0, "MSFT": 430.0, "NVDA": 140.0, "GOOGL": 195.0, "AMZN": 225.0,
    "META": 600.0, "AVGO": 230.0, "TSLA": 400.0, "BRK-B": 470.0, "JPM": 240.0,
    "WMT": 92.0,  "LLY": 780.0, "V": 315.0,  "ORCL": 175.0, "MA": 525.0,
    "NFLX": 900.0, "XOM": 110.0, "COST": 950.0, "JNJ": 155.0, "AMD": 125.0,
}

SYNTHETIC_CAPS = {
    "AAPL": 3.80e12, "MSFT": 3.20e12, "NVDA": 3.40e12, "GOOGL": 2.40e12,
    "AMZN": 2.35e12, "META": 1.50e12, "AVGO": 1.08e12, "TSLA": 1.28e12,
    "BRK-B": 1.01e12, "JPM": 6.75e11, "WMT": 7.40e11, "LLY": 7.00e11,
    "V": 6.10e11, "ORCL": 4.90e11, "MA": 4.85e11, "NFLX": 3.85e11,
    "XOM": 4.75e11, "COST": 4.20e11, "JNJ": 3.75e11, "AMD": 2.05e11,
}

# Annualised drift per symbol - gives the YTD ranking some spread.
DRIFT = {
    "NVDA": 0.55, "AVGO": 0.42, "META": 0.30, "AMZN": 0.22, "MSFT": 0.18,
    "AAPL": 0.15, "GOOGL": 0.25, "ORCL": 0.35, "NFLX": 0.28, "AMD": -0.12,
    "TSLA": -0.05, "WMT": 0.20, "LLY": 0.10, "V": 0.12, "MA": 0.11,
    "JPM": 0.14, "BRK-B": 0.09, "XOM": 0.04, "COST": 0.16, "JNJ": 0.02,
}


def business_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """Weekdays only - a crude but adequate stand-in for a trading calendar."""
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += dt.timedelta(days=1)
    return days


def generate_history(symbol: str, start: dt.date, end: dt.date) -> list[dict]:
    """Geometric random walk with per-symbol drift and volatility."""
    rng = random.Random(f"{symbol}-{start.year}")  # stable across runs
    price = SEED_PRICES.get(symbol, 100.0)
    mu = DRIFT.get(symbol, 0.10) / 252.0   # daily drift
    sigma = 0.018                           # ~1.8% daily vol

    rows = []
    for day in business_days(start, end):
        shock = rng.gauss(mu, sigma)
        open_ = price
        close = max(1.0, price * (1 + shock))
        high = max(open_, close) * (1 + abs(rng.gauss(0, 0.004)))
        low = min(open_, close) * (1 - abs(rng.gauss(0, 0.004)))
        rows.append({
            "date": day,
            "open": round(open_, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": int(rng.uniform(8e6, 90e6)),
        })
        price = close
    return rows
