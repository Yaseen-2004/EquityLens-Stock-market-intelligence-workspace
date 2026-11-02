"""Fetch daily prices + fundamentals from Yahoo Finance into the database.

Run it as a script:

    python ingest.py              # this year -> today, live data
    python ingest.py --offline    # deterministic synthetic data, no network
    python ingest.py --reset      # wipe price history first
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time

from sqlalchemy import delete, select

import config
import net
from database import init_db, session_scope
from models import Company, PriceHistory

log = logging.getLogger("ingest")


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------
def _to_epoch(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day).timestamp())


def fetch_history(symbol: str, start: dt.date, end: dt.date) -> list[dict]:
    """Daily OHLCV bars for `symbol` in [start, end].

    Uses the public chart endpoint, which needs no crumb.
    """
    url = config.YAHOO_CHART_URL.format(symbol=symbol)
    params = {
        "period1": _to_epoch(start),
        # +1 day so today's bar is inclusive.
        "period2": _to_epoch(end + dt.timedelta(days=1)),
        "interval": "1d",
        "events": "div,split",
    }
    r = net.get_session().get(url, params=params, timeout=config.REQUEST_TIMEOUT)
    r.raise_for_status()
    payload = r.json()

    result = (payload.get("chart") or {}).get("result")
    if not result:
        log.warning("%s: no chart result", symbol)
        return []

    block = result[0]
    stamps = block.get("timestamp") or []
    quote = (block.get("indicators", {}).get("quote") or [{}])[0]
    # Adjusted close accounts for splits/dividends - better for comparisons.
    adj = (block.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")

    rows: list[dict] = []
    for i, ts in enumerate(stamps):
        close = (adj[i] if adj and i < len(adj) and adj[i] is not None
                 else _at(quote.get("close"), i))
        if close is None:
            continue  # holiday / halted day
        day = dt.datetime.fromtimestamp(ts).date()
        # Yahoo's period1 is timezone-sensitive and can bleed one bar in from
        # the previous year; clamp hard to the requested window.
        if day < start or day > end:
            continue
        rows.append({
            "date":   day,
            "open":   _at(quote.get("open"), i),
            "high":   _at(quote.get("high"), i),
            "low":    _at(quote.get("low"), i),
            "close":  float(close),
            "volume": int(_at(quote.get("volume"), i) or 0),
        })
    return rows


def _at(seq, i):
    """Safe indexed access - Yahoo pads arrays with nulls."""
    if not seq or i >= len(seq):
        return None
    return seq[i]


def fetch_quotes(symbols: list[str]) -> dict[str, dict]:
    """Market cap / exchange metadata. Returns {} if the crumb handshake fails."""
    crumb = net.get_crumb()
    if not crumb:
        log.warning("no crumb - market caps unavailable, ranking will fall back")
        return {}

    out: dict[str, dict] = {}
    # Yahoo caps the symbol list length; chunk to stay well under it.
    for i in range(0, len(symbols), 10):
        chunk = symbols[i:i + 10]
        try:
            r = net.get_session().get(
                config.YAHOO_QUOTE_URL,
                params={"symbols": ",".join(chunk), "crumb": crumb},
                timeout=config.REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            for q in r.json().get("quoteResponse", {}).get("result", []):
                out[q["symbol"]] = q
        except Exception as exc:  # network hiccup shouldn't kill the run
            log.warning("quote chunk %s failed: %s", chunk, exc)
    return out


# --------------------------------------------------------------------------
# Persisting
# --------------------------------------------------------------------------
def upsert_company(session, symbol: str, name: str, quote: dict | None) -> Company:
    company = session.scalar(select(Company).where(Company.symbol == symbol))
    if company is None:
        company = Company(symbol=symbol, name=name)
        session.add(company)

    company.name = name
    if quote:
        company.market_cap = quote.get("marketCap")
        company.currency = quote.get("currency", "USD")
        company.exchange = quote.get("fullExchangeName")
    session.flush()  # assign company.id
    return company


def upsert_prices(session, company: Company, rows: list[dict]) -> tuple[int, int]:
    """Insert new bars, update changed ones. Idempotent, so re-runs are safe."""
    existing = {
        p.date: p
        for p in session.scalars(
            select(PriceHistory).where(PriceHistory.company_id == company.id)
        )
    }
    inserted = updated = 0
    for row in rows:
        current = existing.get(row["date"])
        if current is None:
            session.add(PriceHistory(company_id=company.id, **row))
            inserted += 1
        elif abs((current.close or 0) - row["close"]) > 1e-9:
            # Adjusted closes shift after a split/dividend - keep them fresh.
            for k, v in row.items():
                setattr(current, k, v)
            updated += 1
    return inserted, updated


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(offline: bool = False, reset: bool = False,
        start: dt.date | None = None) -> None:
    init_db()
    start = start or config.year_start()
    end = config.today()
    log.info("window %s -> %s", start, end)

    if reset:
        with session_scope() as s:
            s.execute(delete(PriceHistory))
        log.info("cleared price_history")

    if not offline and not net.online():
        log.warning("Yahoo unreachable - switching to offline synthetic data")
        offline = True

    # Benchmarks ride along so the analysis has a market to compare against.
    universe = {**config.TICKERS, **config.BENCHMARKS}

    if offline:
        from seed_fallback import generate_history, SYNTHETIC_CAPS
        quotes = {}
    else:
        quotes = fetch_quotes(list(universe))

    total_i = total_u = 0
    for symbol, name in universe.items():
        try:
            if offline:
                rows = generate_history(symbol, start, end)
                quote = {"marketCap": SYNTHETIC_CAPS.get(symbol),
                         "currency": "USD", "fullExchangeName": "SYNTHETIC"}
            else:
                rows = fetch_history(symbol, start, end)
                quote = quotes.get(symbol)
                time.sleep(0.3)  # be polite to the API

            if not rows:
                log.warning("%-6s no rows", symbol)
                continue

            with session_scope() as s:
                company = upsert_company(s, symbol, name, quote)
                i, u = upsert_prices(s, company, rows)

            total_i += i
            total_u += u
            log.info("%-6s %3d bars  (+%d new, ~%d updated)  last close %.2f",
                     symbol, len(rows), i, u, rows[-1]["close"])
        except Exception as exc:
            log.error("%-6s FAILED: %s", symbol, exc)

    log.info("done: %d inserted, %d updated", total_i, total_u)


def main() -> None:
    ap = argparse.ArgumentParser(description="Load stock data into the database")
    ap.add_argument("--offline", action="store_true",
                    help="generate deterministic synthetic data instead")
    ap.add_argument("--reset", action="store_true",
                    help="delete existing price history first")
    ap.add_argument("--years", type=int, metavar="N",
                    help="fetch N years of history instead of year-to-date")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    start = config.history_start(args.years) if args.years else None
    run(offline=args.offline, reset=args.reset, start=start)


if __name__ == "__main__":
    main()
