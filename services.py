"""Read-side queries. All the analytics the web layer needs live here."""
from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import Float, cast, func, select
from sqlalchemy.orm import Session

import config
from models import Company, PriceHistory


def _metrics_query(start: dt.date | None = None, end: dt.date | None = None):
    """Per-company aggregates over a DATE-SCOPED window.

    Built as: one GROUP BY pass for the bounds, then two joins back to the
    price table to pick the close on the first and last trading day.  That is
    two index lookups per company rather than pulling every row into Python.

    The window matters: the table also holds several years of history for the
    analysis report, so an unscoped aggregate would silently turn "year to
    date" into "since the beginning of the data".
    """
    start = start or config.year_start()
    end = end or config.today()

    bounds = (
        select(
            PriceHistory.company_id.label("cid"),
            func.min(PriceHistory.date).label("first_date"),
            func.max(PriceHistory.date).label("last_date"),
            func.avg(cast(PriceHistory.volume, Float)).label("avg_volume"),
            func.max(PriceHistory.close).label("period_high"),
            func.min(PriceHistory.close).label("period_low"),
            func.count(PriceHistory.id).label("bar_count"),
        )
        .where(PriceHistory.date >= start, PriceHistory.date <= end)
        .group_by(PriceHistory.company_id)
        .subquery("bounds")
    )

    first_p = PriceHistory.__table__.alias("first_p")
    last_p = PriceHistory.__table__.alias("last_p")

    return (
        select(
            Company.id,
            Company.symbol,
            Company.name,
            Company.currency,
            Company.exchange,
            Company.market_cap,
            bounds.c.first_date,
            bounds.c.last_date,
            bounds.c.avg_volume,
            bounds.c.period_high,
            bounds.c.period_low,
            bounds.c.bar_count,
            first_p.c.close.label("first_close"),
            last_p.c.close.label("last_close"),
        )
        .join(bounds, bounds.c.cid == Company.id)
        .join(
            first_p,
            (first_p.c.company_id == Company.id)
            & (first_p.c.date == bounds.c.first_date),
        )
        .join(
            last_p,
            (last_p.c.company_id == Company.id)
            & (last_p.c.date == bounds.c.last_date),
        )
    )


def get_metrics(
    session: Session,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> list[dict[str, Any]]:
    """Every tracked company with its numbers for the window, unranked.

    Benchmarks (index funds) are filtered out - they are not companies and
    would otherwise show up in the rankings.
    """
    rows = session.execute(_metrics_query(start, end)).all()

    out: list[dict[str, Any]] = []
    for r in rows:
        if r.symbol in config.BENCHMARKS:
            continue
        first_close = r.first_close or 0.0
        ytd_pct = ((r.last_close - first_close) / first_close * 100.0
                   if first_close else 0.0)
        out.append({
            "id": r.id,
            "symbol": r.symbol,
            "name": r.name,
            "currency": r.currency or "USD",
            "exchange": r.exchange,
            "market_cap": r.market_cap,
            "first_date": r.first_date,
            "last_date": r.last_date,
            "first_close": round(first_close, 2),
            "last_close": round(r.last_close, 2),
            "change_abs": round(r.last_close - first_close, 2),
            "ytd_pct": round(ytd_pct, 2),
            "period_high": round(r.period_high, 2),
            "period_low": round(r.period_low, 2),
            "avg_volume": int(r.avg_volume or 0),
            "bar_count": r.bar_count,
        })
    return out


_SORT_KEYS = {
    "market_cap": lambda c: c["market_cap"] or -1,
    "price":      lambda c: c["last_close"],
    "ytd":        lambda c: c["ytd_pct"],
    "volume":     lambda c: c["avg_volume"],
}


def get_top_companies(
    session: Session, mode: str = config.DEFAULT_RANK, limit: int = config.TOP_N
) -> list[dict[str, Any]]:
    """The ranked top-N used by the bar chart, in descending order."""
    if mode not in _SORT_KEYS:
        mode = config.DEFAULT_RANK

    companies = get_metrics(session)

    # If market caps never loaded (no crumb), fall back rather than show an
    # arbitrary order.
    if mode == "market_cap" and not any(c["market_cap"] for c in companies):
        mode = "price"

    companies.sort(key=_SORT_KEYS[mode], reverse=True)
    top = companies[:limit]
    for rank, c in enumerate(top, start=1):
        c["rank"] = rank
        c["rank_mode"] = mode
    return top


def get_series(
    session: Session,
    symbol: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> dict[str, Any]:
    """Daily close series for one company - powers the detail line chart.

    Defaults to the year-to-date window; pass an explicit range (as the
    analysis does) to read deeper history.
    """
    start = start or config.year_start()
    end = end or config.today()

    company = session.scalar(select(Company).where(Company.symbol == symbol))
    if company is None:
        return {}

    rows = session.execute(
        select(PriceHistory.date, PriceHistory.close, PriceHistory.volume)
        .where(PriceHistory.company_id == company.id,
               PriceHistory.date >= start, PriceHistory.date <= end)
        .order_by(PriceHistory.date)
    ).all()
    if not rows:
        return {}

    return {
        "symbol": company.symbol,
        "name": company.name,
        "currency": company.currency or "USD",
        "market_cap": company.market_cap,
        "dates":   [r.date.isoformat() for r in rows],
        "closes":  [round(r.close, 2) for r in rows],
        "volumes": [int(r.volume or 0) for r in rows],
    }


def get_multi_series(
    session: Session,
    symbols: list[str],
    max_points: int = 60,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> dict[str, list]:
    """Down-sampled series for several symbols, for the comparison chart."""
    out: dict[str, list] = {}
    for sym in symbols:
        s = get_series(session, sym, start, end)
        if not s:
            continue
        closes, dates = s["closes"], s["dates"]
        step = max(1, len(closes) // max_points)
        out[sym] = {
            "dates":  dates[::step],
            "closes": closes[::step],
        }
    return out


def get_coverage(
    session: Session,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> dict[str, Any]:
    """Headline stats shown in the dashboard header, for the given window."""
    start = start or config.year_start()
    end = end or config.today()
    row = session.execute(
        select(
            func.count(func.distinct(PriceHistory.company_id)),
            func.count(PriceHistory.id),
            func.min(PriceHistory.date),
            func.max(PriceHistory.date),
        )
        .join(Company, Company.id == PriceHistory.company_id)
        .where(PriceHistory.date >= start, PriceHistory.date <= end,
               Company.symbol.notin_(list(config.BENCHMARKS)))
    ).one()
    companies, bars, first, last = row
    return {
        "companies": companies or 0,
        "bars": bars or 0,
        # Fall back to the window bounds when the table is empty. Returning
        # None here made every template that formats a date raise, turning an
        # empty database into a 500 instead of an empty page.
        "first_date": first or start,
        "last_date": last or end,
        "window_start": start,
        "window_end": end,
        "empty": not bars,
    }
