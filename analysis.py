"""Three-year risk and return analysis of the top 10 companies.

Run it as a script:

    python analysis.py               # full report, 3 years
    python analysis.py --years 5     # a different window
    python analysis.py --json out.json

Everything here reads from the database - no network calls - so the report is
reproducible from a fixed dataset.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

import config
import services
from database import SessionLocal
from models import Company, PriceHistory

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_prices(
    session,
    symbols: list[str],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Wide frame of adjusted closes: one row per date, one column per symbol.

    Rows are aligned across symbols with an inner join, so every return is
    computed from a day on which *all* of them traded.  Comparing series with
    mismatched calendars is the classic way to get a wrong correlation matrix.
    """
    rows = session.execute(
        select(Company.symbol, PriceHistory.date, PriceHistory.close)
        .join(Company, Company.id == PriceHistory.company_id)
        .where(Company.symbol.in_(symbols),
               PriceHistory.date >= start,
               PriceHistory.date <= end)
        .order_by(PriceHistory.date)
    ).all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "date", "close"])
    wide = df.pivot(index="date", columns="symbol", values="close")
    wide.index = pd.to_datetime(wide.index)
    return wide.dropna(how="any").sort_index()


# --------------------------------------------------------------------------
# Per-company metrics
# --------------------------------------------------------------------------
def max_drawdown(prices: pd.Series) -> dict[str, Any]:
    """Deepest peak-to-trough fall, and whether it has been recovered.

    Drawdown is measured against the running maximum, which is what an
    investor actually experiences - not against the first or highest price.
    """
    running_max = prices.cummax()
    drawdown = prices / running_max - 1.0
    trough_idx = drawdown.idxmin()
    depth = float(drawdown.min())

    peak_idx = prices.loc[:trough_idx].idxmax()
    peak_value = float(prices.loc[peak_idx])

    # Recovery: first date after the trough that regains the old peak.
    after = prices.loc[trough_idx:]
    regained = after[after >= peak_value]
    recovered_on = regained.index[0] if len(regained) else None

    return {
        "max_drawdown_pct": round(depth * 100, 2),
        "peak_date": peak_idx.date().isoformat(),
        "trough_date": trough_idx.date().isoformat(),
        "recovered_on": recovered_on.date().isoformat() if recovered_on is not None else None,
        "days_underwater": (
            int((recovered_on - peak_idx).days) if recovered_on is not None
            else int((prices.index[-1] - peak_idx).days)
        ),
        "still_underwater": recovered_on is None,
    }


def analyse_symbol(
    prices: pd.Series,
    returns: pd.Series,
    market_returns: pd.Series | None,
    risk_free: float,
) -> dict[str, Any]:
    """Return, risk and market-sensitivity metrics for one series."""
    n_days = len(returns)
    years = n_days / TRADING_DAYS

    total_return = float(prices.iloc[-1] / prices.iloc[0] - 1.0)
    cagr = float((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1.0)

    vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))

    rf_daily = risk_free / TRADING_DAYS
    excess = returns - rf_daily
    sharpe = float(excess.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if returns.std(ddof=1) else 0.0

    # Sortino: penalises only downside deviation, which is what actually hurts.
    downside = returns[returns < rf_daily]
    downside_dev = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0.0
    sortino = float(excess.mean() * TRADING_DAYS / downside_dev) if downside_dev else 0.0

    out: dict[str, Any] = {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "best_day_pct": round(float(returns.max()) * 100, 2),
        "best_day": returns.idxmax().date().isoformat(),
        "worst_day_pct": round(float(returns.min()) * 100, 2),
        "worst_day": returns.idxmin().date().isoformat(),
        "positive_days_pct": round(float((returns > 0).mean()) * 100, 1),
        "trading_days": n_days,
    }
    out.update(max_drawdown(prices))

    if market_returns is not None and len(market_returns) == len(returns):
        var_m = float(market_returns.var(ddof=1))
        beta = float(returns.cov(market_returns) / var_m) if var_m else float("nan")
        # CAPM alpha, annualised.
        rm = float(market_returns.mean())
        alpha_daily = float(returns.mean()) - (rf_daily + beta * (rm - rf_daily))
        out["beta"] = round(beta, 2)
        out["alpha_pct"] = round(alpha_daily * TRADING_DAYS * 100, 2)
        out["corr_market"] = round(float(returns.corr(market_returns)), 2)

    return out


def calendar_year_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return per calendar year. Partial years at each end are still shown."""
    yearly = prices.resample("YE").last()
    first = prices.iloc[0]
    # Prepend the opening price so the first year's return is measured from it.
    base = pd.concat([first.to_frame().T, yearly])
    return (base.pct_change().dropna(how="all") * 100).round(2)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_analysis(years: int = config.HISTORY_YEARS) -> dict[str, Any]:
    start, end = config.history_start(years), config.today()
    benchmark = next(iter(config.BENCHMARKS), None)

    with SessionLocal() as session:
        top = services.get_top_companies(session, "market_cap", config.TOP_N)
        symbols = [c["symbol"] for c in top]
        names = {c["symbol"]: c["name"] for c in top}
        caps = {c["symbol"]: c["market_cap"] for c in top}

        wanted = symbols + ([benchmark] if benchmark else [])
        prices = load_prices(session, wanted, start, end)

    if prices.empty:
        raise SystemExit(
            "No price data in that window. Run: python ingest.py --years 3"
        )

    returns = prices.pct_change().dropna(how="any")
    market_returns = returns[benchmark] if benchmark in returns else None

    per_symbol: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        per_symbol[sym] = analyse_symbol(
            prices[sym], returns[sym], market_returns, config.RISK_FREE_RATE
        )
        per_symbol[sym]["name"] = names.get(sym, sym)
        per_symbol[sym]["market_cap"] = caps.get(sym)

    bench_stats = None
    if benchmark in prices:
        bench_stats = analyse_symbol(
            prices[benchmark], returns[benchmark], market_returns,
            config.RISK_FREE_RATE
        )
        bench_stats["name"] = config.BENCHMARKS[benchmark]

    # Equal-weight portfolio of the ten, rebalanced daily.
    port_returns = returns[symbols].mean(axis=1)
    port_prices = (1 + port_returns).cumprod() * 100
    port_prices.iloc[0] = 100
    portfolio = analyse_symbol(port_prices, port_returns, market_returns,
                               config.RISK_FREE_RATE)
    portfolio["name"] = f"Equal-weight portfolio of the {len(symbols)}"

    corr = returns[symbols].corr().round(3)
    cal = calendar_year_returns(prices[symbols + ([benchmark] if benchmark in prices else [])])

    return {
        "window": {
            "start": prices.index[0].date().isoformat(),
            "end": prices.index[-1].date().isoformat(),
            "trading_days": len(prices),
            "years": round(len(prices) / TRADING_DAYS, 2),
        },
        "risk_free_rate_pct": round(config.RISK_FREE_RATE * 100, 2),
        "benchmark": benchmark,
        "benchmark_stats": bench_stats,
        "companies": per_symbol,
        "portfolio": portfolio,
        "correlation": corr.to_dict(),
        "correlation_frame": corr,
        "calendar_years": cal,
        "prices": prices,
        "returns": returns,
    }


def analyse_one(symbol: str, years: int = config.HISTORY_YEARS
                ) -> dict[str, Any] | None:
    """Risk metrics for any single company, not just the top 10.

    The main report covers the top 10; the per-company page needs the same
    numbers for whichever company is on screen, so this runs the identical
    calculation for one symbol against the benchmark.
    """
    start, end = config.history_start(years), config.today()
    benchmark = next(iter(config.BENCHMARKS), None)
    symbol = symbol.upper()

    with SessionLocal() as session:
        wanted = [symbol] + ([benchmark] if benchmark else [])
        prices = load_prices(session, wanted, start, end)

    if prices.empty or symbol not in prices:
        return None

    returns = prices.pct_change().dropna(how="any")
    if len(returns) < 60:
        return None

    market = returns[benchmark] if benchmark in returns else None
    stats = analyse_symbol(prices[symbol], returns[symbol], market,
                           config.RISK_FREE_RATE)
    stats["symbol"] = symbol
    stats["years"] = round(len(prices) / TRADING_DAYS, 2)
    stats["window_start"] = prices.index[0].date().isoformat()
    stats["window_end"] = prices.index[-1].date().isoformat()

    if benchmark in prices:
        b = analyse_symbol(prices[benchmark], returns[benchmark], market,
                           config.RISK_FREE_RATE)
        stats["benchmark"] = {"symbol": benchmark, "cagr_pct": b["cagr_pct"],
                              "volatility_pct": b["volatility_pct"],
                              "sharpe": b["sharpe"],
                              "max_drawdown_pct": b["max_drawdown_pct"]}

    # A drawdown curve for the detail chart: how far below its own running
    # high the price sat on each day.
    p = prices[symbol]
    dd = (p / p.cummax() - 1.0) * 100
    step = max(1, len(dd) // 180)
    stats["drawdown_series"] = {
        "dates": [d.date().isoformat() for d in dd.index[::step]],
        "values": [round(float(v), 2) for v in dd.iloc[::step]],
    }
    return stats


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _bar(value: float, lo: float, hi: float, width: int = 18) -> str:
    """Tiny text bar, so the console table is scannable."""
    if hi == lo:
        return " " * width
    filled = int(round((value - lo) / (hi - lo) * width))
    return "#" * max(0, min(width, filled))


def print_report(res: dict[str, Any]) -> None:
    w = res["window"]
    line = "=" * 100

    print(line)
    print(f"  THREE-YEAR ANALYSIS - TOP {config.TOP_N} COMPANIES BY MARKET CAP")
    print(f"  {w['start']} to {w['end']}   |   {w['trading_days']} trading days "
          f"({w['years']} years)   |   risk-free {res['risk_free_rate_pct']}%")
    print(line)

    comps = res["companies"]
    order = sorted(comps, key=lambda s: comps[s]["cagr_pct"], reverse=True)

    # --- returns -----------------------------------------------------------
    print("\nRETURNS (ranked by annualised growth)\n")
    print(f"  {'#':<3}{'SYM':<7}{'COMPANY':<26}{'TOTAL':>10}{'CAGR':>9}"
          f"{'VOL':>9}{'SHARPE':>8}{'SORTINO':>9}")
    print("  " + "-" * 96)
    for i, sym in enumerate(order, 1):
        c = comps[sym]
        print(f"  {i:<3}{sym:<7}{c['name'][:24]:<26}"
              f"{c['total_return_pct']:>9.1f}%{c['cagr_pct']:>8.1f}%"
              f"{c['volatility_pct']:>8.1f}%{c['sharpe']:>8.2f}{c['sortino']:>9.2f}")

    b = res["benchmark_stats"]
    p = res["portfolio"]
    print("  " + "-" * 96)
    print(f"  {'':<3}{'EW10':<7}{p['name'][:24]:<26}"
          f"{p['total_return_pct']:>9.1f}%{p['cagr_pct']:>8.1f}%"
          f"{p['volatility_pct']:>8.1f}%{p['sharpe']:>8.2f}{p['sortino']:>9.2f}")
    if b:
        print(f"  {'':<3}{res['benchmark']:<7}{b['name'][:24]:<26}"
              f"{b['total_return_pct']:>9.1f}%{b['cagr_pct']:>8.1f}%"
              f"{b['volatility_pct']:>8.1f}%{b['sharpe']:>8.2f}{b['sortino']:>9.2f}")

    # --- risk --------------------------------------------------------------
    print("\n\nDRAWDOWN AND MARKET SENSITIVITY\n")
    print(f"  {'SYM':<7}{'MAX DD':>9}  {'PEAK':<12}{'TROUGH':<12}"
          f"{'RECOVERED':<13}{'BETA':>7}{'ALPHA':>9}{'CORR':>7}")
    print("  " + "-" * 96)
    for sym in order:
        c = comps[sym]
        rec = c["recovered_on"] or "not yet"
        print(f"  {sym:<7}{c['max_drawdown_pct']:>8.1f}%  {c['peak_date']:<12}"
              f"{c['trough_date']:<12}{rec:<13}"
              f"{c.get('beta', float('nan')):>7.2f}"
              f"{c.get('alpha_pct', float('nan')):>8.1f}%"
              f"{c.get('corr_market', float('nan')):>7.2f}")
    if b:
        print("  " + "-" * 96)
        print(f"  {res['benchmark']:<7}{b['max_drawdown_pct']:>8.1f}%  "
              f"{b['peak_date']:<12}{b['trough_date']:<12}"
              f"{(b['recovered_on'] or 'not yet'):<13}{1.00:>7.2f}{0.0:>8.1f}%{1.00:>7.2f}")

    # --- daily behaviour ---------------------------------------------------
    print("\n\nDAILY BEHAVIOUR\n")
    print(f"  {'SYM':<7}{'BEST DAY':>10}  {'ON':<13}{'WORST DAY':>11}  "
          f"{'ON':<13}{'UP DAYS':>9}")
    print("  " + "-" * 96)
    for sym in order:
        c = comps[sym]
        print(f"  {sym:<7}{c['best_day_pct']:>9.1f}%  {c['best_day']:<13}"
              f"{c['worst_day_pct']:>10.1f}%  {c['worst_day']:<13}"
              f"{c['positive_days_pct']:>8.1f}%")

    # --- calendar years ----------------------------------------------------
    print("\n\nCALENDAR-YEAR RETURNS (%)\n")
    cal = res["calendar_years"]
    years_hdr = "".join(f"{ix.year:>10}" for ix in cal.index)
    print(f"  {'SYM':<7}{years_hdr}")
    print("  " + "-" * (7 + 10 * len(cal.index)))
    for sym in order + ([res["benchmark"]] if res["benchmark"] in cal.columns else []):
        cells = "".join(
            f"{cal.loc[ix, sym]:>10.1f}" if pd.notna(cal.loc[ix, sym]) else f"{'-':>10}"
            for ix in cal.index
        )
        print(f"  {sym:<7}{cells}")
    print("\n  Note: the first and last years are partial - they start at the")
    print("  beginning of the data window and end at the last trading day.")

    # --- correlation -------------------------------------------------------
    print("\n\nCORRELATION OF DAILY RETURNS\n")
    corr = res["correlation_frame"]
    print("  " + " " * 7 + "".join(f"{s:>7}" for s in corr.columns))
    for sym in corr.index:
        cells = "".join(f"{corr.loc[sym, s]:>7.2f}" for s in corr.columns)
        print(f"  {sym:<7}{cells}")

    # Least- and most-correlated pairs.
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((corr.iloc[i, j], cols[i], cols[j]))
    pairs.sort()
    print("\n  Least correlated:  " + ",  ".join(
        f"{a}/{b} {v:.2f}" for v, a, b in pairs[:3]))
    print("  Most correlated:   " + ",  ".join(
        f"{a}/{b} {v:.2f}" for v, a, b in pairs[-3:][::-1]))
    print(f"  Average pairwise correlation: {np.mean([p[0] for p in pairs]):.2f}")

    print("\n" + line)


def to_json_safe(res: dict[str, Any]) -> dict[str, Any]:
    """Drop the pandas objects so the result can be serialised."""
    out = {k: v for k, v in res.items()
           if k not in ("prices", "returns", "correlation_frame", "calendar_years")}
    cal = res["calendar_years"]
    out["calendar_years"] = {
        str(ix.year): {c: (None if pd.isna(cal.loc[ix, c]) else float(cal.loc[ix, c]))
                       for c in cal.columns}
        for ix in cal.index
    }
    return out


def chart_payload(res: dict[str, Any], max_points: int = 160) -> dict[str, Any]:
    """Everything the /analysis page needs to draw, as plain JSON types."""
    comps = res["companies"]
    order = sorted(comps, key=lambda s: comps[s]["cagr_pct"], reverse=True)
    bench = res["benchmark"]

    rows = [{
        "symbol": sym,
        "name": comps[sym]["name"],
        "cagr": comps[sym]["cagr_pct"],
        "total": comps[sym]["total_return_pct"],
        "vol": comps[sym]["volatility_pct"],
        "sharpe": comps[sym]["sharpe"],
        "sortino": comps[sym]["sortino"],
        "max_dd": comps[sym]["max_drawdown_pct"],
        "beta": comps[sym].get("beta"),
        "alpha": comps[sym].get("alpha_pct"),
        "corr_market": comps[sym].get("corr_market"),
        "up_days": comps[sym]["positive_days_pct"],
        "best_day": comps[sym]["best_day"],
        "best_day_pct": comps[sym]["best_day_pct"],
        "worst_day": comps[sym]["worst_day"],
        "worst_day_pct": comps[sym]["worst_day_pct"],
        "peak_date": comps[sym]["peak_date"],
        "trough_date": comps[sym]["trough_date"],
        "recovered_on": comps[sym]["recovered_on"],
        "market_cap": comps[sym]["market_cap"],
    } for sym in order]

    # Growth of 100: the equal-weight basket against the market.
    returns, symbols = res["returns"], list(comps)
    port = (1 + returns[symbols].mean(axis=1)).cumprod() * 100
    series = {"Equal-weight 10": port}
    if bench in returns:
        series[bench] = (1 + returns[bench]).cumprod() * 100

    idx = port.index
    step = max(1, len(idx) // max_points)
    growth = {
        "dates": [d.date().isoformat() for d in idx[::step]],
        "series": {k: [round(float(v), 2) for v in ser.iloc[::step]]
                   for k, ser in series.items()},
    }

    corr = res["correlation_frame"]
    cols = list(corr.columns)

    return {
        "window": res["window"],
        "risk_free_rate_pct": res["risk_free_rate_pct"],
        "benchmark": bench,
        "benchmark_stats": res["benchmark_stats"],
        "portfolio": res["portfolio"],
        "rows": rows,
        "growth": growth,
        "corr": {"symbols": cols,
                 "matrix": [[float(corr.loc[a, b]) for b in cols] for a in cols]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Risk/return analysis of the top 10")
    ap.add_argument("--years", type=int, default=config.HISTORY_YEARS)
    ap.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    args = ap.parse_args()

    res = run_analysis(args.years)
    print_report(res)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(to_json_safe(res), fh, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
