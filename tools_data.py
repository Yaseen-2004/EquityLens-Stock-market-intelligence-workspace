"""The database-reading tools, as plain functions.

Both AI engines share these, so the answers are grounded in exactly the same
queries whichever provider is configured:

- `assistant.py` (Anthropic) wraps them with the SDK's @beta_tool decorator.
- `assistant_gemini.py` (Google) passes them straight to the SDK, which builds
  the schemas from the type hints and docstrings.

Keep the docstrings in Google style with an `Args:` block - both SDKs read them
to generate the tool schema the model sees, so a vague docstring here becomes a
model that calls the tool wrongly.

Deliberately NO `from __future__ import annotations` in this module. That flag
turns every annotation into a string, and the Google SDK validates arguments
with `isinstance(value, annotation)` when it invokes a tool - against a string
that raises `TypeError: isinstance() arg 2 must be a type`. The failure is
quiet: the SDK reports the error back to the model as the tool result, so the
model says something vague about "a technical issue" and answers without the
data. Every tool that takes an argument breaks; only the no-argument one works.
"""
import datetime as dt
import json
import re
from typing import Any

import config

# A hard ceiling on rows returned to a model, so one broad question cannot push
# the whole price table into the context window.
MAX_ROWS = 200


def _json(payload: Any) -> str:
    return json.dumps(payload, default=str, separators=(",", ":"))


# --------------------------------------------------------------------------
# Read-only SQL guard
# --------------------------------------------------------------------------
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|vacuum|reindex|analyze|begin|commit|rollback)\b",
    re.IGNORECASE,
)


def validate_select(query: str) -> tuple[bool, str]:
    """Allow exactly one read-only SELECT. Rejects everything else."""
    q = query.strip().rstrip(";").strip()
    if not q:
        return False, "empty query"
    # One statement only - a second statement after a semicolon is the classic
    # way to smuggle a write past a naive prefix check.
    if ";" in q:
        return False, "only a single statement is allowed"
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        return False, "only SELECT (or WITH ... SELECT) queries are allowed"
    if _FORBIDDEN.search(q):
        return False, "query contains a non-read-only keyword"
    return True, ""


def apply_row_cap(query: str) -> str:
    """Add a LIMIT when the query has none, so no answer can be unbounded."""
    q = query.strip().rstrip(";").strip()
    if re.search(r"\blimit\s+\d+\s*$", q, re.IGNORECASE):
        return q
    return f"{q} LIMIT {MAX_ROWS}"


# --------------------------------------------------------------------------
# The tools
# --------------------------------------------------------------------------
def list_tracked_companies() -> str:
    """List every company in the database with its ticker symbol and name.

    Call this first when the user names a company but not its ticker, or asks
    what data is available.
    """
    import services
    from database import SessionLocal

    with SessionLocal() as s:
        rows = services.get_metrics(s)
    return _json({
        "companies": [{"symbol": r["symbol"], "name": r["name"],
                       "exchange": r["exchange"]} for r in rows],
        "benchmarks": config.BENCHMARKS,
        "count": len(rows),
    })


def get_rankings(metric: str = "market_cap", limit: int = 10) -> str:
    """Rank companies year-to-date and return the leaders in order.

    Args:
        metric: One of "market_cap", "price", "ytd", "volume".
        limit: How many to return, 1 to 20.
    """
    import services
    from database import SessionLocal

    if metric not in config.RANK_MODES:
        return _json({"error": f"unknown metric {metric!r}",
                      "valid": list(config.RANK_MODES)})
    with SessionLocal() as s:
        rows = services.get_top_companies(s, metric, max(1, min(limit, 20)))
    return _json({"metric": metric, "window": "year to date", "results": rows})


def get_company_detail(symbol: str) -> str:
    """Year-to-date figures for one company: price, return, range, volume.

    Args:
        symbol: Ticker symbol, e.g. "AAPL". Case-insensitive.
    """
    import services
    from database import SessionLocal

    symbol = symbol.upper().strip()
    with SessionLocal() as s:
        row = next((m for m in services.get_metrics(s)
                    if m["symbol"] == symbol), None)
    if row is None:
        return _json({"error": f"{symbol} is not in the database",
                      "hint": "call list_tracked_companies"})
    return _json(row)


def get_risk_analysis(years: int = 3) -> str:
    """Multi-year risk and return metrics for the top 10 companies.

    Returns CAGR, volatility, Sharpe, Sortino, max drawdown, beta, alpha and
    the correlation matrix. Use this for any question about risk, long-run
    performance, drawdowns, or how the companies move together.

    Args:
        years: Length of the analysis window in years, 1 to 10.
    """
    import analysis
    try:
        res = analysis.run_analysis(max(1, min(years, 10)))
    except SystemExit as exc:
        return _json({"error": str(exc)})
    payload = analysis.chart_payload(res)
    payload.pop("growth", None)   # a long series is not useful as text
    return _json(payload)


def get_price_history(symbol: str, start: str = "", end: str = "",
                      max_points: int = 40) -> str:
    """Closing prices for one company over a date range, evenly sampled.

    Args:
        symbol: Ticker symbol, e.g. "NVDA".
        start: ISO date "YYYY-MM-DD". Defaults to 1 January this year.
        end: ISO date "YYYY-MM-DD". Defaults to today.
        max_points: How many samples to return, 5 to 120.
    """
    import services
    from database import SessionLocal

    symbol = symbol.upper().strip()
    try:
        s_date = dt.date.fromisoformat(start) if start else config.year_start()
        e_date = dt.date.fromisoformat(end) if end else config.today()
    except ValueError:
        return _json({"error": "dates must be ISO format YYYY-MM-DD"})

    with SessionLocal() as s:
        series = services.get_series(s, symbol, s_date, e_date)
    if not series:
        return _json({"error": f"no data for {symbol} in {s_date}..{e_date}"})

    closes, dates = series["closes"], series["dates"]
    n = max(5, min(max_points, 120))
    step = max(1, len(closes) // n)
    return _json({
        "symbol": symbol,
        "start": dates[0], "end": dates[-1],
        "first_close": closes[0], "last_close": closes[-1],
        "change_pct": round((closes[-1] / closes[0] - 1) * 100, 2),
        "points": [{"d": d, "c": c}
                   for d, c in zip(dates[::step], closes[::step])],
    })


def run_sql(query: str) -> str:
    """Run a read-only SELECT against the price database for anything the
    other tools do not cover.

    Schema:
      companies(id, symbol, name, currency, exchange, market_cap, updated_at)
      price_history(id, company_id, date, open, high, low, close, volume)
    price_history.company_id references companies.id. `close` is the split- and
    dividend-adjusted close. One row per company per trading day. Dates are
    stored as ISO text and compare correctly as strings.

    Only a single SELECT (or WITH ... SELECT) is permitted; anything else is
    rejected. Results are capped at 200 rows.

    Args:
        query: One SQL SELECT statement.
    """
    ok, problem = validate_select(query)
    if not ok:
        return _json({"error": problem})

    from sqlalchemy import text
    from database import engine

    safe = apply_row_cap(query)
    try:
        with engine.connect() as conn:
            # A read-only transaction: even a cleverly shaped statement cannot
            # write through this connection.
            result = conn.execute(text(safe))
            cols = list(result.keys())
            rows = [list(r) for r in result.fetchmany(MAX_ROWS)]
    except Exception as exc:
        return _json({"error": f"SQL failed: {exc}"})

    return _json({"columns": cols, "rows": rows, "row_count": len(rows),
                  "truncated": len(rows) >= MAX_ROWS})


def search_knowledge(query: str, top_k: int = 4) -> str:
    """Search a knowledge base of written summaries about this market data.

    Use this for open-ended or explanatory questions - "what happened in April
    2025", "which of these is safest", "why did they all fall together" - where
    the answer is spread across several facts rather than sitting in one field.

    For an exact figure (a price, a ranking, a specific date) prefer the other
    tools or run_sql: retrieval returns prose written from the data, and cannot
    be more precise than the query that produced it.

    Args:
        query: What to look for, in plain language.
        top_k: How many passages to return, 1 to 8.
    """
    try:
        import vectorstore
    except ImportError:
        return _json({"error": "vector search is unavailable"})

    if not vectorstore.is_ready():
        return _json({"error": "the knowledge base is empty",
                      "hint": "run: python insertdata.py"})

    hits = vectorstore.search(query, k=max(1, min(top_k, 8)))
    if not hits:
        return _json({"results": [], "note": "nothing matched this query"})
    return _json({
        "results": [{"title": h["title"], "source": h["source"],
                     "symbol": h["symbol"], "relevance": round(h["score"], 3),
                     "text": h["text"]} for h in hits],
        "note": ("Relevance is cosine similarity. Below about 0.35 the passage "
                 "is only loosely related - say so rather than presenting it "
                 "as the answer."),
    })


ALL_TOOLS = [list_tracked_companies, get_rankings, get_company_detail,
             get_risk_analysis, get_price_history, search_knowledge, run_sql]


# --------------------------------------------------------------------------
# Shared system prompt
# --------------------------------------------------------------------------
def system_prompt() -> str:
    """The grounding instructions. Provider-independent."""
    return f"""You are the analyst built into a stock-market dashboard. You answer
questions about the companies in this project's database.

Ground every factual claim in a tool call. You have no reliable memory of these
prices, and the database is the only authority on them. If a tool does not
return what you need, say so plainly rather than estimating.

What the data covers:
- Daily adjusted closing prices, one row per company per trading day.
- The dashboard window is year-to-date: {config.year_start()} to {config.today()}.
- Deeper history is stored for analysis, roughly {config.HISTORY_YEARS} years back.
- {len(config.TICKERS)} companies are tracked; the "top 10" is a live ranking out
  of those, not a fixed list. {", ".join(config.BENCHMARKS)} is a market benchmark,
  not a company, and is excluded from rankings.
- Prices are adjusted for splits and dividends, so they will not always match a
  headline price you might recall.

How to answer:
- Lead with the answer. Keep it short - a few sentences, or a compact Markdown
  table when comparing several companies.
- Give numbers with their units and the period they cover ("+19.7% year to date",
  "47% annualised volatility"). A bare number with no window is misleading.
- When a comparison is uneven, say why: a stock at $1,100 and one at $90 are not
  comparable by price, and a 3-year return is not a year-to-date return.
- Prefer plain language over jargon; when you use a term like beta or drawdown,
  define it in a clause.
- Use run_sql for anything the specific tools do not cover - for example
  aggregates across all companies, or a question about a particular date.
- Use search_knowledge for open-ended or explanatory questions, where the
  answer is spread across several facts rather than sitting in one field.
  Prefer the precise tools whenever an exact figure is what is being asked
  for; retrieval returns prose and cannot beat a query for accuracy.

Boundaries:
- This is an educational project. Do not give investment advice, price targets,
  or buy/sell recommendations. Explaining what the data shows is fine; telling
  someone what to do with their money is not. If asked, say that plainly and
  offer the analysis instead.
- Past performance in a {config.HISTORY_YEARS}-year window is a small sample and
  is not predictive. Say so when a question leans on it.
- Never invent a ticker, a price, or a date."""
