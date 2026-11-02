"""A chatbot that answers from the database with no API key and no network.

This is the fallback the web panel uses when no Anthropic key is configured, so
the feature always works. It is not a language model: it matches the question
against a set of intents and answers from real queries. Within the narrow
subject this app covers - 20 companies, prices, and risk metrics - that
handles most of what people actually ask.

Where it cannot match an intent it says so and offers what it does know,
rather than guessing.

`assistant.py` is the richer path and takes over automatically when a key is
present. Both emit the same event shape, so the UI does not care which answered.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Iterator

import config
import services
from database import SessionLocal

MODE = "offline"

# --------------------------------------------------------------------------
# Understanding the question
# --------------------------------------------------------------------------
_STOPWORDS = {"the", "a", "an", "of", "for", "is", "was", "in", "on", "to",
              "and", "or", "me", "my", "what", "whats", "how", "show", "tell",
              "please", "about", "with", "it", "its", "do", "does", "did"}

# Words that identify a company beyond its ticker.
_ALIASES = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "alphabet": "GOOGL",
    "google": "GOOGL", "amazon": "AMZN", "meta": "META", "facebook": "META",
    "broadcom": "AVGO", "tesla": "TSLA", "berkshire": "BRK-B",
    "jpmorgan": "JPM", "jp morgan": "JPM", "walmart": "WMT", "lilly": "LLY",
    "eli lilly": "LLY", "visa": "V", "oracle": "ORCL", "mastercard": "MA",
    "netflix": "NFLX", "exxon": "XOM", "costco": "COST",
    "johnson": "JNJ", "johnson & johnson": "JNJ", "amd": "AMD",
    "advanced micro": "AMD", "s&p": "SPY", "sp500": "SPY", "s&p 500": "SPY",
    # Deliberately NOT "market" or "index": "market cap" and "which index"
    # would then resolve to SPY and hijack the question.
}

_DEFINITIONS = {
    "beta": ("Beta measures how much a stock moves with the market. A beta of "
             "1.0 moves in step with the S&P 500; 2.0 swings roughly twice as "
             "hard in both directions; below 1.0 is steadier than the market."),
    "alpha": ("Alpha is the annualised return left over after accounting for "
              "the market exposure a stock's beta implies. Positive alpha "
              "means it did better than its market risk alone would explain."),
    "sharpe": ("The Sharpe ratio is return per unit of total risk: excess "
               "return over the risk-free rate, divided by volatility. Higher "
               "is better. Above 1.0 is generally considered good."),
    "sortino": ("Sortino is like Sharpe but it only counts downside "
                "volatility, on the reasoning that upside swings are not a "
                "problem an investor needs compensating for."),
    "drawdown": ("A drawdown is the fall from a running high to the following "
                 "low. Maximum drawdown is the worst such fall in the period - "
                 "the biggest loss someone holding through it actually felt."),
    "cagr": ("CAGR is the compound annual growth rate: the constant yearly "
             "rate that would turn the starting price into the ending price. "
             "It makes periods of different lengths comparable."),
    "volatility": ("Volatility is the annualised standard deviation of daily "
                   "returns - the usual numeric stand-in for risk. Higher "
                   "means wider swings in both directions."),
    "correlation": ("Correlation measures whether two stocks move together on "
                    "the same days. 1.00 is lockstep, 0.00 is no relationship. "
                    "Low correlation is what makes a mix of holdings safer "
                    "than any one of them."),
    "market cap": ("Market capitalisation is the share price multiplied by the "
                   "number of shares outstanding - the total market value of "
                   "the company."),
    "ytd": ("Year to date: the change from the first trading day of this year "
            f"({config.year_start()}) to the most recent close."),
    "adjusted close": ("An adjusted close is the closing price corrected "
                       "backwards for stock splits and dividends, so a 10-for-1 "
                       "split does not look like a 90% crash."),
    "volume": ("Volume is the number of shares traded in a day. We report the "
               "average over the period."),
}


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9&\s.\-]", " ", text.lower()).strip()


def _find_symbols(question: str) -> list[str]:
    """Tickers named in the question, by symbol or by company name."""
    q = _normalise(question)
    found: list[str] = []

    # Longest aliases first, so "eli lilly" wins over "lilly".
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            sym = _ALIASES[alias]
            if sym not in found:
                found.append(sym)

    known = list(config.TICKERS) + list(config.BENCHMARKS)
    for sym in known:
        pattern = r"\b" + re.escape(sym.lower().replace("-", "[- ]?")) + r"\b"
        if re.search(pattern, q) and sym not in found:
            found.append(sym)
    return found


def _wants(question: str, *words: str) -> bool:
    q = _normalise(question)
    return any(re.search(r"\b" + re.escape(w) + r"\b", q) for w in words)


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def _money(v: float) -> str:
    return f"${v:,.2f}"


def _cap(v: float | None) -> str:
    if not v:
        return "n/a"
    if v >= 1e12:
        return f"${v / 1e12:.2f}T"
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    return f"${v / 1e6:.1f}M"


def _pct(v: float) -> str:
    return f"{v:+.2f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Intents
# --------------------------------------------------------------------------
def _metric_from(question: str) -> str:
    if _wants(question, "cap", "capitalisation", "capitalization", "biggest",
              "largest", "valuable"):
        return "market_cap"
    if _wants(question, "volume", "traded", "liquid"):
        return "volume"
    if _wants(question, "price", "expensive", "costly", "dearest"):
        return "price"
    if _wants(question, "grew", "grown", "growth", "gain", "gained", "return",
              "returns", "performed", "performance", "performer", "performers",
              "performing", "up", "rose", "risen", "ytd", "best"):
        return "ytd"
    # "worst"/"fell" on its own is a question about returns, not about which
    # company is cheapest.
    if _wants(question, "worst", "fell", "fallen", "dropped", "lost", "loser",
              "losers", "down", "decline", "declined"):
        return "ytd"
    return "market_cap"


def _answer_rankings(question: str) -> tuple[str, list[str]]:
    metric = _metric_from(question)
    worst = _wants(question, "worst", "lowest", "bottom", "fell", "fallen",
                   "dropped", "lost", "loser", "losers", "down")
    limit = 5
    m = re.search(r"\b(?:top|bottom|first|last)\s+(\d{1,2})\b", question.lower())
    if m:
        limit = max(1, min(int(m.group(1)), 20))

    with SessionLocal() as s:
        rows = services.get_top_companies(s, metric, config.TOP_N
                                          if not worst else 20)
    if worst:
        rows = sorted(rows, key=lambda r: r["ytd_pct"] if metric == "ytd"
                      else r["last_close"])[:limit]
    else:
        rows = rows[:limit]

    label = {"market_cap": "market capitalisation", "price": "share price",
             "ytd": "return this year", "volume": "average daily volume"}[metric]
    head = f"{'Weakest' if worst else 'Top'} {len(rows)} by {label}:"

    body = _table(
        ["#", "Company", "Last close", "YTD", "Market cap"],
        [[str(i), f"{r['symbol']} {r['name'][:20]}", _money(r["last_close"]),
          _pct(r["ytd_pct"]), _cap(r["market_cap"])]
         for i, r in enumerate(rows, 1)],
    )
    return f"{head}\n\n{body}", ["read rankings"]


def _answer_company(symbols: list[str], question: str) -> tuple[str, list[str]]:
    sym = symbols[0]
    with SessionLocal() as s:
        row = next((m for m in services.get_metrics(s) if m["symbol"] == sym),
                   None)
    if row is None:
        return _answer_benchmark(sym)

    lines = [
        f"**{row['name']} ({sym})** — year to date",
        "",
        f"- Last close: {_money(row['last_close'])} on {row['last_date']}",
        f"- Change this year: {_pct(row['ytd_pct'])} "
        f"(from {_money(row['first_close'])})",
        f"- Range: {_money(row['period_low'])} to {_money(row['period_high'])}",
        f"- Market cap: {_cap(row['market_cap'])}",
        f"- Average volume: {row['avg_volume'] / 1e6:.1f}M shares a day",
    ]
    used = ["read company"]

    if _wants(question, "risk", "risky", "volatile", "volatility", "sharpe",
              "drawdown", "beta", "safe"):
        risk = _risk_row(sym)
        if risk:
            lines += [
                "",
                f"Over the last {config.HISTORY_YEARS} years:",
                f"- Annualised return: {risk['cagr']:+.1f}%",
                f"- Volatility: {risk['vol']:.1f}%",
                f"- Sharpe: {risk['sharpe']:.2f}",
                f"- Worst fall: {risk['max_dd']:.1f}%",
                f"- Beta: {risk['beta']:.2f}",
            ]
            used.append("ran risk analysis")
    return "\n".join(lines), used


def _answer_benchmark(sym: str) -> tuple[str, list[str]]:
    """Benchmarks are excluded from rankings but still have prices."""
    with SessionLocal() as s:
        series = services.get_series(s, sym)
    if not series:
        return (f"I have no data stored for {sym}.", [])

    first, last = series["closes"][0], series["closes"][-1]
    name = config.BENCHMARKS.get(sym, sym)
    lines = [
        f"**{name} ({sym})** — the market benchmark, year to date",
        "",
        f"- Last close: {_money(last)} on {series['dates'][-1]}",
        f"- Change this year: {_pct((last / first - 1) * 100)} "
        f"(from {_money(first)})",
    ]
    used = ["read price history"]

    import analysis
    try:
        payload = analysis.chart_payload(analysis.run_analysis())
        b = payload["benchmark_stats"]
        lines += [
            "",
            f"Over the last {config.HISTORY_YEARS} years:",
            f"- Annualised return: {b['cagr_pct']:+.1f}%",
            f"- Volatility: {b['volatility_pct']:.1f}%",
            f"- Sharpe: {b['sharpe']:.2f}",
            f"- Worst fall: {b['max_drawdown_pct']:.1f}%",
            "",
            "It is tracked to measure the companies against, and is left out "
            "of the rankings because it is an index fund, not a company.",
        ]
        used.append("ran risk analysis")
    except SystemExit:
        pass
    return "\n".join(lines), used


def _risk_row(sym: str) -> dict | None:
    import analysis
    try:
        payload = analysis.chart_payload(analysis.run_analysis())
    except SystemExit:
        return None
    return next((r for r in payload["rows"] if r["symbol"] == sym), None)


def _answer_compare(symbols: list[str]) -> tuple[str, list[str]]:
    a, b = symbols[0], symbols[1]
    with SessionLocal() as s:
        metrics = {m["symbol"]: m for m in services.get_metrics(s)}

    import analysis
    try:
        payload = analysis.chart_payload(analysis.run_analysis())
        risk = {r["symbol"]: r for r in payload["rows"]}
    except SystemExit:
        risk = {}

    rows: list[list[str]] = []

    def add(label: str, fn) -> None:
        try:
            rows.append([label, fn(a), fn(b)])
        except (KeyError, TypeError):
            pass

    add("Last close", lambda x: _money(metrics[x]["last_close"]))
    add("YTD", lambda x: _pct(metrics[x]["ytd_pct"]))
    add("Market cap", lambda x: _cap(metrics[x]["market_cap"]))
    if risk.get(a) and risk.get(b):
        add("Return a year", lambda x: f"{risk[x]['cagr']:+.1f}%")
        add("Volatility", lambda x: f"{risk[x]['vol']:.1f}%")
        add("Sharpe", lambda x: f"{risk[x]['sharpe']:.2f}")
        add("Worst fall", lambda x: f"{risk[x]['max_dd']:.1f}%")
        add("Beta", lambda x: f"{risk[x]['beta']:.2f}")

    if not rows:
        return f"I do not have comparable figures for {a} and {b}.", []

    text = f"**{a} vs {b}**\n\n" + _table(["", a, b], rows)

    # A short verdict, when the risk data supports one.
    if risk.get(a) and risk.get(b):
        ra, rb = risk[a], risk[b]
        better = a if ra["sharpe"] > rb["sharpe"] else b
        other = b if better == a else a
        text += (f"\n\nOn risk-adjusted return {better} is ahead: a Sharpe of "
                 f"{risk[better]['sharpe']:.2f} against {risk[other]['sharpe']:.2f}. "
                 "Sharpe is return per unit of risk, so it rewards steadier "
                 "gains over volatile ones.")
    used = ["read company", "ran risk analysis"] if risk else ["read company"]
    return text, used


def _answer_correlation(symbols: list[str]) -> tuple[str, list[str]]:
    import analysis
    try:
        payload = analysis.chart_payload(analysis.run_analysis())
    except SystemExit:
        return "The analysis window has no data yet.", []

    syms, matrix = payload["corr"]["symbols"], payload["corr"]["matrix"]

    if len(symbols) >= 2 and symbols[0] in syms and symbols[1] in syms:
        i, j = syms.index(symbols[0]), syms.index(symbols[1])
        v = matrix[i][j]
        strength = ("very closely" if v >= 0.6 else "moderately" if v >= 0.35
                    else "loosely" if v >= 0.15 else "barely")
        return (f"{symbols[0]} and {symbols[1]} have a daily-return correlation "
                f"of **{v:.2f}** over the last {config.HISTORY_YEARS} years — "
                f"they move together {strength}. 1.00 would be lockstep, 0.00 "
                f"no relationship.", ["ran risk analysis"])

    pairs = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            pairs.append((matrix[i][j], syms[i], syms[j]))
    pairs.sort()
    low = ", ".join(f"{a}/{b} {v:.2f}" for v, a, b in pairs[:3])
    high = ", ".join(f"{a}/{b} {v:.2f}" for v, a, b in pairs[-3:][::-1])
    avg = sum(p[0] for p in pairs) / len(pairs)
    return (f"Across the top {len(syms)}, the average pair correlates at "
            f"**{avg:.2f}**.\n\n- Least correlated: {low}\n- Most correlated: "
            f"{high}\n\nLow pairs are the ones that actually diversify: they "
            f"tend not to fall on the same days.", ["ran risk analysis"])


def _answer_risk_ranking(question: str) -> tuple[str, list[str]]:
    import analysis
    try:
        payload = analysis.chart_payload(analysis.run_analysis())
    except SystemExit:
        return "The analysis window has no data yet.", []

    rows = payload["rows"]
    if _wants(question, "drawdown", "fell", "fall", "crash", "worst loss"):
        rows = sorted(rows, key=lambda r: r["max_dd"])
        head = f"Deepest falls from a high, last {config.HISTORY_YEARS} years:"
        cell = lambda r: f"{r['max_dd']:.1f}%"
        col = "Worst fall"
    elif _wants(question, "safe", "safest", "steady", "stable", "least risky"):
        rows = sorted(rows, key=lambda r: r["vol"])
        head = "Steadiest of the top 10 (lowest volatility):"
        cell = lambda r: f"{r['vol']:.1f}%"
        col = "Volatility"
    elif _wants(question, "sharpe", "risk adjusted", "best value"):
        rows = sorted(rows, key=lambda r: r["sharpe"], reverse=True)
        head = "Best return per unit of risk (Sharpe ratio):"
        cell = lambda r: f"{r['sharpe']:.2f}"
        col = "Sharpe"
    else:
        rows = sorted(rows, key=lambda r: r["vol"], reverse=True)
        head = "Most volatile of the top 10:"
        cell = lambda r: f"{r['vol']:.1f}%"
        col = "Volatility"

    body = _table([col, "Company", "Return a year", "Sharpe"],
                  [[cell(r), r["symbol"], f"{r['cagr']:+.1f}%",
                    f"{r['sharpe']:.2f}"] for r in rows[:6]])
    return f"{head}\n\n{body}", ["ran risk analysis"]


def _answer_definition(question: str) -> tuple[str, list[str]] | None:
    q = _normalise(question)
    for term in sorted(_DEFINITIONS, key=len, reverse=True):
        if term in q:
            return _DEFINITIONS[term], []
    return None


def _answer_coverage() -> tuple[str, list[str]]:
    with SessionLocal() as s:
        ytd = services.get_coverage(s)
        full = services.get_coverage(s, config.history_start(), config.today())
    return (
        f"The database holds **{full['bars']:,} daily bars** for "
        f"{full['companies']} companies, from {full['first_date']} to "
        f"{full['last_date']}.\n\nThe dashboard shows the year-to-date slice: "
        f"{ytd['bars']:,} bars from {ytd['first_date']}. Prices are adjusted "
        f"closes from Yahoo Finance. {', '.join(config.BENCHMARKS)} is tracked "
        f"as a market benchmark and left out of the rankings.",
        ["listed companies"],
    )


def _answer_list() -> tuple[str, list[str]]:
    with SessionLocal() as s:
        rows = services.get_metrics(s)
    rows.sort(key=lambda r: r["market_cap"] or 0, reverse=True)
    listing = ", ".join(f"{r['symbol']}" for r in rows)
    return (f"I track **{len(rows)} companies**:\n\n{listing}\n\n"
            "Ask about any of them by ticker or by name.", ["listed companies"])


_HELP = (
    "I answer from this project's price database. Things I can do:\n\n"
    "- **Rankings** — \"top 5 by market cap\", \"which grew most this year\", "
    "\"worst performers\"\n"
    "- **One company** — \"how is NVDA doing\", \"Tesla price\"\n"
    "- **Comparisons** — \"compare NVDA and TSLA\"\n"
    "- **Risk** — \"which is most volatile\", \"deepest drawdown\", "
    "\"best Sharpe\"\n"
    "- **Correlation** — \"which two move together least\"\n"
    "- **Definitions** — \"what is beta\", \"explain drawdown\"\n"
    "- **Coverage** — \"what data do you have\"\n\n"
    "I am the offline engine: I read the database directly and do not use a "
    "language model, so I only handle the patterns above. Adding an Anthropic "
    "API key swaps in a much more capable assistant."
)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
def respond(question: str) -> tuple[str, list[str]]:
    """Return (answer_markdown, tool_labels_used)."""
    q = question.strip()
    if not q:
        return "Ask me something about the data.", []

    if _wants(q, "help", "what can you do", "commands", "how do you work"):
        return _HELP, []
    if re.fullmatch(r"(hi|hey|hello|yo|sup|good morning|good evening)[!. ]*",
                    _normalise(q)):
        return ("Hello. I answer questions about the "
                f"{len(config.TICKERS)} companies in this database — rankings, "
                "prices, risk, correlations. Ask \"help\" to see the range.", [])

    symbols = _find_symbols(q)

    # Definitions win only when no company is named, so "what is Tesla's beta"
    # returns Tesla's beta rather than a lecture on beta.
    if not symbols:
        found = _answer_definition(q)
        if found and _wants(q, "what", "explain", "mean", "means", "definition",
                            "define", "whats"):
            return found

    if _wants(q, "correlation", "correlated", "together", "move with",
              "relationship", "diversif"):
        return _answer_correlation(symbols)

    if len(symbols) >= 2 and _wants(q, "compare", "versus", "vs", "against",
                                    "better", "difference", "or"):
        return _answer_compare(symbols)
    if len(symbols) >= 2:
        return _answer_compare(symbols)

    # Open-ended phrasings go to retrieval first: the rankings catch-all below
    # would otherwise swallow "why did everything fall" and answer the wrong
    # question confidently.
    if _wants(q, "tell me about", "what do you know", "summarise", "summarize",
              "overview", "why did", "what caused", "happened", "crash",
              "recovered", "recovery"):
        retrieved = _answer_retrieval(q)
        if retrieved is not None:
            return retrieved

    if _wants(q, "data", "coverage", "how many rows", "database", "range",
              "how far back", "source"):
        return _answer_coverage()

    if _wants(q, "list", "which companies", "what companies", "tickers",
              "symbols", "track"):
        return _answer_list()

    if symbols:
        return _answer_company(symbols, q)

    if _wants(q, "volatile", "volatility", "risky", "riskiest", "safe",
              "safest", "sharpe", "drawdown", "risk"):
        return _answer_risk_ranking(q)

    if _wants(q, "top", "best", "biggest", "largest", "highest", "worst",
              "lowest", "rank", "ranking", "leader", "grew", "growth",
              "performer", "performers", "gained", "fell", "expensive"):
        return _answer_rankings(q)

    found = _answer_definition(q)
    if found:
        return found

    # Nothing matched a known intent. Rather than give up, search the vector
    # store: semantic retrieval is exactly the right tool for the open-ended
    # questions these patterns cannot enumerate in advance.
    retrieved = _answer_retrieval(q)
    if retrieved is not None:
        return retrieved

    return (
        "I could not match that to anything I know how to answer.\n\n" + _HELP,
        [],
    )


def _answer_retrieval(question: str) -> tuple[str, list[str]] | None:
    """Semantic search over the knowledge base, or None if it finds nothing.

    Returning None rather than the least-bad match matters: retrieval that
    always answers is how a system ends up confidently quoting an irrelevant
    passage. The caller falls back to saying it does not know.
    """
    try:
        import vectorstore

        if not vectorstore.is_ready():
            return None
        if not vectorstore.search(question, k=3):
            return None
        return vectorstore.report(question, k=3), ["searched the knowledge base"]
    except Exception:          # retrieval is a bonus - never break the chat
        return None


def answer(question: str, history: list[dict] | None = None) -> Iterator[dict]:
    """Same event shape as assistant.answer, so the UI is identical."""
    try:
        text, used = respond(question)
        for label in used:
            yield {"type": "tool", "name": label, "input": {}}
        yield {"type": "text", "text": text}
        yield {"type": "done"}
    except Exception as exc:                       # never break the panel
        yield {"type": "error", "message": f"Could not answer that: {exc}"}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(respond(" ".join(sys.argv[1:]))[0])
        raise SystemExit(0)

    print("Offline stock chatbot. Ctrl-C to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q:
            print("\n" + respond(q)[0] + "\n")
