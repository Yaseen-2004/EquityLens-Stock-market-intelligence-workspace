"""Build the vector store: narrow -> chunk -> embed -> insert.

The price table holds numbers. This turns those numbers into readable prose
about each company, splits that prose into chunks, embeds each chunk, and
stores the vectors so questions can be matched by meaning instead of keyword.

    python insertdata.py              # build (or refresh) the vector store
    python insertdata.py --rebuild    # drop everything and start clean
    python insertdata.py --stats      # what is currently stored

A note on what this is good for. For "which company grew most this year", the
SQL query in services.py is strictly better: it is exact, and retrieval can
only lose information a database already has. Vector search earns its place on
the open-ended questions - "which of these is the safest bet", "what happened
in April 2025", "explain the risk here" - where the useful answer is spread
across several facts and no single query names it.

The embedding model runs locally (sentence-transformers, ~90MB, cached after
first use), so building the store costs nothing and needs no API key.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from typing import Any, Iterable

import numpy as np
from sqlalchemy import delete, func, select

import config
import services
from database import SessionLocal, init_db
from models import DocumentChunk

log = logging.getLogger("insertdata")

# Small, fast, and good enough for a corpus this size. 384 dimensions.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking. These documents are short and already topic-scoped, so the target
# is generous and the overlap only needs to stop a sentence being orphaned
# from the sentence that gives it meaning.
CHUNK_CHARS = 600
CHUNK_OVERLAP = 100

_model = None


def get_model():
    """Load the embedding model once per process (it is slow to construct)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        log.info("loading embedding model %s", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# --------------------------------------------------------------------------
# 1. Narrow - pull just the facts worth writing about
# --------------------------------------------------------------------------
def narrow() -> dict[str, Any]:
    """Select the source data. Deliberately narrow: everything gathered here
    becomes text that competes for retrieval space, so raw daily prices are
    left out - they are better served by a SQL query."""
    import analysis

    with SessionLocal() as session:
        metrics = services.get_metrics(session)
        top = services.get_top_companies(session, "market_cap", config.TOP_N)
        coverage = services.get_coverage(session)
        long_coverage = services.get_coverage(session, config.history_start(),
                                              config.today())

    try:
        risk = analysis.chart_payload(analysis.run_analysis())
    except SystemExit:
        risk = None

    forecast = None
    try:
        import forecast as fc
        forecast = fc.load_results(fc.HORIZON)
    except Exception:                       # the store is useful without it
        forecast = None

    return {"metrics": metrics, "top": top, "coverage": coverage,
            "long_coverage": long_coverage, "risk": risk, "forecast": forecast}


def narrow_prices() -> list[dict[str, Any]]:
    """Every stored daily bar, with the previous close attached.

    This is the raw price table. It dominates the corpus by volume - roughly
    15,700 rows against ~80 narrative chunks - so it is gathered separately and
    can be skipped with --no-prices when only the narrative side has changed.
    """
    from models import Company, PriceHistory

    out: list[dict[str, Any]] = []
    with SessionLocal() as session:
        for company in list(session.scalars(select(Company))):
            rows = session.execute(
                select(PriceHistory.date, PriceHistory.open, PriceHistory.high,
                       PriceHistory.low, PriceHistory.close,
                       PriceHistory.volume)
                .where(PriceHistory.company_id == company.id)
                .order_by(PriceHistory.date)
            ).all()
            prev = None
            for r in rows:
                out.append({
                    "symbol": company.symbol, "name": company.name,
                    "date": r.date, "open": r.open, "high": r.high,
                    "low": r.low, "close": r.close, "volume": r.volume,
                    "prev_close": prev,
                })
                prev = r.close
    return out


def build_price_documents(bars: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One document per company per trading day.

    The weekday and month are spelled out because embeddings get far more from
    "Tuesday 8 September 2026" than from "2026-09-08" - the numeric form is
    close to meaningless to the model. The day's move is stated in words too
    ("rose 1.2 percent"), so a question phrased that way has something to match.
    """
    docs: list[dict[str, str]] = []
    for b in bars:
        d = b["date"]
        when = f"{d:%A} {d.day} {d:%B} {d.year}"

        move = ""
        if b["prev_close"]:
            change = (b["close"] / b["prev_close"] - 1) * 100
            verb = "rose" if change >= 0 else "fell"
            move = (f" The price {verb} {abs(change):.2f} percent that day, "
                    f"from a previous close of {_money(b['prev_close'])}.")

        text = (
            f"{b['name']} ({b['symbol']}) share price on {when}. "
            f"It closed at {_money(b['close'])}.{move} "
            f"It opened at {_money(b['open'] or b['close'])}, reached a high of "
            f"{_money(b['high'] or b['close'])} and a low of "
            f"{_money(b['low'] or b['close'])}, with "
            f"{(b['volume'] or 0) / 1e6:.1f} million shares traded."
        )
        docs.append({
            "doc_id": f"price:{b['symbol']}:{d.isoformat()}",
            "source": "price", "symbol": b["symbol"],
            "title": f"{b['symbol']} on {d.isoformat()}", "text": text,
        })
    return docs


# --------------------------------------------------------------------------
# 2. Documents - turn the numbers into prose worth embedding
# --------------------------------------------------------------------------
def _money(v: float) -> str:
    return f"${v:,.2f}"


def _cap(v: float | None) -> str:
    if not v:
        return "not available"
    if v >= 1e12:
        return f"${v / 1e12:.2f} trillion"
    return f"${v / 1e9:.1f} billion"


def build_documents(data: dict[str, Any]) -> list[dict[str, str]]:
    """One document per topic. Each is written as prose, because that is what
    the embedding model was trained on - a wall of comma-separated numbers
    embeds poorly and retrieves worse."""
    docs: list[dict[str, str]] = []
    risk_rows = {r["symbol"]: r for r in (data["risk"] or {}).get("rows", [])}
    forecasts = (data["forecast"] or {}).get("results", {})
    year = config.year_start().year

    # --- per company: this year -------------------------------------------
    for m in data["metrics"]:
        sym, name = m["symbol"], m["name"]
        direction = "risen" if m["ytd_pct"] >= 0 else "fallen"
        text = (
            f"{name} ({sym}) is listed on {m['exchange'] or 'a major exchange'}. "
            f"So far in {year} its share price has {direction} "
            f"{abs(m['ytd_pct']):.2f} percent, from {_money(m['first_close'])} "
            f"on {m['first_date']} to {_money(m['last_close'])} on "
            f"{m['last_date']}. Over that period it traded as low as "
            f"{_money(m['period_low'])} and as high as {_money(m['period_high'])}. "
            f"Its market capitalisation is {_cap(m['market_cap'])}, and an "
            f"average of {m['avg_volume'] / 1e6:.1f} million shares change hands "
            f"each day. "
        )
        if m["ytd_pct"] >= 0:
            text += (f"{sym} is one of the year's gainers, adding "
                     f"{_money(m['change_abs'])} per share.")
        else:
            text += (f"{sym} is down this year, losing "
                     f"{_money(abs(m['change_abs']))} per share.")
        docs.append({"doc_id": f"company:{sym}:ytd", "source": "company",
                     "symbol": sym, "title": f"{name} ({sym}) this year",
                     "text": text})

    # --- per company: risk over the long window ---------------------------
    for sym, r in risk_rows.items():
        recovered = (f"It regained that level on {r['recovered_on']}."
                     if r.get("recovered_on")
                     else "It has not regained that level yet.")
        text = (
            f"{r['name']} ({sym}) risk profile over roughly "
            f"{config.HISTORY_YEARS} years. It returned {r['cagr']:.1f} percent "
            f"a year on average, a total of {r['total']:.1f} percent. Its "
            f"annualised volatility - how widely the daily price swings - was "
            f"{r['vol']:.1f} percent. Its Sharpe ratio, the return earned per "
            f"unit of risk taken, was {r['sharpe']:.2f}, and its Sortino ratio, "
            f"which counts only downside swings, was {r['sortino']:.2f}. "
            f"The worst fall from a previous high was {r['max_dd']:.1f} percent, "
            f"peaking on {r['peak_date']} and bottoming on {r['trough_date']}. "
            f"{recovered} "
        )
        if r.get("beta") is not None:
            behaviour = ("swings harder than the market" if r["beta"] > 1.15
                         else "is steadier than the market" if r["beta"] < 0.85
                         else "moves roughly in step with the market")
            text += (f"Its beta against the S&P 500 is {r['beta']:.2f}, meaning "
                     f"it {behaviour}. Its annualised alpha - the return left "
                     f"over after accounting for that market exposure - was "
                     f"{r['alpha']:.1f} percent. ")
        text += (f"It closed higher on {r['up_days']:.1f} percent of trading "
                 f"days.")
        docs.append({"doc_id": f"company:{sym}:risk", "source": "risk",
                     "symbol": sym, "title": f"{r['name']} ({sym}) risk profile",
                     "text": text})

    # --- per company: forecast --------------------------------------------
    for sym, entry in forecasts.items():
        p = entry.get("projection") or {}
        if not p or p.get("error"):
            continue
        skill = entry["backtest"]["scores"].get(p["model"], {})
        verdict = ("slightly better than assuming no change"
                   if skill.get("skill_vs_naive", -1) > 0
                   else "worse than simply assuming the price does not change")
        text = (
            f"Forecast for {sym} over the next {p['horizon']} trading days, "
            f"made on {p['as_of']} using the {p['model']} model. The point "
            f"estimate is {_money(p['point'])}, a change of {p['change_pct']:.2f} "
            f"percent from the last close of {_money(p['last_close'])}. The 80 "
            f"percent range is {_money(p['lo80'])} to {_money(p['hi80'])}, and "
            f"the 95 percent range is {_money(p['lo95'])} to {_money(p['hi95'])}. "
            f"In walk-forward backtesting this model scored "
            f"{skill.get('skill_vs_naive', 0):+.3f} against a naive random walk, "
            f"which is {verdict}. It predicted the direction of the move "
            f"correctly {skill.get('direction_pct', 50):.1f} percent of the time. "
            f"The width of the range is the trustworthy part of this forecast; "
            f"the single point estimate is not a prediction to act on."
        )
        docs.append({"doc_id": f"company:{sym}:forecast", "source": "forecast",
                     "symbol": sym, "title": f"{sym} forecast", "text": text})

    # --- market-wide ------------------------------------------------------
    cov, lcov = data["coverage"], data["long_coverage"]
    docs.append({
        "doc_id": "market:coverage", "source": "market", "symbol": None,
        "title": "What data is available",
        "text": (
            f"This database tracks {cov['companies']} companies plus the "
            f"{', '.join(config.BENCHMARKS)} exchange traded fund as a market "
            f"benchmark. It holds {lcov['bars']:,} daily price records running "
            f"from {lcov['first_date']} to {lcov['last_date']}. The dashboard "
            f"shows the year to date slice, {cov['bars']:,} records beginning "
            f"{cov['first_date']}. Prices are adjusted closing prices from "
            f"Yahoo Finance, corrected for stock splits and dividends, so they "
            f"will not always match a headline price quoted elsewhere."),
    })

    ranked = sorted(data["metrics"], key=lambda m: m["ytd_pct"], reverse=True)
    winners = ", ".join(f"{m['symbol']} up {m['ytd_pct']:.1f} percent"
                        for m in ranked[:5])
    losers = ", ".join(f"{m['symbol']} down {abs(m['ytd_pct']):.1f} percent"
                       for m in ranked[-5:] if m["ytd_pct"] < 0)
    docs.append({
        "doc_id": "market:performance", "source": "market", "symbol": None,
        "title": f"Best and worst performers of {year}",
        "text": (f"The strongest performers so far in {year} are {winners}. "
                 + (f"The weakest are {losers}. " if losers else "")
                 + f"The largest companies by market value are "
                 + ", ".join(f"{c['symbol']} at {_cap(c['market_cap'])}"
                             for c in data["top"][:5]) + "."),
    })

    if data["risk"]:
        corr = data["risk"]["corr"]
        syms, matrix = corr["symbols"], corr["matrix"]
        pairs = [(matrix[i][j], syms[i], syms[j])
                 for i in range(len(syms)) for j in range(i + 1, len(syms))]
        pairs.sort()
        avg = sum(p[0] for p in pairs) / len(pairs)
        docs.append({
            "doc_id": "market:correlation", "source": "market", "symbol": None,
            "title": "How the companies move together",
            "text": (
                f"Across the largest {len(syms)} companies the average pair of "
                f"daily returns correlates at {avg:.2f}. The least correlated "
                f"pairs, which therefore diversify a holding best, are "
                + ", ".join(f"{a} and {b} at {v:.2f}" for v, a, b in pairs[:4])
                + ". The most correlated pairs, which tend to rise and fall "
                f"together and so diversify least, are "
                + ", ".join(f"{a} and {b} at {v:.2f}"
                            for v, a, b in pairs[-4:][::-1])
                + ". A correlation of 1.00 means two stocks move in lockstep "
                  "every day; 0.00 means no relationship at all."),
        })

        b = data["risk"]["benchmark_stats"]
        port = data["risk"]["portfolio"]
        docs.append({
            "doc_id": "market:benchmark", "source": "market", "symbol": None,
            "title": "Performance against the market",
            "text": (
                f"The {config.BENCHMARKS['SPY']} returned {b['cagr_pct']:.1f} "
                f"percent a year with volatility of {b['volatility_pct']:.1f} "
                f"percent and a Sharpe ratio of {b['sharpe']:.2f}. Its worst "
                f"fall was {b['max_drawdown_pct']:.1f} percent. An equally "
                f"weighted basket of the ten largest companies returned "
                f"{port['cagr_pct']:.1f} percent a year at "
                f"{port['volatility_pct']:.1f} percent volatility, a Sharpe of "
                f"{port['sharpe']:.2f}. Spreading money across the ten produced "
                f"a better risk adjusted result than most of the ten "
                f"individually, which is diversification doing its job."),
        })

    if data["forecast"] and data["forecast"].get("aggregate"):
        agg = data["forecast"]["aggregate"]
        beat = [n for n, s in agg.items()
                if n != "naive" and s["skill_vs_naive"] > 0]
        docs.append({
            "doc_id": "market:forecast-quality", "source": "forecast",
            "symbol": None, "title": "Can these prices be predicted?",
            "text": (
                "Five forecasting approaches were tested by walk-forward "
                "validation, each scored only on data it had never seen. "
                + ("None of them beat the naive random walk - the assumption "
                   "that tomorrow's price equals today's. "
                   if not beat else
                   f"Only {', '.join(beat)} edged past the naive random walk, "
                   "and by a margin well inside the noise. ")
                + "Directional accuracy sat within a few points of fifty "
                  "percent, which is a coin flip. This is the expected result: "
                  "daily closing prices behave close to a random walk, and "
                  "short horizon price prediction is not a solved problem. "
                  "The useful output of a forecast here is the range of "
                  "plausible outcomes, not the single predicted number."),
        })

    if risk_rows:
        by_vol = sorted(risk_rows.values(), key=lambda r: r["vol"])
        by_dd = sorted(risk_rows.values(), key=lambda r: r["max_dd"],
                       reverse=True)
        by_sharpe = sorted(risk_rows.values(), key=lambda r: r["sharpe"],
                           reverse=True)
        docs.append({
            "doc_id": "market:safety", "source": "risk", "symbol": None,
            "title": "Which companies are safest and which are riskiest",
            "text": (
                "Ranked by how steady they have been, the least volatile of "
                "the largest companies are "
                + ", ".join(f"{r['symbol']} at {r['vol']:.1f} percent"
                            for r in by_vol[:4])
                + ". The most volatile, meaning the widest price swings, are "
                + ", ".join(f"{r['symbol']} at {r['vol']:.1f} percent"
                            for r in by_vol[-4:][::-1])
                + ". Measured instead by the worst fall from a previous high, "
                  "the mildest declines were "
                + ", ".join(f"{r['symbol']} at {r['max_dd']:.1f} percent"
                            for r in by_dd[:3])
                + " and the most severe were "
                + ", ".join(f"{r['symbol']} at {r['max_dd']:.1f} percent"
                            for r in by_dd[-3:][::-1])
                + ". The best returns for the risk taken, by Sharpe ratio, "
                  "came from "
                + ", ".join(f"{r['symbol']} at {r['sharpe']:.2f}"
                            for r in by_sharpe[:3])
                + ", and the worst from "
                + ", ".join(f"{r['symbol']} at {r['sharpe']:.2f}"
                            for r in by_sharpe[-3:][::-1])
                + ". A safer holding here means lower volatility, a shallower "
                  "worst-case fall, and a beta closer to or below one; it does "
                  "not mean the price cannot fall."),
        })

    # --- notable dates ----------------------------------------------------
    # Semantic search is poor at dates on its own; giving it a document that
    # actually names them is what makes "what happened in April 2025" findable.
    if risk_rows:
        from collections import Counter

        troughs = Counter(r["trough_date"][:7] for r in risk_rows.values())
        worst_days = Counter(r["worst_day"][:7] for r in risk_rows.values()
                             if r.get("worst_day"))
        best_days = Counter(r["best_day"][:7] for r in risk_rows.values()
                            if r.get("best_day"))

        def _month_name(ym: str) -> str:
            y, m = ym.split("-")
            return f"{dt.date(int(y), int(m), 1):%B %Y}"

        lines = []
        if troughs:
            month, count = troughs.most_common(1)[0]
            if count >= 3:
                lines.append(
                    f"{_month_name(month)} was a market wide low point: "
                    f"{count} of the {len(risk_rows)} largest companies hit "
                    f"their worst price of the whole period that month. When "
                    f"that many bottom together the cause is the market, not "
                    f"any one company.")
        for r in sorted(risk_rows.values(), key=lambda x: x["max_dd"])[:5]:
            lines.append(
                f"{r['symbol']} fell {abs(r['max_dd']):.1f} percent from its "
                f"high of {r['peak_date']} to its low of {r['trough_date']}"
                + (f", recovering on {r['recovered_on']}."
                   if r.get("recovered_on") else ", and has not recovered."))
        for r in sorted(risk_rows.values(),
                        key=lambda x: x.get("worst_day_pct", 0))[:4]:
            if r.get("worst_day"):
                lines.append(
                    f"{r['symbol']} had its single worst day on "
                    f"{r['worst_day']}, dropping {abs(r['worst_day_pct']):.1f} "
                    f"percent, and its best on {r['best_day']}, gaining "
                    f"{r['best_day_pct']:.1f} percent.")
        if best_days:
            month, count = best_days.most_common(1)[0]
            if count >= 3:
                lines.append(
                    f"{_month_name(month)} contained the single best day for "
                    f"{count} of these companies, which usually follows a "
                    f"sharp market wide fall rather than good news at any one "
                    f"of them.")

        docs.append({
            "doc_id": "market:events", "source": "market", "symbol": None,
            "title": "Notable dates, crashes and recoveries",
            "text": " ".join(lines),
        })

    # --- glossary ---------------------------------------------------------
    glossary = {
        "beta": "Beta measures how much a stock moves with the market. A beta "
                "of 1.0 moves in step with the S&P 500. A beta of 2.0 swings "
                "roughly twice as hard in both directions. Below 1.0 is "
                "steadier than the market.",
        "sharpe ratio": "The Sharpe ratio is return per unit of total risk: "
                        "the return above the risk free rate divided by "
                        "volatility. Higher is better, and above 1.0 is "
                        "generally considered good.",
        "volatility": "Volatility is the annualised standard deviation of "
                      "daily returns. It is the usual numerical stand in for "
                      "risk. Higher volatility means wider swings in both "
                      "directions, not only downward.",
        "drawdown": "A drawdown is the fall from a running high to the "
                    "following low. Maximum drawdown is the worst such fall "
                    "in a period - the largest loss somebody holding through "
                    "it actually experienced.",
        "cagr": "CAGR, the compound annual growth rate, is the constant "
                "yearly rate that would turn the starting price into the "
                "ending price. It makes periods of different lengths "
                "comparable in a way a raw total return does not.",
        "alpha": "Alpha is the annualised return left over after accounting "
                 "for the market exposure implied by a stock's beta. Positive "
                 "alpha means it did better than its market risk alone would "
                 "explain.",
        "market capitalisation": "Market capitalisation is the share price "
                                 "multiplied by the number of shares "
                                 "outstanding: the total market value of the "
                                 "company.",
        "adjusted close": "An adjusted closing price is corrected backwards "
                          "for stock splits and dividends, so that a ten for "
                          "one split does not appear in the data as a ninety "
                          "percent crash.",
    }
    for term, meaning in glossary.items():
        docs.append({"doc_id": f"glossary:{term.replace(' ', '-')}",
                     "source": "glossary", "symbol": None,
                     "title": f"What is {term}?", "text": meaning})

    return docs


# --------------------------------------------------------------------------
# 3. Chunk
# --------------------------------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on sentence boundaries, packing up to `size` characters.

    Splitting mid-sentence is the usual way to make retrieval worse: half a
    sentence embeds to something the other half does not mean. The overlap
    carries the trailing sentence into the next chunk so a fact that straddles
    a boundary is still findable.
    """
    text = " ".join(text.split())
    if len(text) <= size:
        return [text]

    import re
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sentence in sentences:
        if current and length + len(sentence) + 1 > size:
            chunks.append(" ".join(current))
            # Carry whole trailing SENTENCES into the next chunk, never a
            # character slice. Slicing by characters produces chunks that
            # begin mid-sentence ("on 2025-06-25. Its beta is...") - the
            # fragment embeds to something neither half actually means.
            carried: list[str] = []
            carried_len = 0
            for prev in reversed(current):
                if carried_len + len(prev) + 1 > overlap and carried:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + 1
            current = carried
            length = carried_len
        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(docs: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc["text"])):
            out.append({**doc, "chunk_index": i, "text": piece})
    return out


# --------------------------------------------------------------------------
# 4. Embed
# --------------------------------------------------------------------------
def embed_texts(texts: list[str], progress_every: int = 0) -> np.ndarray:
    """Embed and L2-normalise, so a dot product is the cosine similarity."""
    model = get_model()
    if progress_every and len(texts) > progress_every:
        parts = []
        for start in range(0, len(texts), progress_every):
            parts.append(model.encode(texts[start:start + progress_every],
                                      batch_size=32, show_progress_bar=False,
                                      convert_to_numpy=True))
            log.info("  embedded %d / %d",
                     min(start + progress_every, len(texts)), len(texts))
        vectors = np.vstack(parts)
    else:
        vectors = model.encode(texts, batch_size=32, show_progress_bar=False,
                               convert_to_numpy=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# --------------------------------------------------------------------------
# 5. Insert
# --------------------------------------------------------------------------
def store(chunks: list[dict[str, Any]], vectors: np.ndarray) -> tuple[int, int]:
    """Write chunks and vectors, replacing any previous version of each.

    Existing keys are read once into a set rather than queried per chunk. With
    the price table included that is 15,000+ lookups, which turned a few
    seconds of writing into several minutes.
    """
    with SessionLocal() as session:
        existing = {
            (doc_id, idx) for doc_id, idx in session.execute(
                select(DocumentChunk.doc_id, DocumentChunk.chunk_index)).all()
        }

        new_rows = []
        updates = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "doc_id": chunk["doc_id"], "chunk_index": chunk["chunk_index"],
                "source": chunk["source"], "symbol": chunk.get("symbol"),
                "title": chunk["title"], "text": chunk["text"],
                "embedding": vector.astype(np.float32).tobytes(),
                "dim": int(vector.shape[0]), "model": EMBED_MODEL,
            }
            if (chunk["doc_id"], chunk["chunk_index"]) in existing:
                updates.append(payload)
            else:
                new_rows.append(payload)

        if new_rows:
            session.bulk_insert_mappings(DocumentChunk, new_rows)
        for payload in updates:
            session.execute(
                DocumentChunk.__table__.update()
                .where(DocumentChunk.doc_id == payload["doc_id"],
                       DocumentChunk.chunk_index == payload["chunk_index"])
                .values(**{k: v for k, v in payload.items()
                           if k not in ("doc_id", "chunk_index")}))
        session.commit()
    return len(new_rows), len(updates)


def build(rebuild: bool = False, include_prices: bool = True) -> dict[str, Any]:
    """Run the whole pipeline."""
    init_db()
    if rebuild:
        with SessionLocal() as session:
            session.execute(delete(DocumentChunk))
            session.commit()
        log.info("cleared the vector store")

    log.info("narrowing source data")
    data = narrow()

    docs = build_documents(data)
    log.info("built %d documents", len(docs))

    if include_prices:
        bars = narrow_prices()
        price_docs = build_price_documents(bars)
        log.info("built %d daily price documents", len(price_docs))
        docs.extend(price_docs)

    chunks = chunk_documents(docs)
    log.info("split into %d chunks", len(chunks))

    # Embedding is the entire cost here. A daily refresh changes one trading
    # day - twenty rows out of nearly sixteen thousand - and re-embedding text
    # that has not moved produces byte-identical vectors. Compare the stored
    # text against the freshly built text and embed only the difference.
    with SessionLocal() as session:
        stored = {
            (doc_id, idx): text
            for doc_id, idx, text in session.execute(
                select(DocumentChunk.doc_id, DocumentChunk.chunk_index,
                       DocumentChunk.text)).all()
        }

    # A model change invalidates every vector, whatever the text says.
    model_changed = False
    if stored:
        with SessionLocal() as session:
            previous_model = session.scalar(select(DocumentChunk.model).limit(1))
        model_changed = previous_model != EMBED_MODEL
        if model_changed:
            log.info("embedding model changed (%s -> %s) - re-embedding all",
                     previous_model, EMBED_MODEL)

    pending = [
        c for c in chunks
        if model_changed
        or stored.get((c["doc_id"], c["chunk_index"])) != c["text"]
    ]
    unchanged = len(chunks) - len(pending)

    if not pending:
        log.info("nothing to embed - all %d chunks are current", len(chunks))
        _prune(chunks)
        return {"documents": len(docs), "chunks": len(chunks),
                "inserted": 0, "replaced": 0, "unchanged": unchanged,
                "dim": 0, "model": EMBED_MODEL}

    log.info("embedding %d changed chunk(s) with %s (%d unchanged, skipped)",
             len(pending), EMBED_MODEL, unchanged)
    vectors = embed_texts([c["text"] for c in pending], progress_every=2000)

    inserted, replaced = store(pending, vectors)
    removed = _prune(chunks)
    log.info("stored: %d new, %d updated, %d unchanged, %d removed",
             inserted, replaced, unchanged, removed)
    return {"documents": len(docs), "chunks": len(chunks),
            "inserted": inserted, "replaced": replaced,
            "unchanged": unchanged, "removed": removed,
            "dim": int(vectors.shape[1]), "model": EMBED_MODEL}


def _prune(chunks: list[dict[str, Any]]) -> int:
    """Delete stored chunks that the source no longer produces.

    Without this, renaming a document or shortening one so it splits into
    fewer pieces would leave orphans behind - still searchable, quietly wrong.

    Scoped to the sources this run actually rebuilt. That restriction is the
    whole safety of the function: `--no-prices` builds only the narrative
    chunks, and an unscoped prune reads the other 15,000 price chunks as
    orphans and deletes every one of them.
    """
    if not chunks:
        return 0

    built_sources = {c["source"] for c in chunks}
    current = {(c["doc_id"], c["chunk_index"]) for c in chunks}

    removed = 0
    with SessionLocal() as session:
        rows = session.execute(
            select(DocumentChunk.id, DocumentChunk.doc_id,
                   DocumentChunk.chunk_index, DocumentChunk.source)
            .where(DocumentChunk.source.in_(built_sources))
        ).all()
        for row in rows:
            if (row.doc_id, row.chunk_index) not in current:
                session.execute(
                    delete(DocumentChunk).where(DocumentChunk.id == row.id))
                removed += 1
        if removed:
            session.commit()
    return removed


def stats() -> dict[str, Any]:
    with SessionLocal() as session:
        total = session.scalar(select(func.count(DocumentChunk.id))) or 0
        by_source = session.execute(
            select(DocumentChunk.source, func.count(DocumentChunk.id))
            .group_by(DocumentChunk.source)
        ).all()
        newest = session.scalar(select(func.max(DocumentChunk.created_at)))
        dim = session.scalar(select(DocumentChunk.dim).limit(1))
    return {"chunks": total, "by_source": dict(by_source),
            "dim": dim, "built": newest}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the vector store")
    ap.add_argument("--rebuild", action="store_true",
                    help="delete every stored chunk first")
    ap.add_argument("--stats", action="store_true",
                    help="show what is stored and exit")
    ap.add_argument("--no-prices", action="store_true",
                    help="skip the raw daily price rows (much faster)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.stats:
        s = stats()
        if not s["chunks"]:
            print("The vector store is empty. Run: python insertdata.py")
            return
        print(f"  {s['chunks']} chunks, {s['dim']} dimensions, built {s['built']}")
        for source, count in sorted(s["by_source"].items()):
            print(f"    {source:<10} {count}")
        return

    result = build(rebuild=args.rebuild, include_prices=not args.no_prices)
    print(f"\n  {result['documents']} documents -> {result['chunks']} chunks")
    print(f"  {result['inserted']} inserted, {result['replaced']} updated, "
          f"{result.get('unchanged', 0)} unchanged (not re-embedded)"
          + (f", {result['removed']} removed" if result.get('removed') else ""))
    print(f"  model: {result['model']}")


if __name__ == "__main__":
    main()
