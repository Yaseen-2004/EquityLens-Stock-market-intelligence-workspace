"""Semantic search over the chunks that insertdata.py stored.

    python vectorstore.py "which of these is the safest"
    python vectorstore.py "what happened in april 2025" --k 3

Vectors are L2-normalised at insert time, so cosine similarity is a plain dot
product and the whole search is one matrix multiply. At this corpus size that
is exact and instant - an approximate index (FAISS, Chroma) would add a second
store to keep in sync and buy nothing until there are orders of magnitude more
chunks.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any

import numpy as np
from sqlalchemy import select

from database import SessionLocal
from models import DocumentChunk

log = logging.getLogger("vectorstore")

# Above this cosine score a hit is presented as an answer; below it, as
# "closest thing I have". Calibrated against real queries: correct hits on this
# corpus land around 0.35-0.65, and a question the store knows nothing about
# still pulls its nearest neighbour up to roughly 0.30.
CONFIDENT_SCORE = 0.35

# The matrix is rebuilt when the row count changes, which is enough to catch a
# rebuild of the store without paying for a query on every search.
_cache: dict[str, Any] = {"matrix": None, "rows": None, "count": -1}


def _load() -> tuple[np.ndarray, list[DocumentChunk]]:
    with SessionLocal() as session:
        rows = list(session.scalars(
            select(DocumentChunk).order_by(DocumentChunk.id)))

    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []

    if _cache["count"] == len(rows) and _cache["matrix"] is not None:
        return _cache["matrix"], _cache["rows"]

    matrix = np.vstack([
        np.frombuffer(r.embedding, dtype=np.float32) for r in rows
    ])
    _cache.update(matrix=matrix, rows=rows, count=len(rows))
    log.info("loaded %d vectors of %d dimensions", *matrix.shape)
    return matrix, rows


def is_ready() -> bool:
    """True when there is anything to search."""
    with SessionLocal() as session:
        return bool(session.scalar(select(DocumentChunk.id).limit(1)))


# Daily price rows outnumber the narrative chunks roughly 200 to 1, and they
# all read almost identically ("X share price on <date>. It closed at ..."), so
# a general question pulls a wall of near-duplicate days and buries the summary
# that actually answers it. Two controls keep both useful:
#
#   * `max_per_source` - stops one source monopolising the results.
#   * `prefer_narrative` - only lets price rows through when the question looks
#     like it is asking about a specific day or price.
#
# This is a ranking policy, not a filter on what is stored: every day is in the
# index and reachable, via a date, a `source="price"` search, or the price
# tools.
PRICE_SOURCE = "price"

_DATE_HINT = re.compile(
    r"\b(19|20)\d{2}\b"                                    # a year
    r"|\b\d{4}-\d{2}-\d{2}\b"                              # an ISO date
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"  # a month
    r"|\b(monday|tuesday|wednesday|thursday|friday)\b"
    r"|\b(close[sd]?|closing|opened?|open|high|low|volume|traded|price"
    r"|on that day|that day)\b",
    re.IGNORECASE,
)


def looks_price_specific(query: str) -> bool:
    """True when the question names a date or asks about a day's trading."""
    return bool(_DATE_HINT.search(query))


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))

_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?"
                     rf"\s+(\d{{4}})\b", re.IGNORECASE)
_MDY_RE = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?"
                     rf",?\s+(\d{{4}})\b", re.IGNORECASE)


def extract_date(query: str) -> dt.date | None:
    """The specific calendar day a question names, if it names one.

    Only full dates count. "April 2025" is a month, not a day, and must not
    pin retrieval to an arbitrary date inside it.
    """
    m = _ISO_RE.search(query)
    if m:
        try:
            return dt.date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None

    m = _DMY_RE.search(query)
    if m:
        try:
            return dt.date(int(m[3]), _MONTHS[m[2].lower().rstrip(".")],
                           int(m[1]))
        except (ValueError, KeyError):
            return None

    m = _MDY_RE.search(query)
    if m:
        try:
            return dt.date(int(m[3]), _MONTHS[m[1].lower().rstrip(".")],
                           int(m[2]))
        except (ValueError, KeyError):
            return None
    return None


def extract_symbols(query: str) -> list[str]:
    """Tickers named in the question, matched whole-word."""
    import config

    upper = query.upper()
    known = list(config.TICKERS) + list(config.BENCHMARKS)
    return [sym for sym in known
            if re.search(r"\b" + re.escape(sym.replace("-", "[- ]?")) + r"\b",
                         upper)]


def search(query: str, k: int = 5, source: str | None = None,
           symbol: str | None = None, min_score: float = 0.15,
           max_per_source: int | None = None,
           prefer_narrative: bool = True) -> list[dict[str, Any]]:
    """Return the k chunks closest in meaning to `query`.

    `min_score` exists so a question about something the store knows nothing
    about returns nothing, rather than the least-bad match. Retrieval that
    always returns its top result is how a system ends up confidently citing
    an irrelevant passage.
    """
    matrix, rows = _load()
    if matrix.size == 0 or not query.strip():
        return []

    import insertdata
    q = insertdata.embed_texts([query])[0]

    scores = matrix @ q                      # cosine, both sides normalised

    keep = np.ones(len(rows), dtype=bool)
    if source:
        keep &= np.array([r.source == source for r in rows])
    if symbol:
        sym = symbol.upper()
        keep &= np.array([r.symbol == sym for r in rows])

    # Hold back the daily rows unless the question sounds like it wants one.
    if (prefer_narrative and source != PRICE_SOURCE
            and not looks_price_specific(query)):
        keep &= np.array([r.source != PRICE_SOURCE for r in rows])

    # A named date is a fact, not a similarity. Every daily row embeds to
    # nearly the same point, so without this a question about 8 September
    # returns some other day at a confident-looking 0.65. Narrative rows are
    # left alone - they are not date-scoped.
    wanted_date = extract_date(query)
    if wanted_date is not None:
        stamp = wanted_date.isoformat()
        keep &= np.array([r.source != PRICE_SOURCE
                          or r.doc_id.endswith(stamp) for r in rows])

        # If the question also names companies, drop the other tickers: an
        # exact date across 21 symbols is otherwise 21 equally good hits.
        named = extract_symbols(query)
        if named:
            keep &= np.array([r.source != PRICE_SOURCE or r.symbol in named
                              for r in rows])

    scores = np.where(keep, scores, -1.0)

    # Rank far enough down to still fill k after the per-source cap.
    order = np.argsort(scores)[::-1][:max(1, k) * 12]

    if max_per_source is None:
        # One source may take half the results, and at least one.
        max_per_source = max(1, k // 2)

    picked: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}
    for i in order:
        if scores[i] < min_score:
            break
        src = rows[i].source
        if per_source.get(src, 0) >= max_per_source and len(order) > k:
            continue
        per_source[src] = per_source.get(src, 0) + 1
        picked.append({
            "score": float(scores[i]), "doc_id": rows[i].doc_id,
            "source": src, "symbol": rows[i].symbol,
            "title": rows[i].title, "text": rows[i].text,
        })
        if len(picked) >= k:
            break
    return picked


def report(query: str, k: int = 4) -> str:
    """A readable answer assembled from what retrieval found.

    No language model is involved: this states what was retrieved and how
    closely it matched, then quotes it. That is the honest shape for a
    retrieval-only system - it can find the relevant passage, but it cannot
    write a new sentence that is true.
    """
    hits = search(query, k=k)
    if not hits:
        return ("I could not find anything about that in the knowledge base.\n\n"
                "It covers each company's performance and risk, the forecast "
                "results, how the companies correlate, and a glossary of the "
                "terms used. Try asking about one of those.")

    top = hits[0]["score"]

    # Cosine similarity does not separate "related but not it" from "the
    # answer" at any single cutoff: a correct hit can score 0.25 while a wrong
    # one scores 0.27. So rather than pretend a threshold settles it, say out
    # loud when nothing matched closely and let the reader judge.
    if top < CONFIDENT_SCORE:
        lines = [
            f"I have nothing that closely matches **{query.strip()}** — it may "
            "not be covered by this data at all. The nearest things I hold "
            "are below; treat them as related reading, not an answer.",
            "",
        ]
    else:
        lines = [f"Here is what I found on **{query.strip()}**:", ""]

    for i, hit in enumerate(hits, 1):
        confidence = ("strong match" if hit["score"] >= 0.55
                      else "good match" if hit["score"] >= CONFIDENT_SCORE
                      else "weak match")
        lines.append(f"**{i}. {hit['title']}** — {confidence} "
                     f"({hit['score']:.2f})")
        lines.append(hit["text"])
        lines.append("")

    lines.append("_Retrieved from the knowledge base by meaning, not keyword. "
                 "For exact figures — rankings, a specific price — ask "
                 "directly; those come from a database query and are precise._")
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Search the vector store")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--source", help="company, risk, forecast, market, glossary")
    ap.add_argument("--symbol", help="restrict to one ticker")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    q = " ".join(args.query)

    if not is_ready():
        raise SystemExit("The vector store is empty. Run: python insertdata.py")

    hits = search(q, k=args.k, source=args.source, symbol=args.symbol)
    if not hits:
        print("No match above the relevance threshold.")
        return
    for hit in hits:
        print(f"\n[{hit['score']:.3f}] {hit['title']}  ({hit['source']})")
        print("  " + hit["text"][:400])


if __name__ == "__main__":
    main()
