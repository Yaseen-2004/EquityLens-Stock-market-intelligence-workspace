"""Small end-to-end checks. Run with:  python test_app.py"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

from sqlalchemy import func, select

import config
import services
from app import app
from database import SessionLocal, init_db
from models import Company, PriceHistory

failures: list[str] = []

# Deliberately not key-shaped. These checks only care that a value is long
# enough and is not one of the placeholder spellings; a literal that matches a
# provider's real key format would be flagged by secret scanners and is a bad
# pattern to copy.
FAKE_KEY = "not-a-real-credential-" + "0" * 24


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def main() -> int:
    init_db()
    print("Database / services")
    with SessionLocal() as s:
        cov = services.get_coverage(s)
        check("database has price rows", cov["bars"] > 0, f"bars={cov['bars']}")
        check("covers this year only",
              cov["first_date"] >= config.year_start()
              and cov["last_date"] <= config.today(),
              f"{cov['first_date']}..{cov['last_date']}")

        top = services.get_top_companies(s, "market_cap")
        check("returns exactly TOP_N rows", len(top) == config.TOP_N,
              f"got {len(top)}")
        check("ranked descending by market cap",
              all((top[i]["market_cap"] or 0) >= (top[i + 1]["market_cap"] or 0)
                  for i in range(len(top) - 1)))
        check("ranks are 1..N in order",
              [c["rank"] for c in top] == list(range(1, len(top) + 1)))

        ytd = services.get_top_companies(s, "ytd")
        check("ytd mode ranked descending",
              all(ytd[i]["ytd_pct"] >= ytd[i + 1]["ytd_pct"]
                  for i in range(len(ytd) - 1)))

        check("unknown symbol returns empty", services.get_series(s, "NOPE") == {})

    print("HTTP routes")
    c = app.test_client()
    for url in ("/", "/?mode=ytd", "/?mode=price", "/?mode=volume",
                "/api/top", "/api/coverage", "/healthz"):
        check(f"GET {url}", c.get(url).status_code == 200)

    check("GET /company/<known>", c.get("/company/AAPL").status_code == 200)
    check("unknown company is 404", c.get("/company/NOPE").status_code == 404)
    check("bad mode falls back, still 200",
          c.get("/?mode=bogus").status_code == 200)

    payload = c.get("/api/top?limit=3").get_json()
    check("api honours limit", len(payload["results"]) == 3)

    print("Analysis")
    import analysis
    res = analysis.run_analysis(config.HISTORY_YEARS)
    comps = res["companies"]
    check("analysis covers all top-N", len(comps) == config.TOP_N)
    check("analysis window is multi-year", res["window"]["years"] >= 2.5,
          f"years={res['window']['years']}")

    bench = res["benchmark_stats"]
    # The benchmark regressed against itself must give beta 1 and alpha 0.
    # If this drifts, the beta/alpha maths is wrong.
    check("benchmark beta is 1.00", abs(bench["beta"] - 1.0) < 0.01,
          f"beta={bench['beta']}")
    check("benchmark alpha is 0.00", abs(bench["alpha_pct"]) < 0.01,
          f"alpha={bench['alpha_pct']}")

    for sym, m in comps.items():
        # CAGR must reproduce the total return over the measured span.
        implied = ((1 + m["cagr_pct"] / 100) ** (m["trading_days"] / 252) - 1) * 100
        if abs(implied - m["total_return_pct"]) > 1.0:
            check(f"{sym} CAGR reconciles with total", False,
                  f"{implied:.1f} vs {m['total_return_pct']:.1f}")
            break
    else:
        check("CAGR reconciles with total return for all", True)

    check("drawdowns are negative or zero",
          all(m["max_drawdown_pct"] <= 0 for m in comps.values()))
    check("volatility is positive",
          all(m["volatility_pct"] > 0 for m in comps.values()))
    check("correlation diagonal is 1.0",
          all(abs(res["correlation"][s][s] - 1.0) < 1e-9 for s in comps))

    check("GET /analysis", c.get("/analysis").status_code == 200)
    ares = c.get("/api/analysis").get_json()
    check("api/analysis returns rows", len(ares["rows"]) == config.TOP_N)
    check("api/analysis rows sorted by CAGR",
          all(ares["rows"][i]["cagr"] >= ares["rows"][i + 1]["cagr"]
              for i in range(len(ares["rows"]) - 1)))

    print("Assistant")
    import json as _json
    import assistant

    # The SQL tool is the one place a model could do damage, so its guard is
    # tested directly rather than trusted.
    blocked = ["DROP TABLE companies", "DELETE FROM price_history",
               "UPDATE companies SET name='x'", "INSERT INTO companies VALUES (1)",
               "SELECT 1; DROP TABLE companies", "PRAGMA table_info(companies)",
               "ATTACH DATABASE 'x' AS y", ""]
    check("SQL guard blocks every write/DDL form",
          all(not assistant._validate_select(q)[0] for q in blocked))
    allowed = ["SELECT symbol FROM companies",
               "  select * from price_history limit 5",
               "WITH x AS (SELECT 1) SELECT * FROM x"]
    check("SQL guard allows plain SELECTs",
          all(assistant._validate_select(q)[0] for q in allowed))
    check("SQL guard caps unbounded queries",
          assistant._apply_row_cap("SELECT * FROM companies").endswith(
              f"LIMIT {assistant.MAX_ROWS}"))

    tools = {t.name: t for t in assistant._build_tools()}
    import tools_data
    check("every tool is exposed to the model",
          len(tools) == len(tools_data.ALL_TOOLS), f"got {len(tools)}")

    payload = _json.loads(tools["get_rankings"].call({"metric": "ytd", "limit": 3}))
    check("rankings tool returns rows", len(payload["results"]) == 3)
    check("rankings tool rejects a bad metric",
          "error" in _json.loads(tools["get_rankings"].call({"metric": "nope"})))
    check("company tool handles unknown symbols",
          "error" in _json.loads(tools["get_company_detail"].call({"symbol": "NOPE"})))
    sql = _json.loads(tools["run_sql"].call(
        {"query": "SELECT COUNT(*) n FROM price_history"}))
    check("sql tool runs a real query", sql["rows"][0][0] > 0)
    check("sql tool refuses a write",
          "error" in _json.loads(tools["run_sql"].call({"query": "DROP TABLE companies"})))

    st = c.get("/api/assistant/status")
    check("GET /api/assistant/status", st.status_code == 200
          and "ready" in st.get_json())
    check("chat rejects an empty question",
          c.post("/api/assistant/chat", json={}).status_code == 400)
    _prev_engine = os.environ.get("CHAT_ENGINE")
    os.environ["CHAT_ENGINE"] = "offline"   # never spend real API quota here
    try:
        chat = c.post("/api/assistant/chat", json={"question": "hi"})
        check("chat streams server-sent events",
              chat.status_code == 200
              and "text/event-stream" in chat.headers.get("Content-Type", ""))
        check("chat stream always terminates",
              '"end"' in chat.get_data(as_text=True))
    finally:
        if _prev_engine is None:
            os.environ.pop("CHAT_ENGINE", None)
        else:
            os.environ["CHAT_ENGINE"] = _prev_engine

    print("Offline chatbot")
    import chatbot

    check("resolves tickers by name",
          chatbot._find_symbols("how is nvidia doing") == ["NVDA"])
    check("resolves multiple companies",
          set(chatbot._find_symbols("compare tesla and apple")) == {"TSLA", "AAPL"})
    # "market cap" must not resolve to the SPY benchmark via the word "market".
    check("'market cap' is not read as a ticker",
          chatbot._find_symbols("top 5 by market cap") == [])
    check("longest alias wins",
          chatbot._find_symbols("eli lilly results") == ["LLY"])

    cases = {
        "top 5 by market cap": "market capitalisation",
        "which company grew most this year": "return this year",
        "worst performers this year": "return this year",
        "how is NVDA doing": "NVIDIA",
        "compare NVDA and TSLA": "NVDA vs TSLA",
        "what is beta": "Beta measures",
        "which is most volatile": "volatile",
        "which two move together least": "correlat",
        "what data do you have": "daily bars",
        "help": "Rankings",
    }
    bad = [q for q, expect in cases.items()
           if expect.lower() not in chatbot.respond(q)[0].lower()]
    check("every sample question answers sensibly", not bad, f"missed: {bad}")

    # An off-topic question must be declined, whether by the intent matcher or
    # by retrieval - what matters is that it is never answered confidently.
    off_topic = chatbot.respond("what is the weather in Berlin tomorrow")[0].lower()
    check("off-topic questions are declined, not answered",
          "could not match" in off_topic
          or "nothing that closely matches" in off_topic,
          off_topic[:70])
    check("empty question is handled", chatbot.respond("")[0] != "")
    check("benchmark reports its own figures",
          "benchmark" in chatbot.respond("how did the S&P do")[0].lower())

    events = list(chatbot.answer("top 3 by market cap"))
    check("chatbot emits assistant-shaped events",
          events[-1]["type"] == "done"
          and any(e["type"] == "text" for e in events))

    st = c.get("/api/assistant/status").get_json()
    check("chat is always available", st["ready"] is True)
    check("status names the engine",
          st["mode"] in ("gemini", "claude", "offline"), st["mode"])

    # Pin the offline engine for this check. Without it the test would call
    # whichever paid API is configured - spending real quota, and failing when
    # that quota runs out, neither of which is a property of this codebase.
    _prev = os.environ.get("CHAT_ENGINE")
    os.environ["CHAT_ENGINE"] = "offline"
    try:
        live = c.post("/api/assistant/chat",
                      json={"question": "top 3 by market cap"})
        check("chat answers with no API key at all",
              "NVDA" in live.get_data(as_text=True))
    finally:
        if _prev is None:
            os.environ.pop("CHAT_ENGINE", None)
        else:
            os.environ["CHAT_ENGINE"] = _prev

    print("Engine selection")
    import assistant_gemini
    import tools_data

    check("both engines share one tool set",
          [t.name for t in assistant._build_tools()]
          == [f.__name__ for f in tools_data.ALL_TOOLS])
    check("both engines share one system prompt",
          assistant._system_prompt() == tools_data.system_prompt())

    saved = {k: os.environ.get(k) for k in
             ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
              "CHAT_ENGINE")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        import app as app_module
        mode = lambda: app_module._chat_engine()[1]["mode"]

        check("no keys falls back to offline", mode() == "offline")

        os.environ["GEMINI_API_KEY"] = FAKE_KEY
        check("a Gemini key selects gemini", mode() == "gemini")

        os.environ["ANTHROPIC_API_KEY"] = FAKE_KEY
        check("gemini wins when both are set", mode() == "gemini")

        os.environ["CHAT_ENGINE"] = "claude"
        check("CHAT_ENGINE pins claude", mode() == "claude")
        os.environ["CHAT_ENGINE"] = "offline"
        check("CHAT_ENGINE pins offline", mode() == "offline")
        os.environ.pop("CHAT_ENGINE")

        os.environ.pop("GEMINI_API_KEY")
        check("claude key alone selects claude", mode() == "claude")

        # A placeholder must never select an engine that will then 400.
        os.environ["GEMINI_API_KEY"] = "your-key-here"
        check("placeholder Gemini key is not selected", mode() == "claude")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    check("gemini reports unavailable without a key",
          assistant_gemini.availability()["ready"] is False
          or bool(assistant_gemini._api_key()))

    print("Forecasting")
    import json as _json2
    import numpy as _np
    import pandas as _pd
    import forecast

    idx = _pd.bdate_range("2024-01-01", periods=400)
    rng = _np.random.default_rng(0)
    walk = 100 * _np.exp(_np.cumsum(rng.normal(0.0004, 0.015, len(idx))))
    prices = _pd.Series(walk, index=idx)

    feats = forecast.build_features(prices)
    check("features have no lookahead",
          not feats.join(forecast.make_target(prices, 5).rename("y"))
                   .corr()["y"].drop("y").abs().gt(0.99).any())

    tgt = forecast.make_target(prices, 5)
    # The target at t must equal the realised move from t to t+5.
    manual = float(_np.log(prices.iloc[105] / prices.iloc[100]))
    check("target is the realised forward return",
          abs(float(tgt.iloc[100]) - manual) < 1e-9)

    bt = forecast.backtest(prices, horizon=5, test_days=60)
    check("backtest runs", "scores" in bt, bt.get("error", ""))
    if "scores" in bt:
        check("naive is its own benchmark",
              abs(bt["scores"]["naive"]["skill_vs_naive"]) < 1e-9)
        check("every model is scored",
              set(bt["scores"]) == {"naive", "drift", "sma20", "ridge", "gbt"})
        check("errors are positive",
              all(v["mae"] > 0 for v in bt["scores"].values()))

    proj = forecast.project(prices, horizon=5)
    check("projection returns a band",
          proj["lo95"] < proj["lo80"] < proj["hi80"] < proj["hi95"],
          str(proj)[:80])
    check("point estimate sits inside the band",
          proj["lo80"] <= proj["point"] <= proj["hi80"])

    # Non-finite values must never reach the page: bare NaN is invalid JSON
    # and silently blanks every chart.
    saved = forecast.load_results(5)
    if saved is not None:
        raw = _json2.dumps(saved)
        check("saved forecast is valid JSON",
              "NaN" not in raw and "Infinity" not in raw)
        check("cached run has an aggregate", "aggregate" in saved)
        fc = c.get("/forecast")
        check("GET /forecast", fc.status_code == 200)
        check("GET /api/forecast", c.get("/api/forecast").status_code == 200)
    else:
        print("  SKIP  no saved forecast (run: python forecast.py --save)")

    check("json sanitiser strips NaN",
          forecast._json_safe({"a": float("nan"),
                               "b": [1.0, float("inf")]}) == {"a": None,
                                                              "b": [1.0, None]})

    print("Company detail page")
    import json as _js
    import re as _re

    def _series_len(resp):
        """Points in the embedded price series, not bytes of the whole page.

        Comparing page sizes was fragile: shared chrome (the nav rail, the
        toolbar) is a fixed cost on both pages and dilutes the ratio, so a
        redesign could fail this check without the data changing at all.
        """
        body = resp.get_data(as_text=True)
        m = _re.search(r'id="series-data"[^>]*>(.*?)</script>', body, _re.S)
        return len(_js.loads(m.group(1))["closes"]) if m else 0

    ytd = c.get("/company/NVDA")
    y3 = c.get("/company/NVDA?range=3y")
    check("range=3y returns more data than ytd",
          _series_len(y3) > _series_len(ytd) * 2,
          f"{_series_len(y3)} vs {_series_len(ytd)}")
    check("bad range falls back to ytd",
          _series_len(c.get("/company/NVDA?range=bogus")) == _series_len(ytd))
    body = ytd.get_data(as_text=True)
    check("risk section is rendered", "Risk and return over" in body)
    check("drawdown data is embedded", 'id="drawdown-data"' in body)
    check("forecast section is rendered", "Next 5 trading days" in body)
    check("forecast carries its caveat",
          "backtesting" in body and "random walk" in body)

    # A company outside the top 10 has no forecast, but must still render.
    wmt = c.get("/company/WMT")
    check("non-top-10 company still renders", wmt.status_code == 200)
    check("non-top-10 company has risk but no forecast",
          "Risk and return over" in wmt.get_data(as_text=True)
          and "Next 5 trading days" not in wmt.get_data(as_text=True))

    one = analysis.analyse_one("WMT")
    check("analyse_one works outside the top 10",
          one is not None and one["volatility_pct"] > 0)
    check("analyse_one returns a drawdown series",
          len(one["drawdown_series"]["values"]) > 50)
    check("drawdowns are never positive",
          max(one["drawdown_series"]["values"]) <= 0.001)
    check("analyse_one rejects an unknown symbol",
          analysis.analyse_one("NOPE") is None)

    print("Tool invocation (both AI engines)")
    import tools_data as _td

    # tools_data must NOT use `from __future__ import annotations`. With it,
    # annotations become strings and the Google SDK's isinstance() check on
    # tool arguments raises TypeError - quietly, because the SDK hands the
    # error to the model as the tool result. Every tool taking an argument
    # breaks while the no-argument one keeps working, which is a confusing
    # way to find out.
    import inspect as _inspect
    for _fn in _td.ALL_TOOLS:
        hints = _fn.__annotations__
        bad = [k for k, v in hints.items() if isinstance(v, str)]
        if bad:
            check(f"{_fn.__name__} has real (not string) annotations", False,
                  f"string annotations: {bad}")
            break
    else:
        check("every tool has real, non-string annotations", True)

    try:
        from google.genai import _extra_utils as _gx
        cases = [("list_tracked_companies", {}),
                 ("get_rankings", {"metric": "ytd", "limit": 3}),
                 ("get_company_detail", {"symbol": "NVDA"}),
                 ("get_price_history", {"symbol": "NVDA", "max_points": 5}),
                 ("run_sql", {"query": "SELECT COUNT(*) n FROM companies"})]
        broken = []
        for name, args in cases:
            fn = next(f for f in _td.ALL_TOOLS if f.__name__ == name)
            try:
                _gx.invoke_function_from_dict_args(args, fn)
            except Exception as exc:
                broken.append(f"{name}: {type(exc).__name__}")
        check("Google SDK can invoke every tool", not broken, str(broken))
    except ImportError:
        print("  SKIP  google-genai not installed")

    print("Vector store (RAG)")
    import insertdata
    import vectorstore

    check("chunker never splits mid-sentence",
          all(c[0].isupper() or c[0].isdigit() for c in insertdata.chunk_text(
              "First sentence about NVDA here. Second sentence adds detail on "
              "volatility. Third mentions the drawdown ended on 2025-06-25. "
              "Fourth covers beta and alpha. Fifth wraps up.", size=110,
              overlap=55)))
    check("short text is one chunk",
          len(insertdata.chunk_text("A single short sentence.")) == 1)
    check("chunker preserves every sentence",
          "Fifth" in " ".join(insertdata.chunk_text(
              "One. Two. Three. Four. Fifth last.", size=12, overlap=6)))

    if vectorstore.is_ready():
        st = insertdata.stats()
        check("vector store has chunks", st["chunks"] > 20, str(st["chunks"]))
        check("embeddings are 384-dimensional", st["dim"] == 384, str(st["dim"]))
        check("store covers every source type",
              {"company", "risk", "market", "glossary"} <= set(st["by_source"]))

        hits = vectorstore.search("what is beta", k=3)
        check("search returns ranked hits",
              len(hits) > 0 and hits[0]["score"] >= hits[-1]["score"])
        check("the glossary answers a definition query",
              "beta" in hits[0]["title"].lower(), hits[0]["title"])
        check("scores are valid cosine values",
              all(-1.01 <= h["score"] <= 1.01 for h in hits))
        check("a filter restricts the source",
              all(h["source"] == "glossary"
                  for h in vectorstore.search("risk", k=3, source="glossary")))
        check("an empty query returns nothing",
              vectorstore.search("   ", k=3) == [])

        # An unrelated question must not be presented as answered.
        rep = vectorstore.report("what is a stochastic oscillator", k=2)
        check("weak matches are flagged, not passed off as answers",
              "nothing that closely matches" in rep, rep[:60])

        answer, used = chatbot.respond("what happened in april 2025")
        check("chatbot falls back to retrieval",
              used == ["searched the knowledge base"], str(used))
        check("retrieval finds the April 2025 event",
              "April 2025" in answer)

        # A refresh must re-embed only what changed. Embedding is the whole
        # cost of the pipeline, so a daily update that re-embeds all 15,000+
        # chunks turns seconds of work into twenty minutes.
        #
        # This build is deliberately narrative-only, which also guards the
        # pruning step: pruning deletes stored chunks the source no longer
        # produces, and an unscoped prune reads every price chunk as an orphan
        # and deletes all 15,000 of them. Counting them across the build is the
        # cheapest way to catch that, and it is not hypothetical - an earlier
        # unscoped prune destroyed the entire price table.
        def _price_chunks() -> int:
            with SessionLocal() as session:
                return session.scalar(
                    select(func.count()).select_from(insertdata.DocumentChunk)
                    .where(insertdata.DocumentChunk.source == "price")) or 0

        prices_before = _price_chunks()
        again = insertdata.build(include_prices=False)
        check("a partial build leaves out-of-scope chunks alone",
              _price_chunks() == prices_before,
              f"{prices_before} price chunks before, {_price_chunks()} after")
        check("a partial build reports pruning nothing",
              again.get("removed", 0) == 0, str(again.get("removed")))
        check("a repeat build re-embeds nothing",
              again["inserted"] == 0 and again["replaced"] == 0,
              f"{again['inserted']} new, {again['replaced']} updated")
        check("a repeat build still accounts for every chunk",
              again["unchanged"] == again["chunks"],
              f"{again['unchanged']} of {again['chunks']}")

        import json as _j
        payload = _j.loads(tools_data.search_knowledge("which is safest", 2))
        check("search_knowledge tool returns results",
              len(payload.get("results", [])) > 0)

        # --- raw daily price rows ---------------------------------------
        if "price" in st["by_source"]:
            check("every trading day is embedded",
                  st["by_source"]["price"] > 15000,
                  str(st["by_source"]["price"]))

            check("a date question is routed to the price rows",
                  vectorstore.looks_price_specific(
                      "what did NVDA close at on 8 September 2026"))
            check("a conceptual question is not",
                  not vectorstore.looks_price_specific("which is safest"))

            dated = vectorstore.search(
                "What did NVIDIA close at on 8 September 2026?", k=5)
            check("a dated query reaches a daily price row",
                  any(h["source"] == "price" for h in dated),
                  str([h["source"] for h in dated]))

            # The whole risk of adding 15,000 near-identical rows: they drown
            # out the summaries that answer general questions.
            general = vectorstore.search("which company is safest to hold", k=5)
            check("a general question is not swamped by price rows",
                  any(h["source"] != "price" for h in general),
                  str([h["source"] for h in general]))
            check("no single source monopolises the results",
                  max((sum(1 for h in general if h["source"] == s2)
                       for s2 in {h["source"] for h in general}), default=0)
                  <= max(1, 5 // 2) or len(general) < 2)

            check("the glossary still wins a definition query",
                  "beta" in vectorstore.search("what is beta", k=3)[0]
                  ["title"].lower())

            only_prices = vectorstore.search("NVDA", k=3, source="price")
            check("a source filter still reaches prices directly",
                  all(h["source"] == "price" for h in only_prices))

            # Embeddings cannot tell one trading day from another - every
            # daily row reads alike - so a named date must be applied as a
            # filter. Without this, "8 September" returned some other day at
            # a confident-looking 0.65.
            check("a named date parses to the right day",
                  vectorstore.extract_date("NVDA on 8 September 2026")
                  == dt.date(2026, 9, 8))
            check("a bare month is not treated as a day",
                  vectorstore.extract_date("what happened in april 2025") is None)

            exact = vectorstore.search(
                "What did NVIDIA close at on 8 September 2026?", k=3)
            price_hits = [h for h in exact if h["source"] == "price"]
            check("a dated query returns that exact day, not a similar one",
                  price_hits and all(h["doc_id"].endswith("2026-09-08")
                                     for h in price_hits),
                  str([h["doc_id"] for h in price_hits]))
            # Compare against the database rather than a literal. The close
            # was hardcoded here once and the test broke the first time Yahoo
            # restated that bar - it was asserting a snapshot, not a property.
            # What actually has to hold is that the embedded text agrees with
            # the row it was built from.
            with SessionLocal() as _s:
                _close = _s.scalar(
                    select(PriceHistory.close)
                    .join(Company, Company.id == PriceHistory.company_id)
                    .where(Company.symbol == "NVDA",
                           PriceHistory.date == dt.date(2026, 9, 8)))
            _expected = f"{_close:,.2f}"
            check("the retrieved day carries the real closing price",
                  bool(price_hits) and _expected in price_hits[0]["text"],
                  f"expected {_expected}: "
                  + (price_hits[0]["text"][:80] if price_hits else "no hit"))
    else:
        print("  SKIP  vector store empty (run: python insertdata.py)")

    check("search_knowledge is offered to both AI engines",
          "search_knowledge" in [f.__name__ for f in tools_data.ALL_TOOLS])

    print("Chat resilience")
    # An overloaded provider must degrade the answer, not withhold one. This
    # regressed once already: a mid-loop 503 failed the turn, the retry re-ran
    # every tool call across six models, and the panel showed nothing for
    # minutes before erroring.
    import app as _app
    import assistant_gemini as _ag

    _real_answer, _real_avail = _ag.answer, _ag.availability
    _prev = os.environ.get("CHAT_ENGINE")
    os.environ.pop("CHAT_ENGINE", None)
    try:
        def _broken(question, history=None):
            yield {"type": "error", "message": "503 Service Unavailable"}

        _ag.answer = _broken
        _ag.availability = lambda: {"ready": True, "reason": "", "model": "x"}

        raw = c.post("/api/assistant/chat",
                     json={"question": "top 3 by market cap"}).get_data(as_text=True)
        events = [json.loads(l[5:]) for l in raw.strip().splitlines()
                  if l.startswith("data:")]
        kinds = [e["type"] for e in events]
        text = next((e["text"] for e in events if e["type"] == "text"), "")

        check("a failing AI engine still produces an answer", bool(text),
              str(kinds))
        check("NVDA appears in the fallback answer", "NVDA" in text)
        check("the switch to the offline engine is announced",
              "notice" in kinds, str(kinds))
        check("no error is shown once the fallback answered",
              "error" not in kinds, str(kinds))
        check("the stream still terminates", kinds[-1] == "end", str(kinds[-1]))
    finally:
        _ag.answer, _ag.availability = _real_answer, _real_avail
        if _prev is None:
            os.environ.pop("CHAT_ENGINE", None)
        else:
            os.environ["CHAT_ENGINE"] = _prev

    check("gemini bounds a turn by wall-clock time",
          getattr(_ag, "MAX_TOTAL_SECONDS", 0) > 0)
    check("gemini bounds a single request too",
          0 < getattr(_ag, "REQUEST_TIMEOUT_MS", 0)
          <= _ag.MAX_TOTAL_SECONDS * 1000)
    check("the working model is remembered across restarts",
          callable(getattr(_ag, "_load_last_good", None))
          and callable(getattr(_ag, "_save_last_good", None)))
    # Discovering a usable model costs a walk down the chain; without this the
    # first question after every restart pays for it again.
    _ag._save_last_good("gemini-test-model")
    check("the remembered model survives a reload",
          _ag._load_last_good() == "gemini-test-model")
    if _ag._last_good:
        _ag._save_last_good(_ag._last_good)      # restore the real one

    # Every module must import cleanly. A SyntaxError in an engine used to be
    # invisible: engine selection catches exceptions, so a broken engine just
    # silently stopped being chosen.
    import importlib
    broken = []
    for name in ("config", "models", "database", "services", "net", "ingest",
                 "analysis", "forecast", "chatbot", "assistant",
                 "assistant_gemini", "tools_data", "insertdata", "vectorstore",
                 "seed_fallback", "app"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            broken.append(f"{name}: {type(exc).__name__}")
    check("every module imports cleanly", not broken, str(broken))

    # A slow engine must not leave the connection silent: an idle stream is
    # what browsers and proxies drop, which showed up as "Failed to fetch".
    import time as _time
    import chatbot as _cb

    _real_cb = _cb.answer
    _prev2 = os.environ.get("CHAT_ENGINE")
    os.environ["CHAT_ENGINE"] = "offline"
    try:
        def _slow(question, history=None):
            _time.sleep(11)
            yield {"type": "text", "text": "slow answer"}
            yield {"type": "done"}

        _cb.answer = _slow
        raw = c.post("/api/assistant/chat",
                     json={"question": "hi"}).get_data(as_text=True)
        check("a slow turn still sends heartbeats",
              raw.count("keep-alive") >= 1, str(raw.count("keep-alive")))
        check("a slow turn still delivers its answer", "slow answer" in raw)
    finally:
        _cb.answer = _real_cb
        if _prev2 is None:
            os.environ.pop("CHAT_ENGINE", None)
        else:
            os.environ["CHAT_ENGINE"] = _prev2

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
