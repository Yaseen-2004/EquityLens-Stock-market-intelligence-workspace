"""Flask web layer: HTML pages + a small JSON API the charts consume."""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
from typing import Any

from flask import (Flask, Response, abort, jsonify, render_template,
                   request)

import config
import services
from database import SessionLocal, init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("app")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Tracks the background refresh so the UI can show progress.
_refresh = {"running": False, "message": "idle"}


@app.teardown_appcontext
def _cleanup(exc):  # pragma: no cover
    pass


# The navigation rail lists the tracked companies on every page. That ranking
# only changes when new data lands, so it is cached rather than queried on
# each request.
_nav_cache: dict[str, Any] = {}


@app.context_processor
def inject_nav() -> dict[str, Any]:
    if "companies" not in _nav_cache:
        with SessionLocal() as session:
            top = services.get_top_companies(session, "market_cap",
                                             config.TOP_N)
        _nav_cache["companies"] = [
            {"symbol": c["symbol"], "ytd": c["ytd_pct"]} for c in top
        ]
    return {
        "nav_companies": _nav_cache["companies"],
        "current_symbol": (request.view_args or {}).get("symbol", "").upper()
                          if request.view_args else "",
    }


def _mode_from_request() -> str:
    mode = request.args.get("mode", config.DEFAULT_RANK)
    return mode if mode in config.RANK_MODES else config.DEFAULT_RANK


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.route("/")
def index():
    mode = _mode_from_request()
    with SessionLocal() as session:
        top = services.get_top_companies(session, mode, config.TOP_N)
        coverage = services.get_coverage(session)
        compare = services.get_multi_series(
            session, [c["symbol"] for c in top[:5]]
        )
        everything = services.get_metrics(session)
        # A small close series per row, for the inline sparklines. Rendered as
        # SVG in the template rather than as ten more canvases - it is faster,
        # crisper, and needs no JavaScript.
        sparks = services.get_multi_series(
            session, [c["symbol"] for c in top], max_points=40)

    # An empty database is a normal state - a fresh clone, or a deploy where
    # the data has not been loaded - so it gets a page explaining what to run,
    # not a stack trace.
    if not top:
        return render_template("empty.html", coverage=coverage), 200

    # Headline movers come from the whole tracked universe, not just the ten
    # on screen - the year's best performer is often not a top-ten company.
    by_ytd = sorted(everything, key=lambda m: m["ytd_pct"], reverse=True)
    summary = {
        "best": by_ytd[0] if by_ytd else None,
        "worst": by_ytd[-1] if by_ytd else None,
        "total_cap": sum(c["market_cap"] or 0 for c in top),
        "leader": top[0] if top else None,
    }

    for c in top:
        c["spark"] = sparks.get(c["symbol"], {}).get("closes", [])

    return render_template(
        "index.html",
        top=top,
        coverage=coverage,
        compare=compare,
        mode=mode,
        summary=summary,
        rank_modes=config.RANK_MODES,
        top_n=config.TOP_N,
    )


# The analysis is a few hundred milliseconds of pandas work over a dataset
# that only changes when the ingest runs, so it is cached and invalidated
# there rather than recomputed per request.
_analysis_cache: dict[int, dict] = {}


def get_analysis(years: int) -> dict:
    if years not in _analysis_cache:
        import analysis
        _analysis_cache[years] = analysis.chart_payload(analysis.run_analysis(years))
    return _analysis_cache[years]


@app.route("/analysis")
def analysis_page():
    years = request.args.get("years", config.HISTORY_YEARS, type=int)
    years = max(1, min(years, 20))
    try:
        payload = get_analysis(years)
    except SystemExit as exc:      # raised when the window holds no data
        return render_template("404.html", message=str(exc)), 404
    return render_template("analysis.html", a=payload, years=years)


@app.route("/api/analysis")
def api_analysis():
    years = request.args.get("years", config.HISTORY_YEARS, type=int)
    years = max(1, min(years, 20))
    try:
        return jsonify(get_analysis(years))
    except SystemExit as exc:
        return jsonify({"error": str(exc)}), 404


# The forecast backtest takes minutes, so it is never computed inside a
# request - the page reads the last saved run and offers to regenerate it.
_forecast_job = {"running": False, "message": "idle"}


@app.route("/forecast")
def forecast_page():
    import forecast
    horizon = request.args.get("horizon", forecast.HORIZON, type=int)
    horizon = max(1, min(horizon, 30))
    data = forecast.load_results(horizon)
    return render_template("forecast.html", f=data, horizon=horizon,
                           job=_forecast_job)


@app.route("/api/forecast")
def api_forecast():
    import forecast
    horizon = request.args.get("horizon", forecast.HORIZON, type=int)
    data = forecast.load_results(max(1, min(horizon, 30)))
    if data is None:
        return jsonify({"error": "no saved run",
                        "hint": "python forecast.py --save"}), 404
    return jsonify(data)


@app.route("/api/forecast/run", methods=["POST"])
def api_forecast_run():
    """Regenerate the backtest on a background thread."""
    if _forecast_job["running"]:
        return jsonify(_forecast_job), 409

    horizon = request.args.get("horizon", 5, type=int)
    horizon = max(1, min(horizon, 30))

    def worker():
        _forecast_job.update(running=True,
                             message=f"running {horizon}-day backtest… "
                                     "this takes a couple of minutes")
        try:
            import forecast
            res = forecast.run_all(horizon=horizon)
            if res["results"]:
                forecast.save_results(res)
                _forecast_job["message"] = "backtest complete"
            else:
                _forecast_job["message"] = "no data - run ingest.py first"
        except Exception as exc:
            log.exception("forecast run failed")
            _forecast_job["message"] = f"failed: {exc}"
        finally:
            _forecast_job["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify(_forecast_job), 202


@app.route("/api/forecast/status")
def api_forecast_status():
    return jsonify(_forecast_job)


# How far back the detail chart can look.
RANGES = {"ytd": "This year", "1y": "1 year", "3y": "3 years"}

# Per-company risk numbers are a second of pandas work over a dataset that
# only changes on ingest, so they are cached like the main analysis.
_company_cache: dict[str, dict] = {}


def _range_start(key: str):
    import datetime as _dt
    if key == "1y":
        return config.today() - _dt.timedelta(days=365)
    if key == "3y":
        return config.history_start()
    return config.year_start()


@app.route("/company/<symbol>")
def company(symbol: str):
    symbol = symbol.upper()
    rng = request.args.get("range", "ytd")
    if rng not in RANGES:
        rng = "ytd"

    with SessionLocal() as session:
        series = services.get_series(session, symbol,
                                     _range_start(rng), config.today())
        if not series:
            abort(404, f"No data stored for {symbol!r}")
        metrics = next(
            (m for m in services.get_metrics(session)
             if m["symbol"] == symbol), None
        )

    if symbol not in _company_cache:
        import analysis
        _company_cache[symbol] = analysis.analyse_one(symbol) or {}
    risk = _company_cache[symbol] or None

    # The forecast is only generated for the top 10, so this is often absent.
    projection = None
    try:
        import forecast
        saved = forecast.load_results(forecast.HORIZON)
        if saved and symbol in saved.get("results", {}):
            entry = saved["results"][symbol]
            projection = entry.get("projection")
            if projection and projection.get("error"):
                projection = None
            if projection:
                projection["scores"] = entry["backtest"]["scores"]
                projection["generated_at"] = saved.get("generated_at")
    except Exception:                       # a missing cache must not 500
        projection = None

    return render_template("company.html", series=series, metrics=metrics,
                           risk=risk, projection=projection,
                           rng=rng, ranges=RANGES)


# --------------------------------------------------------------------------
# JSON API
# --------------------------------------------------------------------------
@app.route("/api/top")
def api_top():
    mode = _mode_from_request()
    limit = request.args.get("limit", config.TOP_N, type=int)
    limit = max(1, min(limit, 50))
    with SessionLocal() as session:
        data = services.get_top_companies(session, mode, limit)
    return jsonify({"mode": mode, "limit": limit, "results": data})


@app.route("/api/company/<symbol>")
def api_company(symbol: str):
    with SessionLocal() as session:
        data = services.get_series(session, symbol.upper())
    if not data:
        return jsonify({"error": f"unknown symbol {symbol}"}), 404
    return jsonify(data)


@app.route("/api/coverage")
def api_coverage():
    with SessionLocal() as session:
        return jsonify(services.get_coverage(session))


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Kick off an ingest in the background so the request returns instantly."""
    if _refresh["running"]:
        return jsonify(_refresh), 409

    def worker():
        _refresh.update(running=True, message="fetching from Yahoo Finance...")
        try:
            import ingest
            ingest.run()
            _analysis_cache.clear()   # data changed - drop the cached analysis
            _company_cache.clear()
            _nav_cache.clear()
            _refresh["message"] = "refresh complete"
        except Exception as exc:
            log.exception("refresh failed")
            _refresh["message"] = f"refresh failed: {exc}"
        finally:
            _refresh["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify(_refresh), 202


@app.route("/api/refresh/status")
def api_refresh_status():
    return jsonify(_refresh)


# --------------------------------------------------------------------------
# AI assistant
# --------------------------------------------------------------------------
def _chat_engine() -> tuple[Any, dict]:
    """Pick the chat engine from whatever is configured.

    Order: an explicit CHAT_ENGINE override, then Gemini, then Claude, then the
    offline engine. The offline engine needs nothing at all, so the panel is
    always usable - a missing API key downgrades the answers rather than
    removing the feature.

    Gemini is tried first only because it has a free tier; set CHAT_ENGINE to
    "claude", "gemini" or "offline" in .env to pin one.
    """
    import assistant
    import assistant_gemini
    import chatbot

    engines = {
        "gemini": (assistant_gemini, "gemini"),
        "claude": (assistant, "claude"),
    }

    forced = os.environ.get("CHAT_ENGINE", "").strip().lower()
    if forced == "offline":
        return chatbot, {"ready": True, "mode": "offline",
                         "model": "offline engine",
                         "note": "Pinned by CHAT_ENGINE=offline."}

    order = [forced] if forced in engines else ["gemini", "claude"]
    notes = []
    for name in order:
        module, mode = engines[name]
        status = module.availability()
        if status["ready"]:
            return module, {"ready": True, "mode": mode,
                            "model": status["model"], "note": ""}
        notes.append(f"{mode}: {status['reason']}")

    return chatbot, {
        "ready": True,
        "mode": "offline",
        "model": "offline engine",
        "note": " | ".join(notes),
    }


@app.route("/api/assistant/status")
def api_assistant_status():
    _, status = _chat_engine()
    return jsonify(status)


@app.route("/api/assistant/chat", methods=["POST"])
def api_assistant_chat():
    """Stream the assistant's progress and answer as server-sent events.

    Events are emitted as they happen so the UI can show which tools ran while
    the model is still working, instead of a spinner over a silent request.
    """
    engine, _ = _chat_engine()

    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    history = body.get("history") or []

    if not question:
        return jsonify({"error": "question is required"}), 400
    if len(question) > 4000:
        return jsonify({"error": "question is too long"}), 413

    def run_engine(module, sink: "queue.Queue"):
        """Drive one engine on a worker thread, pushing events to a queue."""
        try:
            for event in module.answer(question, history):
                sink.put(event)
        except Exception as exc:
            log.exception("chat engine failed")
            sink.put({"type": "error", "message": str(exc)})
        finally:
            sink.put(None)          # sentinel: this engine is finished

    def pump(module, sep):
        """Yield SSE frames from an engine, with heartbeats while it thinks.

        A chat turn against an overloaded provider can take half a minute, and
        the engine's generator blocks for all of it. Running it on a thread and
        emitting an SSE comment every few seconds keeps the connection alive -
        an idle stream is what browsers and proxies drop, which surfaced as
        "Failed to fetch" mid-answer.
        """
        sink: queue.Queue = queue.Queue()
        threading.Thread(target=run_engine, args=(module, sink),
                         daemon=True).start()
        while True:
            try:
                event = sink.get(timeout=5)
            except queue.Empty:
                yield ": keep-alive" + sep      # a comment; clients ignore it
                continue
            if event is None:
                return
            yield event

    def stream():
        sep = "\n\n"                              # SSE frames end with a blank line
        answered = False
        failure = ""

        for event in pump(engine, sep):
            if isinstance(event, str):            # heartbeat
                yield event
                continue
            if event["type"] == "text" and event.get("text"):
                answered = True
            if event["type"] == "error":
                # Hold the error back - the offline engine may still answer,
                # and showing both an error and an answer is confusing.
                failure = event["message"]
                continue
            yield "data: " + json.dumps(event) + sep

        # The offline engine reads the same database and needs nothing, so a
        # provider being overloaded should degrade the answer rather than
        # withhold one. It is the backstop for every engine, not only the
        # no-API-key case.
        if not answered:
            import chatbot
            if engine is not chatbot:
                log.info("chat: primary engine failed (%s) - using offline",
                         failure[:120])
                yield "data: " + json.dumps({
                    "type": "notice",
                    "message": "The AI service did not respond, so this is the "
                               "built-in offline engine answering from the "
                               "database.",
                }) + sep
                for event in pump(chatbot, sep):
                    if isinstance(event, str):
                        yield event
                        continue
                    if event["type"] == "text" and event.get("text"):
                        answered = True
                    if event["type"] == "error":
                        failure = failure or event["message"]
                        continue
                    yield "data: " + json.dumps(event) + sep

        if not answered and failure:
            yield "data: " + json.dumps({"type": "error",
                                         "message": failure}) + sep
        yield "data: " + json.dumps({"type": "end"}) + sep

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/healthz")
def healthz():
    """Liveness plus enough detail to diagnose a deployment from outside.

    A hosted app is a black box when something fails: the pages return a
    generic 500 and the logs are somewhere else. This reports what the app can
    actually see - which database it resolved, whether it holds any rows, and
    which optional packages are installed - so a failure can be identified
    with one request.
    """
    import importlib.util
    import sys

    info = {
        "status": "ok",
        "python": sys.version.split()[0],
        "database": config.DATABASE_URL.rsplit("/", 1)[-1],
        "packages": {
            name: importlib.util.find_spec(name) is not None
            for name in ("pandas", "numpy", "sklearn", "google.genai",
                         "anthropic", "sentence_transformers")
        },
    }

    try:
        with SessionLocal() as session:
            coverage = services.get_coverage(session)
        info["data"] = {
            "companies": coverage["companies"],
            "bars": coverage["bars"],
            "first": str(coverage["first_date"]),
            "last": str(coverage["last_date"]),
        }
    except Exception as exc:
        info["status"] = "degraded"
        info["data_error"] = f"{type(exc).__name__}: {exc}"[:200]

    return jsonify(info)


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html", message=str(e)), 404


if __name__ == "__main__":
    init_db()

    # Say once, in the console, whether the assistant is on - the UI simply
    # hides it when it is not, so this is where you find out why.
    try:
        _, st = _chat_engine()
        if st["mode"] == "gemini":
            log.info("Chatbot: Google Gemini (%s)", st["model"])
        elif st["mode"] == "claude":
            log.info("Chatbot: Claude (%s)", st["model"])
        else:
            log.info("Chatbot: offline engine (answers from the database, "
                     "no API key needed)")
            log.info("         Add a GEMINI_API_KEY (free tier) or "
                     "ANTHROPIC_API_KEY for richer answers.")
    except Exception as exc:
        log.warning("Chatbot unavailable: %s", exc)

    # Debug is OFF by default, deliberately. The auto-reloader restarts the
    # server whenever a file changes and drops every in-flight request with
    # it - which surfaces in the chat panel as "Failed to fetch" halfway
    # through an answer, because a chat turn can take tens of seconds.
    # Set FLASK_DEBUG=1 while editing the code.
    debug = os.environ.get("FLASK_DEBUG", "").strip() in ("1", "true", "yes")
    if debug:
        log.warning("debug mode: the reloader will drop in-flight requests")

    # threaded=True so a slow chat turn does not block the pages behind it.
    app.run(debug=debug, port=5000, threaded=True, use_reloader=debug)
