"""The Google Gemini chat engine.

Same contract as assistant.py (Anthropic) and chatbot.py (offline): an
`availability()` and an `answer()` that yields UI events. The app picks whichever
engine is configured, so the three are interchangeable.

Requires a Gemini API key, which has a free tier:

    https://aistudio.google.com/apikey

Put it in .env as either name - both are read, and the Google SDK itself also
recognises them:

    GEMINI_API_KEY=...
    GOOGLE_API_KEY=...

Tool calling is automatic here: the SDK reads the type hints and docstrings in
tools_data.py, generates the schemas, executes the calls, and loops - so this
file is mostly plumbing and error handling.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterator

import config
import tools_data

log = logging.getLogger("assistant_gemini")

# Default model. NOT the newest one: gemini-3.8-flash is the most contended
# on the free tier and answers 503 "high demand" most of the time, and its
# free quota is 5 requests a day against 20 for 3.6-flash. Starting on a model
# that actually answers is worth more than starting on the best one.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# If the chosen model is overloaded or not enabled for the key, work down this
# list. Free-tier keys routinely see 503 "high demand" on the newest model
# while an older one answers immediately, and which models a key may use
# varies by account - so a single hard-coded id is a fragile default.
MODEL_FALLBACKS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

# Whichever model last answered is tried first next time. Availability on the
# free tier swings minute to minute, so the cheapest possible signal - "this
# one worked a moment ago" - beats a fixed order.
#
# It is remembered on disk as well as in memory. Discovering a working model
# costs one walk down the chain, and without persistence every restart pays
# that again - which is exactly the slow first question after a restart.
_LAST_GOOD_FILE = os.path.join(config.DATA_DIR, "gemini_last_model.json")


def _load_last_good() -> str | None:
    import json
    try:
        with open(_LAST_GOOD_FILE, encoding="utf-8") as fh:
            name = json.load(fh).get("model")
        return name if isinstance(name, str) and name else None
    except (OSError, ValueError):
        return None


def _save_last_good(name: str) -> None:
    import json
    try:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(_LAST_GOOD_FILE, "w", encoding="utf-8") as fh:
            json.dump({"model": name}, fh)
    except OSError:                     # a cache miss is not worth failing on
        pass


_last_good: str | None = _load_last_good()

# Transient failures worth retrying rather than surfacing.
# Worth a short retry: these are transient. 429 is handled separately - on
# the free tier it is a daily quota, which no amount of waiting will clear.
RETRY_STATUSES = (500, 502, 503, 504)
MAX_ATTEMPTS = 2

# A hard ceiling on one question. Every attempt replays the whole tool loop,
# so without this a run of 503s across the fallback chain can spend minutes
# before giving up - with the panel showing nothing the entire time. Better to
# fail quickly and let the offline engine answer.
MAX_TOTAL_SECONDS = 35

# Cap a single HTTP call. The loop deadline alone is not enough: it is only
# checked between attempts, so one slow call could run for 45 seconds on its
# own. Not too tight, though - a tool-calling turn is two or three calls and
# the model has to think on each, so an aggressive cap kills answers that
# would have arrived.
REQUEST_TIMEOUT_MS = 20_000

# No SDK-internal retry. It stacks on top of our own fallback chain - each
# retry replays a call that already timed out - and moving to another model is
# both faster and more likely to work than waiting for an overloaded one.
SDK_RETRY_ATTEMPTS = 1

MAX_OUTPUT_TOKENS = 8000

# The SDK loops tool calls for us; this caps how many it will make for one
# question, so a confused turn cannot spin.
MAX_TOOL_CALLS = 8


def _api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY") or "").strip()


def availability() -> dict[str, Any]:
    """Whether this engine can run, and why not if it cannot."""
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return {"ready": False,
                "reason": "The 'google-genai' package is not installed. "
                          "Run: pip install google-genai"}

    key = _api_key()
    if not key:
        return {"ready": False,
                "reason": "No GEMINI_API_KEY (or GOOGLE_API_KEY) found."}

    from assistant import _looks_like_placeholder
    if _looks_like_placeholder(key):
        return {"ready": False,
                "reason": "The Gemini key still looks like a placeholder. "
                          "Put a real key in .env."}

    return {"ready": True, "reason": "", "model": MODEL}


# Friendly labels for the UI chips, keyed by function name.
TOOL_LABELS = {
    "list_tracked_companies": "listed companies",
    "get_rankings": "read rankings",
    "get_company_detail": "read company",
    "get_risk_analysis": "ran risk analysis",
    "get_price_history": "read price history",
    "search_knowledge": "searched the knowledge base",
    "run_sql": "queried the database",
}


def _history_to_contents(history: list[dict] | None,
                         question: str | None = None) -> list:
    """Turn the UI's chat history into Gemini `contents`.

    Gemini uses "model" where the UI (and Anthropic) say "assistant". A chat
    session's history must start with a user turn and alternate, so leading
    model turns are dropped rather than sent and rejected.
    """
    from google.genai import types

    contents = []
    for turn in (history or [])[-10:]:
        text = (turn.get("content") or "").strip()
        if not text:
            continue
        role = "model" if turn.get("role") == "assistant" else "user"
        if not contents and role == "model":
            continue
        contents.append(types.Content(role=role,
                                      parts=[types.Part(text=text)]))
    if question:
        contents.append(types.Content(role="user",
                                      parts=[types.Part(text=question)]))
    return contents


def answer(question: str, history: list[dict] | None = None) -> Iterator[dict]:
    """Answer one question, yielding {"type": ...} events for the web layer."""
    status = availability()
    if not status["ready"]:
        yield {"type": "error", "message": status["reason"]}
        return

    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors

    global _last_good

    try:
        client = genai.Client(
            api_key=_api_key(),
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(
                    attempts=SDK_RETRY_ATTEMPTS,
                    initial_delay=0.5,
                    max_delay=2.0,
                ),
            ),
        )

        cfg = types.GenerateContentConfig(
            system_instruction=tools_data.system_prompt(),
            tools=tools_data.ALL_TOOLS,     # schemas built from the docstrings
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.2,                # analytic, not creative
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=MAX_TOOL_CALLS,
            ),
        )

        # The SDK explicitly recommends the chat surface for automatic
        # function calling; generate_content warns and is the unsupported path.
        history_contents = _history_to_contents(history, question=None)

        preferred = _last_good or MODEL
        candidates = [preferred] + [m for m in ([MODEL] + MODEL_FALLBACKS)
                                    if m != preferred]
        if os.environ.get("GEMINI_MODEL"):
            # An explicit choice is honoured, but still gets its retries.
            candidates = [MODEL]

        response = None
        last_error: Exception | None = None
        used_model = MODEL
        deadline = time.monotonic() + MAX_TOTAL_SECONDS

        for candidate in candidates:
            if time.monotonic() > deadline:
                log.warning("gemini: deadline reached, giving up on %s",
                            candidate)
                break
            for attempt in range(MAX_ATTEMPTS):
                if time.monotonic() > deadline:
                    break
                try:
                    chat = client.chats.create(
                        model=candidate, config=cfg, history=history_contents,
                    )
                    response = chat.send_message(question)
                    used_model = candidate
                    break
                except genai_errors.ClientError as exc:
                    last_error = exc
                    code = getattr(exc, "code", None)
                    if code == 404:
                        break            # not available to this key; move on
                    if code == 429:
                        # Free-tier 429s are a per-DAY quota, per model
                        # (gemini-3.8-flash allows 5 a day, 3.6-flash 20).
                        # Sleeping will not clear that, so move straight to
                        # the next model instead of stalling the chat.
                        break
                    if code in RETRY_STATUSES:
                        if attempt < MAX_ATTEMPTS - 1:
                            time.sleep(1.5 * (attempt + 1))
                            continue
                        break
                    raise                # a real error - surface it
                except genai_errors.ServerError as exc:
                    last_error = exc
                    if attempt < MAX_ATTEMPTS - 1:
                        time.sleep(2 ** attempt)
                        continue
                    break                # overloaded; try the next model
            if response is not None:
                break

        if response is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError(
                "No Gemini model answered. Tried: " + ", ".join(candidates))

        if used_model != _last_good:
            _last_good = used_model
            _save_last_good(used_model)
            log.info("gemini: %s answered - trying it first from now on",
                     used_model)

        # Report which tools ran, so the answer is visibly grounded.
        #
        # The chat surface does NOT populate automatic_function_calling_history
        # - only models.generate_content does - so the calls have to be read
        # out of the session's own history. Only the entries added by this turn
        # are ours; everything before that is the conversation we seeded.
        seen: set[str] = set()
        turn_contents = list(response.automatic_function_calling_history or [])
        if not turn_contents:
            try:
                full = chat.get_history()
                turn_contents = full[len(history_contents):]
            except Exception:                 # history is a nicety, not the answer
                turn_contents = []

        for content in turn_contents:
            for part in (getattr(content, "parts", None) or []):
                call = getattr(part, "function_call", None)
                if call and call.name and call.name not in seen:
                    seen.add(call.name)
                    yield {"type": "tool",
                           "name": TOOL_LABELS.get(call.name, call.name),
                           "input": dict(call.args or {})}

        text = (response.text or "").strip()
        if not text:
            # A blocked or empty candidate is not an exception in this SDK.
            reason = ""
            if response.candidates:
                reason = str(getattr(response.candidates[0],
                                     "finish_reason", "") or "")
            yield {"type": "error",
                   "message": ("Gemini returned no text"
                               + (f" ({reason})" if reason else "")
                               + ". Try rephrasing the question.")}
            return

        yield {"type": "text", "text": text}
        yield {"type": "done", "model": used_model}

    except genai_errors.ClientError as exc:
        code = getattr(exc, "code", None)
        if code == 429:
            yield {"type": "error",
                   "message": "Gemini quota exhausted on every model tried. "
                              "The free tier allows only a few requests per "
                              "minute on the newest models - wait a minute, "
                              "or set GEMINI_MODEL in .env to an older model "
                              "with more headroom."}
        elif code in (401, 403) or "api key not valid" in str(exc).lower():
            yield {"type": "error",
                   "message": "Gemini rejected the API key. Check "
                              "GEMINI_API_KEY in .env."}
        elif code == 404:
            yield {"type": "error",
                   "message": f"Model {MODEL!r} was not found for this key. "
                              "Set GEMINI_MODEL in .env to one your key can use."}
        else:
            yield {"type": "error", "message": f"Gemini error: {exc}"}
    except genai_errors.ServerError as exc:
        yield {"type": "error",
               "message": f"Gemini server error, try again shortly: {exc}"}
    except Exception as exc:                       # never 500 the chat endpoint
        log.exception("gemini assistant failed")
        yield {"type": "error", "message": f"Unexpected error: {exc}"}


def ask(question: str, history: list[dict] | None = None) -> str:
    """Blocking convenience wrapper - returns just the final answer text."""
    text, error = "", ""
    for event in answer(question, history):
        if event["type"] == "text":
            text = event["text"]
        elif event["type"] == "error":
            error = event["message"]
    if error and not text:
        raise RuntimeError(error)
    return text


def list_models() -> list[str]:
    """Model names this key can use - handy when the default 404s."""
    from google import genai

    client = genai.Client(api_key=_api_key())
    out = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            out.append(m.name or "")
    return [n.replace("models/", "") for n in out if n]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if len(sys.argv) > 1 and sys.argv[1] == "--models":
        st = availability()
        if not st["ready"]:
            print(st["reason"])
            raise SystemExit(1)
        for name in list_models():
            print(" ", name)
        raise SystemExit(0)

    st = availability()
    if not st["ready"]:
        print(f"Gemini engine unavailable: {st['reason']}")
        raise SystemExit(1)

    print(f"Gemini assistant ({MODEL}). Ctrl-C to quit.\n")
    convo: list[dict] = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        reply = ""
        for ev in answer(q, convo):
            if ev["type"] == "tool":
                print(f"  [{ev['name']}]")
            elif ev["type"] == "text":
                reply = ev["text"]
                print(f"\n{reply}\n")
            elif ev["type"] == "error":
                print(f"  ! {ev['message']}\n")
        convo.append({"role": "user", "content": q})
        convo.append({"role": "assistant", "content": reply})
