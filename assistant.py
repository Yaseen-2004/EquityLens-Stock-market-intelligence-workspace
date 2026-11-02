"""An AI assistant that answers questions about the stock database.

It is grounded: the model does not answer from memory, it calls tools that read
this project's SQLite database and reports what they return. That matters for a
finance app - a plausible-sounding invented number is worse than no answer.

Requires an Anthropic API key.  Easiest: copy .env.example to .env and put the
key in it (config.py loads .env at import).  An exported ANTHROPIC_API_KEY
environment variable also works and takes precedence.

Note on Windows: `setx` sets the variable permanently but NOT in the window you
ran it in - open a new terminal, or use the .env file instead.

Everything degrades gracefully when the key is absent - the web UI says so
rather than erroring.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Iterator

import config
import tools_data

log = logging.getLogger("assistant")

MODEL = "claude-opus-5"
MAX_TOKENS = 8000

# Chat over a small, well-defined dataset does not need deep reasoning on every
# turn. Raise to "high" or "xhigh" if you start asking multi-step analytical
# questions; that trades latency and cost for thoroughness.
EFFORT = "medium"

# A hard ceiling on rows returned to the model, so one broad question cannot
# push the whole price table into the context window.
MAX_ROWS = 200


class AssistantUnavailable(RuntimeError):
    """Raised when the assistant cannot run (missing key or SDK)."""


def availability() -> dict[str, Any]:
    """Whether the assistant can run, and why not if it cannot."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return {"ready": False,
                "reason": "The 'anthropic' package is not installed. "
                          "Run: pip install anthropic"}

    key = (os.environ.get("ANTHROPIC_API_KEY")
           or os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip()

    if not key:
        return {
            "ready": False,
            "configured": False,
            "reason": (
                "The AI assistant is turned off because no API key is set. "
                "Everything else in the dashboard works without one."
            ),
            "help_url": "https://console.anthropic.com/settings/keys",
            "env_file": os.path.join(config.BASE_DIR, ".env"),
        }

    # A copied .env.example is a real failure mode: the placeholder is a
    # non-empty string, so without this check it sails through to a 401 from
    # the API instead of being caught here.
    if _looks_like_placeholder(key):
        return {
            "ready": False,
            "configured": False,
            "reason": (
                "The API key still looks like the example placeholder, so the "
                "assistant is turned off. Replace it in .env with a real key."
            ),
            "help_url": "https://console.anthropic.com/settings/keys",
            "env_file": os.path.join(config.BASE_DIR, ".env"),
        }

    return {"ready": True, "configured": True, "reason": "", "model": MODEL}


_PLACEHOLDER_HINTS = ("your-key", "your_key", "yourkey", "xxx", "...",
                      "replace", "changeme", "example", "placeholder",
                      "sk-ant-here", "invalid", "dummy", "test-value")


def _looks_like_placeholder(key: str) -> bool:
    """True when the value is obviously a template rather than a real key."""
    low = key.lower()
    if any(hint in low for hint in _PLACEHOLDER_HINTS):
        return True
    # Real keys are long; anything this short cannot be one.
    return len(key) < 24


# --------------------------------------------------------------------------
# Tools - each one reads the database and returns JSON
# --------------------------------------------------------------------------
def _json(payload: Any) -> str:
    return json.dumps(payload, default=str, separators=(",", ":"))


def _build_tools():
    """Wrap the shared tools with the Anthropic SDK decorator.

    The implementations live in tools_data.py so the Gemini engine runs exactly
    the same queries - a difference there would mean the two providers could
    give different answers to the same question.
    """
    from anthropic import beta_tool

    return [beta_tool(fn) for fn in tools_data.ALL_TOOLS]


# Re-exported so existing callers and tests keep working.
_validate_select = tools_data.validate_select
_apply_row_cap = tools_data.apply_row_cap
MAX_ROWS = tools_data.MAX_ROWS


def _system_prompt() -> str:
    return tools_data.system_prompt()


# --------------------------------------------------------------------------
# Running a turn
# --------------------------------------------------------------------------
def answer(question: str, history: list[dict] | None = None) -> Iterator[dict]:
    """Answer one question, yielding progress events as it goes.

    Yields dicts shaped {"type": "tool"|"text"|"done"|"error", ...} so the web
    layer can stream them to the browser without knowing anything about the SDK.
    """
    status = availability()
    if not status["ready"]:
        yield {"type": "error", "message": status["reason"]}
        return

    import anthropic

    client = anthropic.Anthropic()
    tools = _build_tools()

    messages: list[dict] = []
    for turn in (history or [])[-10:]:      # keep the last few turns only
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    try:
        runner = client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            # The system prompt and tool schemas are identical on every turn,
            # so caching this prefix makes follow-up questions much cheaper.
            system=[{"type": "text", "text": _system_prompt(),
                     "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT},
            tools=tools,
            messages=messages,
        )

        final_text = ""
        for message in runner:
            # Surface which tools ran, so the user can see the answer is
            # grounded rather than guessed.
            for block in message.content:
                if block.type == "tool_use":
                    yield {"type": "tool", "name": block.name,
                           "input": block.input}
                elif block.type == "text" and block.text.strip():
                    final_text = block.text

            if message.stop_reason == "refusal":
                detail = getattr(message, "stop_details", None)
                yield {"type": "error",
                       "message": "The model declined this request"
                                  + (f" ({detail.category})" if detail else "")}
                return

        yield {"type": "text", "text": final_text}
        yield {"type": "done"}

    except anthropic.AuthenticationError:
        yield {"type": "error",
               "message": "The API key was rejected. Check ANTHROPIC_API_KEY."}
    except anthropic.RateLimitError as exc:
        retry = exc.response.headers.get("retry-after", "60")
        yield {"type": "error",
               "message": f"Rate limited by the API. Try again in {retry}s."}
    except anthropic.APIStatusError as exc:
        if exc.status_code >= 500:
            yield {"type": "error",
                   "message": f"Anthropic server error ({exc.status_code}). "
                              "Try again shortly."}
        else:
            yield {"type": "error", "message": f"API error: {exc.message}"}
    except anthropic.APIConnectionError:
        yield {"type": "error",
               "message": "Could not reach the Anthropic API. Check the network."}
    except Exception as exc:                       # never 500 the chat endpoint
        log.exception("assistant failed")
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
        raise AssistantUnavailable(error)
    return text


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    status = availability()
    if not status["ready"]:
        print(f"Assistant unavailable: {status['reason']}")
        raise SystemExit(1)

    if len(sys.argv) > 1:
        for ev in answer(" ".join(sys.argv[1:])):
            if ev["type"] == "tool":
                print(f"  [tool] {ev['name']}({ev['input']})")
            elif ev["type"] == "text":
                print("\n" + ev["text"])
            elif ev["type"] == "error":
                print(f"Error: {ev['message']}")
        raise SystemExit(0)

    print(f"Stock assistant ({MODEL}). Ctrl-C to quit.\n")
    convo: list[dict] = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        convo.append({"role": "user", "content": q})
        reply = ""
        for ev in answer(q, convo[:-1]):
            if ev["type"] == "tool":
                print(f"  [{ev['name']}]")
            elif ev["type"] == "text":
                reply = ev["text"]
                print(f"\n{reply}\n")
            elif ev["type"] == "error":
                print(f"  ! {ev['message']}\n")
        convo.append({"role": "assistant", "content": reply})
