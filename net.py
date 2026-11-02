"""HTTP session for Yahoo Finance.

Two things this module exists to solve:

1. TLS.  On many Windows machines (corporate proxy / MITM inspection) the
   certifi CA bundle that `requests` ships with does not contain the issuing
   root, so every HTTPS call dies with CERTIFICATE_VERIFY_FAILED.  `truststore`
   makes Python use the *Windows certificate store* instead, which does trust
   it.  We never disable verification.

2. Auth.  Yahoo's quote endpoint needs a cookie + "crumb" pair.  We fetch them
   once per session and cache them.
"""
from __future__ import annotations

import logging

import config

log = logging.getLogger(__name__)

# Must run before requests/urllib3 build any SSL context. config does this on
# import for the whole process; this call is the explicit, idempotent guarantee
# that it happened before the first request is built.
if not config.enable_os_truststore():  # pragma: no cover - optional dependency
    log.warning("truststore not installed; falling back to certifi CA bundle")

import requests  # noqa: E402  (import order is deliberate)

_session: requests.Session | None = None
_crumb: str | None = None


def get_session() -> requests.Session:
    """A single shared session so cookies persist across calls."""
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": config.USER_AGENT})
        _session = s
    return _session


def get_crumb() -> str | None:
    """Cookie + crumb handshake required by the /v7/finance/quote endpoint.

    Returns None if the handshake fails - callers must degrade gracefully
    rather than crash, because price history does not need a crumb.
    """
    global _crumb
    if _crumb is not None:
        return _crumb

    s = get_session()
    try:
        # fc.yahoo.com is what actually sets the "A3" consent cookie the crumb
        # endpoint checks for; it answers 404, which is expected and fine.
        for seed in ("https://fc.yahoo.com",
                     "https://finance.yahoo.com/quote/AAPL"):
            try:
                s.get(seed, timeout=config.REQUEST_TIMEOUT)
            except requests.RequestException:
                pass
        r = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb",
                  timeout=config.REQUEST_TIMEOUT)
        if r.status_code == 200 and r.text.strip():
            _crumb = r.text.strip()
            log.info("obtained Yahoo crumb")
            return _crumb
        log.warning("crumb handshake returned HTTP %s", r.status_code)
    except requests.RequestException as exc:
        log.warning("crumb handshake failed: %s", exc)
    return None


def online() -> bool:
    """Cheap reachability probe used to decide online vs offline ingest."""
    try:
        r = get_session().get(
            config.YAHOO_CHART_URL.format(symbol="AAPL"),
            params={"range": "1d", "interval": "1d"},
            timeout=10,
        )
        return r.status_code == 200
    except requests.RequestException:
        return False
