"""Central configuration for the stock dashboard."""
from __future__ import annotations

import datetime as dt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load_dotenv() -> None:
    """Read KEY=value lines from a local .env into the environment.

    Saves fighting with Windows environment-variable scoping, where `setx`
    writes a value the already-running shell cannot see.  Real environment
    variables always win, so this never overrides a deliberately exported key.
    """
    path = os.path.join(BASE_DIR, ".env")
    try:
        # utf-8-sig, not utf-8: PowerShell's `Set-Content -Encoding utf8` and
        # Notepad both write a BOM, which would otherwise become part of the
        # first key name and make the variable silently invisible.
        with open(path, encoding="utf-8-sig") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        pass
    except OSError:
        pass


_load_dotenv()


_truststore_ready = False


def enable_os_truststore() -> bool:
    """Make Python trust the OS certificate store, once per process.

    On many Windows machines a proxy re-signs HTTPS with a root that is in the
    Windows store but not in the CA bundle Python libraries ship with, so every
    outbound call dies with CERTIFICATE_VERIFY_FAILED.  This affects anything
    that talks to the network - the Yahoo fetcher, and the Gemini SDK - so it
    lives here, in the module they all import, rather than in one of them.

    Never disables verification; it swaps which trust store is consulted.
    """
    global _truststore_ready
    if _truststore_ready:
        return True
    try:
        import truststore

        truststore.inject_into_ssl()
        _truststore_ready = True
        return True
    except ImportError:
        return False


enable_os_truststore()

# --- Database -------------------------------------------------------------
def _database_url() -> str:
    """Where the price data lives.

    Order: an explicit STOCK_DB_URL, then the local working database, then the
    committed seed.

    The seed is a small prices-only snapshot checked into the repository so a
    fresh clone - or a serverless deploy, which has no build step to run the
    ingest - renders real data instead of an empty page. It is copied to a
    writable temp directory first, because a serverless filesystem is
    read-only and SQLite needs to write a journal even to read.
    """
    explicit = os.environ.get("STOCK_DB_URL")
    if explicit:
        return explicit

    working = os.path.join(DATA_DIR, "stocks.db")
    if os.path.exists(working) and os.access(working, os.W_OK):
        return "sqlite:///" + working

    seed = os.path.join(DATA_DIR, "seed.db")
    if os.path.exists(seed):
        import shutil
        import tempfile

        target = os.path.join(tempfile.gettempdir(), "stocks_seed.db")
        try:
            if not os.path.exists(target) or (
                    os.path.getmtime(seed) > os.path.getmtime(target)):
                shutil.copyfile(seed, target)
            return "sqlite:///" + target
        except OSError:                      # fall through to the default
            pass

    return "sqlite:///" + working


DATABASE_URL = _database_url()

# --- Timelines -------------------------------------------------------------
# The DASHBOARD window is year-to-date; the ANALYSIS window is the trailing
# few years.  Both live in one table - queries are scoped by date, so loading
# deep history never changes what the year-to-date dashboard reports.
HISTORY_YEARS = 3


def year_start() -> dt.date:
    """First calendar day of the current year (the dashboard window)."""
    return dt.date(dt.date.today().year, 1, 1)


def history_start(years: int = HISTORY_YEARS) -> dt.date:
    """Start of the trailing multi-year analysis window."""
    t = dt.date.today()
    try:
        return t.replace(year=t.year - years)
    except ValueError:      # 29 February
        return t.replace(year=t.year - years, day=28)


def today() -> dt.date:
    return dt.date.today()


# --- The universe we track ------------------------------------------------
# We pull more than 10 so the "top 10" is a genuine ranking, not a fixed list.
TICKERS: dict[str, str] = {
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corporation",
    "NVDA":  "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN":  "Amazon.com, Inc.",
    "META":  "Meta Platforms, Inc.",
    "AVGO":  "Broadcom Inc.",
    "TSLA":  "Tesla, Inc.",
    "BRK-B": "Berkshire Hathaway Inc.",
    "JPM":   "JPMorgan Chase & Co.",
    "WMT":   "Walmart Inc.",
    "LLY":   "Eli Lilly and Company",
    "V":     "Visa Inc.",
    "ORCL":  "Oracle Corporation",
    "MA":    "Mastercard Incorporated",
    "NFLX":  "Netflix, Inc.",
    "XOM":   "Exxon Mobil Corporation",
    "COST":  "Costco Wholesale Corporation",
    "JNJ":   "Johnson & Johnson",
    "AMD":   "Advanced Micro Devices, Inc.",
}

# A market proxy, fetched alongside the companies so risk metrics have
# something to measure against.  Excluded from every ranking - it is an index
# fund, not a company, and it has no market cap.
BENCHMARKS: dict[str, str] = {
    "SPY": "SPDR S&P 500 ETF Trust",
}

# How many companies the dashboard shows.
TOP_N = 10

# Risk-free rate used for Sharpe ratios (annualised).  Roughly the trailing
# 3-year average yield on short-dated US Treasuries; change it and every
# Sharpe number in the report moves, so it is stated in the output.
RISK_FREE_RATE = 0.042

# Ranking modes exposed in the UI.
RANK_MODES = {
    "market_cap": "Market capitalisation",
    "price":      "Latest share price",
    "ytd":        "YTD return %",
    "volume":     "Average daily volume",
}
DEFAULT_RANK = "market_cap"

# --- Network --------------------------------------------------------------
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
REQUEST_TIMEOUT = 25
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
