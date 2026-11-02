# Stock Market Dashboard

## Streamlit UI (recommended)

The project now includes an interactive Streamlit dashboard. It reads the
existing SQLite snapshot at `data/stocks.db` and provides Overview, Stock
Explorer, Historical Database, Returns & Risk, and About sections.

### Run locally on Windows

```powershell
cd stock_prediction
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in
your browser. You do not need to start the Flask server for this interface.

### Important: database section uses stored data only

The **Database** page is intentionally offline and read-only. It queries the
local `data/stocks.db` file, displays the actual date range stored in that
file, supports ticker/date filtering and CSV export, and does not call Yahoo
Finance or any other market-data API. It does not auto-refresh. The bundled
snapshot contains historical records through **10 September 2026**. The values
are historical database records, not current market quotes.

To update the snapshot in the future, deliberately run the separate ingestion
script (`python ingest.py`). The Streamlit UI itself never triggers ingestion.

---

## Quick start

```bash
pip install -r requirements.txt
python ingest.py --years 3   # build the database (~30s, 3 years of history)
python insertdata.py         # build the vector store (~20 min, or --no-prices)
python app.py                # http://127.0.0.1:5000
python analysis.py           # print the risk/return report
python forecast.py --save    # backtest the forecasting models
```

If you have no internet access, add `--offline` to the ingest command.

**Run it with the reloader off** — `python app.py` does this by default. Use
`FLASK_DEBUG=1 python app.py` only while editing the code: the auto-reloader
restarts the server on any file change and **drops every in-flight request**.
A chat turn takes tens of seconds, so that shows up in the panel as
`Failed to fetch` halfway through an answer.

---

## What you get

| Page | What it shows |
|------|---------------|
| `/` | Bar graph of the top 10, a rebased performance line chart, and a full data table |
| `/company/<SYMBOL>` | One company: price (YTD / 1y / 3y), risk vs the benchmark, drawdown curve, and its forecast |
| `/api/top?mode=&limit=` | The ranking as JSON |
| `/api/company/<SYMBOL>` | One company's full daily series as JSON |
| `/api/coverage` | Row counts and the date range actually stored |
| `/analysis` | Three-year risk and return study of the top 10 |
| `/api/analysis?years=N` | The same analysis as JSON |
| `/api/assistant/chat` | Chatbot, streamed as server-sent events |
| `/api/assistant/status` | Which chat engine is active |
| `/forecast` | Forecast backtest: does any model beat a random walk? |
| `/api/forecast` | The saved backtest as JSON |

The ranking metric is switchable: **market capitalisation**, **latest share
price**, **YTD return %**, or **average daily volume**.

---

## The step-by-step process

### Step 1 — Choose the stack

Flask, because this app is a handful of pages over one database and Flask
carries the least ceremony for that. Django would bring an admin, migrations
and an ORM you do not need here; FastAPI is built for JSON APIs and you would
still be bolting Jinja on for the HTML. If this grows a login system or a
multi-user portfolio, Django starts to earn its weight.

The rest: **SQLAlchemy** for the ORM, **SQLite** for storage (a single file, no
server to run), **Chart.js** for the graphs.

### Step 2 — Lay out the project

```
stock_prediction/
├── config.py          # tickers, date window, ranking modes - all knobs in one place
├── models.py          # ORM tables: Company, PriceHistory
├── database.py        # engine + session plumbing
├── net.py             # HTTP session: TLS handling + Yahoo's cookie/crumb auth
├── ingest.py          # fetch -> database (run this to load data)
├── seed_fallback.py   # synthetic data generator for offline use
├── services.py        # the queries: rankings, series, coverage
├── analysis.py        # risk/return metrics over the multi-year window
├── forecast.py        # forecasting models + walk-forward backtest
├── insertdata.py      # RAG pipeline: narrow -> chunk -> embed -> insert
├── vectorstore.py     # semantic search over the embedded chunks
├── chatbot.py         # offline chat engine - no API key needed
├── tools_data.py      # the six DB tools + system prompt, shared by both AIs
├── assistant.py       # Claude engine
├── assistant_gemini.py# Google Gemini engine
├── app.py             # Flask routes (HTML + JSON + the chat stream)
├── test_app.py        # end-to-end checks
├── templates/         # Jinja2: base, index, company, analysis, 404
├── static/css|js/     # stylesheet, chart code, vendored Chart.js
└── data/stocks.db     # created by ingest.py
```

The separation that matters: **`services.py` never imports Flask, and `app.py`
never writes SQL.** That keeps the analytics testable on their own and lets you
put a different front end on the same queries later.

### Step 3 — Design the database schema

Two tables ([models.py](models.py)):

- **`companies`** — one row per company: symbol, name, exchange, market cap.
- **`price_history`** — one row per company **per trading day**: OHLC, volume.

The important detail is this constraint:

```python
UniqueConstraint("company_id", "date", name="uq_company_date")
```

The pair *(company, date)* is the natural key. It makes re-running the ingest
**idempotent** — run it five times and you still have one row per day, not
five. Without it you silently accumulate duplicates and every average you
compute is wrong.

### Step 4 — Define the timeline

The window is computed, never hard-coded ([config.py](config.py)):

```python
def year_start():
    return dt.date(dt.date.today().year, 1, 1)
```

So the dashboard says "since 1 January" and stays correct when the year rolls
over. `ingest.py` also clamps every fetched row to that window — Yahoo's
`period1` parameter is timezone-sensitive and will otherwise slip a
31 December bar into your data.

### Step 5 — Fetch the data

Yahoo Finance has two endpoints worth knowing:

| Endpoint | Gives you | Auth |
|---|---|---|
| `/v8/finance/chart/{symbol}` | daily OHLCV history | none |
| `/v7/finance/quote` | market cap, exchange | cookie + crumb |

The crumb handshake is the fiddly part, handled in [net.py](net.py):

1. `GET https://fc.yahoo.com` — sets the `A3` consent cookie (it answers 404;
   that is expected). **This step is what actually matters** — hitting only
   `finance.yahoo.com` is not enough.
2. `GET /v1/test/getcrumb` — returns a short token.
3. Pass `crumb=<token>` on every quote request, on the same session.

Yahoo rejects the handshake unless the `User-Agent` looks like a real browser.

Prices use the **adjusted close**, which back-corrects for stock splits and
dividends. Without it a 10-for-1 split looks like a 90% crash.

### Step 6 — Solve the SSL problem (Windows)

On many Windows machines every HTTPS call fails with
`CERTIFICATE_VERIFY_FAILED`, because a corporate proxy re-signs traffic with a
root certificate that is in the **Windows certificate store** but not in the
CA bundle `requests` ships with. The fix is one import, before `requests`
builds any SSL context:

```python
import truststore
truststore.inject_into_ssl()   # use the OS certificate store
```

Note what this does **not** do: it never sets `verify=False`. Disabling
verification is the common advice and it is wrong — it turns a certificate
problem into an open door.

### Step 7 — Load the database

```bash
python ingest.py            # live data for the current year
python ingest.py --offline  # deterministic synthetic data, no network
python ingest.py --reset    # clear price history first
```

`ingest.py` reads existing dates per company, inserts only what is missing, and
updates a close only when it has actually changed. Safe to run on a schedule.
If Yahoo is unreachable it falls back to `seed_fallback.py` automatically.

### Step 8 — Write the ranking query

In [services.py](services.py). The shape to copy:

```sql
-- one GROUP BY pass for the per-company bounds...
SELECT company_id, MIN(date), MAX(date), AVG(volume), MAX(close), MIN(close)
FROM price_history GROUP BY company_id
-- ...then join back twice to read the close on the first and last day
```

This is two index lookups per company. The tempting alternative — pull every
row into Python and sort there — works fine at 3,000 rows and falls over at
3,000,000. Write the query that scales.

YTD return is then just:

```python
ytd_pct = (last_close - first_close) / first_close * 100
```

`get_top_companies()` sorts by the chosen metric, slices to 10, and stamps a
`rank` on each row so the template and the chart agree on the order.

### Step 9 — Build the Flask routes

[app.py](app.py) stays thin: parse the query string, call a service, render.
Every route opens a session with `with SessionLocal() as session:` so
connections are always returned.

`/api/refresh` runs an ingest on a background thread and returns `202`
immediately, so the browser is not held open for the length of a network job.

### Step 10 — Choose the chart forms deliberately

This is where most dashboards go wrong, so the reasoning is worth stating:

- **Ranked magnitude → horizontal bar chart.** Horizontal because company names
  and value labels fit on one line without rotating text.
- **All bars are one colour.** Shading them darker-where-bigger is a real
  anti-pattern: bar *length* already encodes magnitude, so a value ramp
  double-encodes it and burns the one free channel you have. The exception is
  YTD mode, where the sign is genuine information — there the bars diverge.
- **Gains are blue, losses are red — not green/red.** Red-green is precisely
  the pair that collapses under the most common form of colour blindness. Blue
  against red separates cleanly for everyone (validated at ΔE 21.6 under
  simulated protanopia; green/red does not clear the floor).
- **The comparison chart is rebased to 100.** A $1,100 stock and a $250 stock
  cannot share a price axis meaningfully. Indexing both to 100 on the first
  trading day puts them on one honest scale — and avoids a dual-axis chart,
  which is the single most misleading thing you can put on a dashboard.
- **The table is not decoration.** Three of the light-mode chart colours sit
  below 3:1 contrast against the surface, so the numbers must also be readable
  as text. Every value in both charts appears in the table.

Colours live as CSS custom properties in [style.css](static/css/style.css) and
are read back in JavaScript with `getComputedStyle`. Light and dark are defined
in exactly one place, and the charts redraw themselves when the theme changes.

### Step 11 — Test it

```bash
python test_app.py
```

Checks that the stored window really is this year, that the ranking is
descending and exactly 10 long, that every route answers, that an unknown
symbol 404s, and that a bogus `?mode=` falls back instead of crashing.

---

## The analysis layer

`python analysis.py` builds a three-year risk and return study of the same top
10, reading only from the database — no network calls, so the report is
reproducible from a fixed dataset.

```bash
python analysis.py                     # 3-year report
python analysis.py --years 5           # a different window
python analysis.py --json out.json     # machine-readable output
```

### Two windows, one table

The table holds three years of history, but the dashboard still reports
year-to-date. That works because **every query is scoped by date** — the
default window is `config.year_start()` and the analysis passes
`config.history_start()` instead.

This is the trap worth naming: the original aggregate had no `WHERE` on the
date, so loading deep history would have silently turned "YTD return" into
"since the beginning of the data" without a single error. Aggregates over a
growing table need an explicit window.

### What is computed

| Metric | Meaning | Why it is here |
|---|---|---|
| **CAGR** | constant yearly growth rate | comparable across periods; a raw total is not |
| **Volatility** | annualised standard deviation of daily returns | the usual definition of risk |
| **Sharpe** | excess return per unit of total risk | return alone flatters volatile stocks |
| **Sortino** | excess return per unit of *downside* risk | upside swings are not a problem |
| **Max drawdown** | deepest fall from a running high | the loss actually experienced |
| **Beta** | sensitivity to the benchmark | separates market exposure from skill |
| **Alpha** | annualised CAPM residual | return left after paying for that exposure |
| **Correlation** | co-movement of daily returns | tells you whether the basket diversifies |

Two details that are easy to get wrong and are handled explicitly:

- **Aligned calendars.** Prices are pivoted wide and inner-joined on date, so
  every return comes from a day on which *all* symbols traded. Mismatched
  calendars are the standard way to get a wrong correlation matrix.
- **Drawdown against the running maximum**, not the first or highest price.
  That is the loss an investor actually lives through.

`SPY` is fetched as a market benchmark and excluded from every ranking — it is
an index fund, not a company, and has no market cap. The regression of the
benchmark against itself is used as a test: it must produce beta 1.00 and
alpha 0.00, and `test_app.py` asserts exactly that.

The Sharpe numbers depend on `config.RISK_FREE_RATE` (currently 4.2%); change
it and every Sharpe moves, which is why the report prints the rate it used.

---

## Forecasting — and whether it works

```bash
python forecast.py               # backtest all ten, ~2 minutes
python forecast.py NVDA          # one company
python forecast.py --save        # write the run for the /forecast page
```

The project is called `stock_prediction`, so it should predict something. It
does — and it measures itself honestly, which is the more useful half.

### The result

**No model beat the naive random walk.** Averaged over ten companies at a
five-day horizon:

| Model | MAPE | Direction | Skill vs naive |
|---|---|---|---|
| naive (benchmark) | 3.68% | — | 0.000 |
| drift | 3.74% | 51.8% | −0.016 |
| sma20 | 5.08% | 54.7% | −0.792 |
| ridge | 3.96% | 50.9% | −0.092 |
| gbt | 4.05% | 50.2% | −0.179 |

Every skill score is negative: each model did *worse* than assuming the price
does not change. Directional accuracy sits within a few points of a coin flip.

That is the expected outcome, and reporting it is the point. Daily closing
prices are close to a random walk; a project that claimed otherwise from 250
test windows would be claiming an edge that professionals spend careers failing
to find.

### How it is measured

- **Walk-forward.** At each step the models see prices only up to that day,
  predict *h* days ahead, and are scored against what happened. Nothing is
  scored on data it was fitted to.
- **No lookahead.** Training rows are restricted to those whose outcome had
  already occurred by the prediction date. Getting this wrong is the classic
  way to build a forecaster that looks brilliant in a backtest and useless in
  reality — `test_app.py` asserts the target really is the realised forward
  return.
- **Skill score** is `1 − MSE/MSE_naive`: above zero beats the random walk,
  below zero is worse than doing nothing.
- **The uncertainty band** comes from recent volatility, widening with the
  square root of time. It is not a model output and does not narrow because a
  model feels confident.

The band is the honest deliverable. "NVDA will be $232.34" is not a useful
statement; "the 80% range is roughly $216–$250" is.

### A bug worth recording

The first version of the page rendered every chart blank with no error
anywhere. The cause: `direction_pct` is NaN for the naive model (it never
predicts a direction), and Python's `json.dump` writes a bare `NaN`, which is
**not valid JSON**. The browser's `JSON.parse` rejected the whole document, and
a `catch { return; }` swallowed it.

Two fixes: non-finite values are converted to `null` before saving (with
`allow_nan=False` so a stray one fails loudly at write time), and every
`JSON.parse` in the front end now logs to the console instead of failing
silently.

---

## Semantic search (RAG)

```bash
python insertdata.py                       # build the vector store
python insertdata.py --stats               # what is in it
python vectorstore.py "which is safest"    # search it
```

`insertdata.py` runs the pipeline: **narrow → chunk → embed → insert**.

| Stage | What happens |
|---|---|
| **narrow** | Every daily price bar, plus summaries of metrics, risk, forecasts and coverage. |
| **document** | Turn the numbers into *prose*. A wall of comma-separated figures embeds poorly; the model was trained on sentences. |
| **chunk** | Split on sentence boundaries, ~600 chars, carrying whole trailing sentences as overlap. |
| **embed** | `all-MiniLM-L6-v2`, 384 dimensions, running locally — no API key, no cost. |
| **insert** | Store each chunk with its vector in the `document_chunks` table, keyed so a rebuild replaces rather than duplicates. |

About **15,850 chunks**: one per company per trading day, plus ~80 narrative
chunks covering performance, risk profiles, forecasts, correlations, notable
dates and a glossary.

```bash
python insertdata.py               # full build, ~12 minutes
python insertdata.py --no-prices   # narrative only, ~10 seconds
```

The price rows dominate: about 200 daily bars for every narrative chunk, and
they all read almost identically — *"X share price on <date>. It closed at
…"*. Embedded naively, a general question returns a wall of near-duplicate
days and buries the summary that actually answers it. Two ranking controls fix
that without removing anything from the index:

- **`max_per_source`** — no single source may take more than half the results.
- **`prefer_narrative`** — daily rows are held back unless the question looks
  like it is asking about a specific day, price or date (`looks_price_specific`
  checks for a year, ISO date, month, weekday, or words like *closed*,
  *volume*, *traded*).

Every day remains reachable: by naming a date, by `source="price"`, or through
the price tools. This is a ranking policy, not a filter on what is stored.

### Dates are filtered, not embedded

Embedding similarity cannot tell one trading day from another. Every row reads
*"X share price on <date>. It closed at …"*, so they all land in nearly the
same place in vector space. Asked *"what did NVIDIA close at on 8 September
2026?"*, the first build confidently returned **9 June 2026** at a similarity
of 0.656 — wrong day, high score, no signal that anything was off.

So a named date is applied as a **metadata filter** before similarity ranks
anything: `extract_date()` parses ISO, `8 September 2026` and `September 8,
2026`, and price rows are restricted to that day (and to the named ticker, if
one is given). A bare month like *"April 2025"* is deliberately **not** treated
as a day — it would pin retrieval to an arbitrary date inside the month.

That combination is the point: similarity for meaning, filters for facts.
Retrieval alone gets dates wrong in a way that looks confident.

**Still true:** for an exact figure the SQL tools remain the better route.
`get_price_history` reads the number; retrieval finds a sentence *about* it.

### Search

Vectors are L2-normalised at insert time, so cosine similarity is a plain dot
product and the search is one matrix multiply — exact and instant at this size.
FAISS or Chroma would add a second store to keep in sync and buy nothing until
there are orders of magnitude more chunks.

`chatbot.py` falls back to retrieval when no intent matches, and both AI
engines get it as a `search_knowledge` tool.

### Where RAG helps here — and where it does not

For *"which company grew most this year"* the SQL query is strictly better:
exact, and retrieval can only lose information a database already holds. Vector
search earns its place on the open-ended questions the intent matcher cannot
enumerate — *"what happened in April 2025"*, *"which of these is safest"*,
*"why did everything fall at once"* — and, with the daily rows embedded, on
questions that name a date in words rather than as a filter.

### Two things that had to be got right

**Chunk boundaries.** The first version sliced overlap by character count and
produced chunks starting mid-sentence — `"on 2025-06-25. Its beta is..."`. That
fragment embeds to something neither half means, and it showed up directly as
bad retrieval. Overlap now carries whole sentences.

**Knowing when it does not know.** Asked *"what is a stochastic oscillator"* —
a term the store has never heard of — retrieval happily returned "What is
volatility?" at 0.27. No single threshold fixes this: a *correct* hit can score
0.25 while a *wrong* one scores 0.27. So below 0.35 the report says plainly
that nothing closely matched and offers the passages as related reading rather
than as an answer. `test_app.py` asserts that behaviour.

---

## Deploying

The repo is set up for Vercel: `vercel.json` routes everything to
`api/index.py`, which exposes the same Flask app.

Two things a serverless deploy needs that a laptop does not:

**Data.** The price database is generated, not committed — but a deploy has no
build step to run the ingest, so an empty database rendered a 500. A small
prices-only snapshot (`data/seed.db`, 1.7 MB against 34 MB for the full one)
is checked in and used when no working database exists. It is copied to a
temp directory first: a serverless filesystem is read-only, and SQLite writes
a journal even to read.

**A small bundle.** `requirements.txt` is the deployable set. The two heavy
optional pieces are separate files:

| File | Adds | Needed for |
|---|---|---|
| `requirements-ai.txt` | `anthropic` | the Claude engine |
| `requirements-rag.txt` | `sentence-transformers` (pulls torch) | local embeddings for semantic search |

Neither is required. Without them the site serves every page and the offline
chatbot still answers; semantic search reports that it is unavailable rather
than failing. `sentence-transformers` in particular is far too large for a
serverless bundle — install it locally only.

Set `GEMINI_API_KEY` in the host's environment variables to enable the AI
chat; without it the deployed site uses the offline engine.

An empty database now renders a page explaining what to run, rather than a
stack trace — which is what the first deploy actually showed.

---

## The chatbot

A chat panel sits on every page (the **Ask** button, bottom right). It answers
questions about the data in this database — rankings, comparisons, risk,
whatever a query can reach.

There are three engines, and the app picks automatically:

| Engine | Needs | Handles |
|---|---|---|
| **Offline** (`chatbot.py`) | nothing | the question patterns listed below — no key, no network |
| **Gemini** (`assistant_gemini.py`) | `GEMINI_API_KEY` — **free tier** | free-form questions, follow-ups, its own SQL |
| **Claude** (`assistant.py`) | `ANTHROPIC_API_KEY` — paid | the same, generally stronger reasoning |

Selection order is **Gemini → Claude → offline**, so the free option wins when
both keys are present. Pin one with `CHAT_ENGINE=gemini|claude|offline` in
`.env`. The offline engine needs nothing, so the feature always works — a
missing key downgrades the answers rather than removing them. The panel header
says which engine is answering, and so does the server's startup line.

### Using Google Gemini

Get a key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
(free tier), then in `.env`:

```
GEMINI_API_KEY=your-key
```

Restart, and the startup log reads `Chatbot: Google Gemini (gemini-3.8-flash)`.

If the default model is not available to your key, list what is and pin one:

```bash
python assistant_gemini.py --models
```

```
GEMINI_MODEL=gemini-3.5-flash-lite
```

Both AI engines call the **same six tools** from `tools_data.py` and use the
same system prompt, so switching provider does not change what the answers are
grounded in — only how well they are written. `test_app.py` asserts that the
tool list and prompt really are shared.

### What the offline engine handles

It is not a language model — it matches intents and answers from real queries:

- **Rankings** — "top 5 by market cap", "which grew most this year", "worst performers"
- **One company** — "how is NVDA doing", "Tesla price"
- **Comparisons** — "compare NVDA and TSLA" (with a risk-adjusted verdict)
- **Risk** — "which is most volatile", "deepest drawdown", "best Sharpe"
- **Correlation** — "which two move together least"
- **Definitions** — "what is beta", "explain drawdown"
- **Coverage** — "what data do you have"

Anything it cannot match, it says so and lists what it can do, rather than
guessing. Try it from the terminal too:

```bash
python chatbot.py "compare nvda and tsla"
python chatbot.py                             # interactive
```

Two ambiguities in there were worth handling deliberately: **"top 5 by market
cap"** must not resolve *market* to the SPY benchmark, and **"worst
performers"** means returns, not the cheapest share price. Both are asserted in
`test_app.py`.

---

## The AI engines (optional)

A chat panel sits on every page (the **Ask** button, bottom right). It answers
questions about the data in this database — rankings, comparisons, risk,
whatever a SQL query can reach.

### Setup

Get a key from [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys),
then pick one of these.

**A `.env` file — recommended, and the least error-prone:**

```bash
copy .env.example .env        # Windows   (cp on bash)
```

Edit `.env` so it reads `ANTHROPIC_API_KEY=sk-ant-...`, then `python app.py`.
`.env` is already in `.gitignore`, so the key will not be committed.

**Or an environment variable:**

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # this PowerShell window only
python app.py                            # must be the SAME window
```

```powershell
setx ANTHROPIC_API_KEY "sk-ant-..."     # permanent, but see below
```

`setx` writes the value permanently **but does not touch the window you typed
it in** — you must open a new terminal, or the app will still report no key.
That trips up nearly everyone; the `.env` file avoids it entirely.

Without a key the app still runs normally; the chat panel reports that it is
unconfigured instead of failing. There is also a terminal client:

```bash
python assistant.py                             # interactive
python assistant.py "which stock fell hardest?" # one-shot
```

### It is grounded, not remembering

The model is given **no** price data in its prompt. It has six tools and must
call them:

| Tool | What it reads |
|---|---|
| `list_tracked_companies` | symbols and names |
| `get_rankings` | the year-to-date leaderboard, any of the four metrics |
| `get_company_detail` | one company's YTD figures |
| `get_risk_analysis` | the multi-year CAGR / Sharpe / drawdown / correlation study |
| `get_price_history` | a sampled price series for a date range |
| `run_sql` | a read-only `SELECT` for anything the rest do not cover |

This matters more here than in most apps: a language model asked about a stock
price will happily produce a plausible number from memory, and in a finance
tool a confident wrong number is worse than no answer. Grounding it in tool
calls makes that failure mode much harder — and the UI shows which tools ran
under each answer, so you can see the answer was fetched rather than guessed.

### The `run_sql` tool is the risky one

Handing a model a SQL tool is the part worth getting right. The guard in
[assistant.py](assistant.py):

- **`SELECT` / `WITH` only** — anything else is refused by regex before it runs.
- **One statement.** A `;` anywhere is rejected, which kills the
  `SELECT 1; DROP TABLE ...` smuggling trick that defeats a naive prefix check.
- **A keyword denylist** for `INSERT`, `UPDATE`, `DROP`, `ATTACH`, `PRAGMA` and
  friends — defence in depth behind the prefix check.
- **A forced `LIMIT`**, so one broad question cannot pull the whole price table
  into the context window (or the reply).

`test_app.py` asserts each of those, because this is the one tool where a miss
would actually matter.

### Model and cost

`claude-opus-5` with adaptive thinking, at `medium` effort — chat over a small
dataset does not need deep reasoning every turn. Raise `EFFORT` in
`assistant.py` to `"high"` or `"xhigh"` for multi-step analytical questions;
that trades latency and cost for thoroughness.

The system prompt and tool schemas are identical on every turn and are marked
with `cache_control`, so follow-up questions in a conversation re-read that
prefix from cache instead of paying for it again.

### Boundaries in the prompt

The assistant is told to refuse investment advice, to state the window that
every number belongs to, to flag that a three-year sample is not predictive,
and never to invent a ticker or a price. Those rules live in `_system_prompt()`
in [assistant.py](assistant.py) — edit them there.

---

## Two bugs worth knowing about

Both were found only by rendering the page and looking at it — the routes
returned `200` the whole time.

1. **Script order.** `dashboard.js` was loading *before* the
   `<script type="application/json">` data blocks, so it read `null` and drew
   nothing. The page looked perfect except both charts were blank. Data blocks
   now come first in [base.html](templates/base.html).
2. **`display` beating `[hidden]`.** `.refresh-status { display: inline-block }`
   overrode the browser default for `[hidden]`, leaving an empty pill on the
   page. Any rule that sets `display` needs a matching `[hidden]` guard.

---

## Configuration

Everything tunable is in [config.py](config.py):

| Setting | Purpose |
|---|---|
| `TICKERS` | the 20 companies tracked — the top 10 is a real ranking out of these |
| `TOP_N` | how many the dashboard shows |
| `RANK_MODES` | the four ranking metrics |
| `DATABASE_URL` | override with the `STOCK_DB_URL` env var |

To track different companies, edit `TICKERS` and re-run
`python ingest.py --reset`.

To use PostgreSQL instead of SQLite:

```bash
pip install psycopg2-binary
set STOCK_DB_URL=postgresql://user:pass@localhost/stocks
python ingest.py
```

No code changes — that is the payoff for going through SQLAlchemy.

---

## Where to take it next

- **Keep it current** — schedule `python ingest.py` daily; it is idempotent.
- **More history** — change `year_start()` to fetch multiple years and add a
  date-range filter.
- **Sector grouping** — add a `sector` column and rank within sectors.
- **Better forecasting** — the honest baseline is in place and nothing beats it
  yet. Realistic next steps: longer horizons (weekly returns are less noisy than
  daily), cross-sectional features (how a stock moves relative to its sector),
  or predicting volatility instead of price — volatility is genuinely more
  predictable than direction.

---

## Notes

Prices are adjusted closes from Yahoo Finance's public endpoints, for
educational use. Yahoo does not guarantee this API and it can change without
notice — if `ingest.py` starts failing, check `net.py` first. Not investment
advice.
