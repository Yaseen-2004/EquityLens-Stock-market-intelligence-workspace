"""Streamlit interface for the stock analytics project.

All screens read the bundled SQLite snapshot. This UI does not call Yahoo
Finance or any other market-data endpoint. To change the snapshot, run the
separate ingestion script intentionally.
"""
from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "stocks.db"

st.set_page_config(page_title="EquityLens | Stock Analytics", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(145deg,#08111f 0%,#0d1728 55%,#101c30 100%); color:#e8eef8;}
[data-testid="stSidebar"] {background:#0a1423;border-right:1px solid #1d2c42;}
[data-testid="stHeader"] {background:rgba(8,17,31,.8);}
h1,h2,h3 {font-family:'Manrope',sans-serif;color:#f4f7fc;}
.hero {padding:24px 28px;border:1px solid #263a56;border-radius:18px;background:linear-gradient(120deg,#142943,#102034);margin-bottom:18px;}
.hero-kicker {color:#72d6c2;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;}
.hero-title {font-family:Manrope,sans-serif;font-size:32px;font-weight:800;margin:5px 0;color:#f5f8ff;}
.hero-sub {color:#a9bad0;font-size:14px;}
.metric {background:#111f32;border:1px solid #263951;border-radius:14px;padding:17px 18px;min-height:104px;}
.metric-label {color:#9eb0c8;font-size:12px;text-transform:uppercase;letter-spacing:1px;}
.metric-value {font-family:Manrope,sans-serif;color:#f3f7ff;font-size:25px;font-weight:800;margin-top:7px;}
.metric-note {color:#7fd7c3;font-size:12px;margin-top:2px;}
.note {padding:12px 15px;border-radius:10px;background:#13283a;border:1px solid #28516a;color:#b9d8e5;font-size:13px;}
section[data-testid="stSidebar"] .stRadio label {color:#c5d2e2;}
.stButton>button {border-radius:9px;border:1px solid #31506d;background:#18324c;color:#eff6ff;}
[data-testid="stDataFrame"] {border:1px solid #263951;border-radius:12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    if not DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()
    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as con:
        prices = pd.read_sql_query("""
            SELECT c.symbol, c.name, c.currency, c.exchange, c.market_cap,
                   p.date, p.open, p.high, p.low, p.close, p.volume
            FROM price_history p JOIN companies c ON c.id=p.company_id
            ORDER BY p.date, c.symbol
        """, con, parse_dates=["date"])
        companies = pd.read_sql_query("SELECT * FROM companies ORDER BY symbol", con)
    return prices, companies

prices, companies = load_data()
if prices.empty:
    st.error("No local stock history was found. Place a populated data/stocks.db file in the project folder.")
    st.stop()

min_date, max_date = prices.date.min().date(), prices.date.max().date()

with st.sidebar:
    st.markdown("## 📈 EquityLens")
    st.caption("Stock market intelligence workspace")
    page = st.radio("WORKSPACE", ["Overview", "Stock Explorer", "Database", "Returns & Risk", "About"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Data source**")
    st.caption("Local SQLite snapshot")
    st.caption(f"{min_date:%d %b %Y} — {max_date:%d %b %Y}")
    st.caption("Offline / read-only in this UI")


def metric(label, value, note=""):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def hero(title, subtitle, kicker="STOCK MARKET ANALYTICS"):
    st.markdown(f'<div class="hero"><div class="hero-kicker">{kicker}</div><div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div>', unsafe_allow_html=True)

if page == "Overview":
    hero("Market overview", "Explore the historical stock dataset, compare companies, and inspect performance indicators.")
    latest = prices[prices.date == prices.date.max()].copy()
    first = prices.sort_values("date").groupby("symbol").first()["close"]
    last = latest.set_index("symbol")["close"]
    returns = ((last / first.reindex(last.index) - 1) * 100).dropna()
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric("Companies", f"{companies.shape[0]:,}", "Tracked in local database")
    with c2: metric("Price records", f"{len(prices):,}", "Daily OHLCV rows")
    with c3: metric("Historical coverage", f"{(max_date-min_date).days:,} days", f"{min_date:%d %b %Y} to {max_date:%d %b %Y}")
    with c4: metric("Best total return", f"{returns.max():+.2f}%" if len(returns) else "N/A", returns.idxmax() if len(returns) else "")
    st.subheader("Latest stored close by company")
    st.caption(f"Latest date in this database: {max_date:%d %B %Y}. These are stored historical values, not live quotes.")
    latest = latest.sort_values("close", ascending=False)
    st.bar_chart(latest.set_index("symbol")["close"], height=320)
    st.subheader("Company snapshot")
    display = latest[["symbol","name","date","close","volume"]].copy()
    display["date"] = display.date.dt.strftime("%d %b %Y")
    st.dataframe(display.rename(columns={"symbol":"Ticker","name":"Company","date":"Stored date","close":"Close","volume":"Volume"}), use_container_width=True, hide_index=True)

elif page == "Stock Explorer":
    hero("Stock explorer", "Select a ticker and date window to inspect its stored daily prices.")
    symbols = sorted(prices.symbol.unique().tolist())
    selected = st.selectbox("Company", symbols, format_func=lambda s: f"{s} — {companies.loc[companies.symbol==s,'name'].iloc[0]}")
    df = prices[prices.symbol == selected].sort_values("date").copy()
    start,end = st.date_input("Date range", value=(max(min_date, max_date.replace(year=max_date.year-1)), max_date), min_value=min_date, max_value=max_date)
    if start > end: st.warning("Start date must be before end date.")
    else: df = df[(df.date.dt.date >= start) & (df.date.dt.date <= end)]
    if df.empty: st.info("No records in this date range.")
    else:
        first_close,last_close=float(df.close.iloc[0]),float(df.close.iloc[-1])
        pct=(last_close/first_close-1)*100 if first_close else 0
        a,b,c=st.columns(3)
        with a: metric("First close", f"${first_close:,.2f}", df.date.iloc[0].strftime("%d %b %Y"))
        with b: metric("Last close", f"${last_close:,.2f}", df.date.iloc[-1].strftime("%d %b %Y"))
        with c: metric("Period return", f"{pct:+.2f}%", "Based on stored closing prices")
        st.subheader("Closing price history")
        st.line_chart(df.set_index("date")["close"], height=340)
        st.subheader("Daily OHLCV records")
        st.dataframe(df[["date","open","high","low","close","volume"]].sort_values("date",ascending=False), use_container_width=True, hide_index=True)

elif page == "Database":
    hero("Historical database", "Browse and export the records physically stored in the bundled SQLite database.", "DATABASE • LOCAL SNAPSHOT")
    st.markdown('<div class="note">This section is deliberately <b>not live data</b>. It reads only <code>data/stocks.db</code> in read-only mode. No market API is called and no automatic refresh occurs. The latest available record is shown with its actual stored date.</div>', unsafe_allow_html=True)
    st.write("")
    d1,d2,d3=st.columns(3)
    with d1: metric("Database file", "stocks.db", "SQLite • read-only view")
    with d2: metric("Total records", f"{len(prices):,}", "Price history rows")
    with d3: metric("Stored through", max_date.strftime("%d %b %Y"), "Latest database date")
    st.subheader("Filter database records")
    f1,f2,f3=st.columns([1,1,1])
    with f1: chosen=st.multiselect("Ticker", sorted(prices.symbol.unique()), default=[])
    with f2: date_from=st.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="db_from")
    with f3: date_to=st.date_input("To", value=max_date, min_value=min_date, max_value=max_date, key="db_to")
    filtered=prices.copy()
    if chosen: filtered=filtered[filtered.symbol.isin(chosen)]
    filtered=filtered[(filtered.date.dt.date>=date_from)&(filtered.date.dt.date<=date_to)]
    st.caption(f"Showing {len(filtered):,} matching records • sorted newest first")
    page_size=st.selectbox("Rows per page", [25,50,100,250,500], index=1)
    pages=max(1,(len(filtered)+page_size-1)//page_size)
    current=st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
    shown=filtered.sort_values(["date","symbol"],ascending=[False,True]).iloc[(current-1)*page_size:current*page_size]
    st.dataframe(shown[["date","symbol","name","open","high","low","close","volume"]], use_container_width=True, hide_index=True)
    csv=filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download filtered records (CSV)", csv, file_name="stock_database_export.csv", mime="text/csv")
    with st.expander("Database schema"):
        st.markdown("**companies** — ticker, company name, currency, exchange, market capitalisation.\n\n**price_history** — one row per company per trading date, containing open, high, low, close and volume.\n\nA unique constraint on `(company_id, date)` prevents duplicate daily records.")

elif page == "Returns & Risk":
    hero("Returns & risk", "A descriptive comparison calculated from the historical rows stored in SQLite.")
    symbols=sorted(prices.symbol.unique().tolist())
    chosen=st.multiselect("Choose companies", symbols, default=symbols[:5])
    df=prices[prices.symbol.isin(chosen)].copy()
    if df.empty: st.info("Choose at least one company.")
    else:
        pivot=df.pivot(index="date",columns="symbol",values="close").sort_index().ffill()
        normalized=pivot.divide(pivot.iloc[0]).multiply(100)
        st.subheader("Indexed closing-price performance (first date = 100)")
        st.line_chart(normalized, height=380)
        daily=pivot.pct_change().dropna()
        summary=pd.DataFrame({"Total return %":(pivot.iloc[-1]/pivot.iloc[0]-1)*100,"Annualized volatility %":daily.std()*((252)**0.5)*100,"Observations":pivot.count()}).round(2)
        st.subheader("Return and volatility summary")
        st.dataframe(summary, use_container_width=True)
        st.caption("Volatility is annualized using 252 trading days. Metrics describe the selected stored period and are not investment advice.")

elif page == "About":
    hero("About this project", "A historical stock analytics and forecasting workspace backed by a local relational database.")
    st.subheader("System components")
    st.markdown("- **SQLite:** stores company metadata and daily OHLCV history.\n- **SQLAlchemy:** database models and query layer used by the original analytics backend.\n- **Pandas:** filtering, aggregation and return calculations.\n- **Streamlit:** interactive dashboard and database browser.\n- **Forecasting modules:** existing project modules can be run separately to generate saved forecast outputs.")
    st.subheader("Run the interface")
    st.code("pip install -r requirements.txt\nstreamlit run streamlit_app.py", language="bash")
    st.info("The Streamlit interface is intentionally offline for market data. Run the separate ingestion workflow only when you deliberately want to update the local database snapshot.")
