"""Short-horizon price forecasting, and an honest test of whether it works.

The point of this module is not to predict stock prices - it is to measure,
properly, how well several standard approaches do against the one benchmark
that matters: the naive random walk ("tomorrow's price is today's price").

That comparison is the whole discipline. A model can look impressive on a chart
and still be worse than assuming nothing changes, and a directional accuracy of
52% is what a coin flip produces on a good day. Every number here is generated
by walk-forward validation, so nothing is scored on data it was fitted to.

Run it:

    python forecast.py                    # backtest every top-10 company
    python forecast.py NVDA               # one company, in detail
    python forecast.py NVDA --horizon 10  # a different horizon
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import config
import services
from database import SessionLocal

# Default horizon: how many trading days ahead we predict.
HORIZON = 5

# Walk-forward settings. Refitting the tree model every day would dominate the
# runtime for no measurable benefit, so the expensive models refit on a stride.
MIN_TRAIN = 200
REFIT_EVERY = 21
TEST_DAYS = 250


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def build_features(prices: pd.Series, volumes: pd.Series | None = None
                   ) -> pd.DataFrame:
    """Features known at time t - nothing that peeks into the future.

    Every column is built from closes at or before t. Getting this wrong is
    the classic way to produce a forecaster that looks brilliant in a backtest
    and useless in reality.
    """
    df = pd.DataFrame(index=prices.index)
    logp = np.log(prices)
    ret = logp.diff()

    for lag in (1, 2, 3, 5, 10):
        df[f"ret_lag{lag}"] = ret.shift(lag - 1)

    for win in (5, 10, 20, 60):
        df[f"mom{win}"] = logp - logp.shift(win)
        df[f"vol{win}"] = ret.rolling(win).std()

    for win in (5, 20, 50):
        df[f"sma_ratio{win}"] = logp - np.log(prices.rolling(win).mean())

    # Where the price sits inside its recent range, 0 = low, 1 = high.
    roll_max = prices.rolling(20).max()
    roll_min = prices.rolling(20).min()
    df["range_pos"] = (prices - roll_min) / (roll_max - roll_min).replace(0, np.nan)

    if volumes is not None:
        v = volumes.replace(0, np.nan)
        df["vol_change"] = np.log(v) - np.log(v.rolling(20).mean())

    return df


def make_target(prices: pd.Series, horizon: int) -> pd.Series:
    """Log return from t to t+horizon. NaN where the future is unknown."""
    logp = np.log(prices)
    return logp.shift(-horizon) - logp


# --------------------------------------------------------------------------
# Models - each maps (history, features) -> predicted log return over horizon
# --------------------------------------------------------------------------
class Model:
    name = "base"
    needs_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict_logret(self, x: np.ndarray, hist: pd.Series,
                       horizon: int) -> float:
        raise NotImplementedError


class Naive(Model):
    """Tomorrow is today. The benchmark everything must beat to be worth using."""
    name = "naive"

    def predict_logret(self, x, hist, horizon):
        return 0.0


class Drift(Model):
    """Random walk with drift: extend the average historical daily return."""
    name = "drift"

    def predict_logret(self, x, hist, horizon):
        r = np.diff(np.log(hist.values))
        if len(r) == 0:
            return 0.0
        return float(np.mean(r) * horizon)


class MovingAverage(Model):
    """Predict reversion toward the recent moving average."""
    name = "sma20"

    def predict_logret(self, x, hist, horizon):
        if len(hist) < 20:
            return 0.0
        target = float(np.log(hist.iloc[-20:].mean()))
        return target - float(np.log(hist.iloc[-1]))


class _Sklearn(Model):
    needs_fit = True

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.fitted = False

    def fit(self, X, y):
        if len(X) < 50:
            self.fitted = False
            return
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.fitted = True

    def predict_logret(self, x, hist, horizon):
        if not self.fitted:
            return 0.0
        xs = self.scaler.transform(x.reshape(1, -1))
        return float(self.model.predict(xs)[0])


class RidgeModel(_Sklearn):
    name = "ridge"

    def __init__(self):
        super().__init__()
        self.model = Ridge(alpha=10.0)


class GBTModel(_Sklearn):
    name = "gbt"

    def __init__(self):
        super().__init__()
        self.model = GradientBoostingRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.06,
            subsample=0.8, random_state=0,
        )


def default_models() -> list[Model]:
    return [Naive(), Drift(), MovingAverage(), RidgeModel(), GBTModel()]


# --------------------------------------------------------------------------
# Walk-forward backtest
# --------------------------------------------------------------------------
def backtest(prices: pd.Series, volumes: pd.Series | None = None,
             horizon: int = HORIZON, min_train: int = MIN_TRAIN,
             test_days: int = TEST_DAYS,
             refit_every: int = REFIT_EVERY) -> dict[str, Any]:
    """Score every model on data it has never seen.

    At each step t the models see prices up to and including t, predict the
    close at t+horizon, and are scored against what actually happened. Models
    are refitted on a stride to keep the run fast; the fit only ever uses
    samples whose outcome was already known at t.
    """
    feats = build_features(prices, volumes)
    target = make_target(prices, horizon)

    usable = feats.dropna().index
    if len(usable) < min_train + horizon + 20:
        return {"error": "not enough history to backtest"}

    positions = [prices.index.get_loc(ix) for ix in usable]
    start_pos = max(min_train, positions[0] + min_train // 2)
    last_pos = len(prices) - horizon - 1
    if test_days:
        start_pos = max(start_pos, last_pos - test_days)
    if start_pos >= last_pos:
        return {"error": "not enough history to backtest"}

    models = default_models()
    records: list[dict[str, Any]] = []
    fit_counter = 0

    for pos in range(start_pos, last_pos + 1):
        t_index = prices.index[pos]
        if t_index not in feats.index or feats.loc[t_index].isna().any():
            continue

        actual = target.loc[t_index]
        if pd.isna(actual):
            continue

        # Training rows must have a *known* outcome as of t: their target
        # window has to have closed on or before t.
        train_mask = (feats.index <= prices.index[pos - horizon])
        Xtr = feats[train_mask].join(target.rename("y")).dropna()
        if len(Xtr) < 50:
            continue

        if fit_counter % refit_every == 0:
            X = Xtr.drop(columns="y").to_numpy()
            y = Xtr["y"].to_numpy()
            for m in models:
                if m.needs_fit:
                    m.fit(X, y)
        fit_counter += 1

        hist = prices.iloc[: pos + 1]
        x = feats.loc[t_index].to_numpy(dtype=float)
        last_close = float(prices.iloc[pos])
        actual_price = float(prices.iloc[pos + horizon])

        row: dict[str, Any] = {"date": t_index, "last_close": last_close,
                               "actual": actual_price}
        for m in models:
            pred_lr = m.predict_logret(x, hist, horizon)
            row[m.name] = last_close * float(np.exp(pred_lr))
        records.append(row)

    if not records:
        return {"error": "no evaluable windows"}

    df = pd.DataFrame(records).set_index("date")
    names = [m.name for m in models]
    naive_mse = float(np.mean((df["naive"] - df["actual"]) ** 2))

    scores: dict[str, dict[str, float]] = {}
    for name in names:
        err = df[name] - df["actual"]
        mse = float(np.mean(err ** 2))
        moved_pred = np.sign(df[name] - df["last_close"])
        moved_true = np.sign(df["actual"] - df["last_close"])
        # A model that always predicts "no change" has no directional opinion;
        # scoring it as correct would flatter it.
        has_view = moved_pred != 0
        direction = (float(np.mean(moved_pred[has_view] == moved_true[has_view]) * 100)
                     if has_view.any() else float("nan"))

        scores[name] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(mse)),
            "mape": float(np.mean(np.abs(err / df["actual"])) * 100),
            "direction_pct": direction,
            # Skill score: >0 beats the naive walk, <0 is worse than doing nothing.
            "skill_vs_naive": float(1 - mse / naive_mse) if naive_mse else 0.0,
        }

    return {
        "horizon": horizon,
        "windows": len(df),
        "start": df.index[0].date().isoformat(),
        "end": df.index[-1].date().isoformat(),
        "scores": scores,
        "frame": df,
    }


# --------------------------------------------------------------------------
# Forward projection
# --------------------------------------------------------------------------
def project(prices: pd.Series, volumes: pd.Series | None = None,
            horizon: int = HORIZON, model_name: str = "ridge") -> dict[str, Any]:
    """Fit on everything and project forward, with an uncertainty band.

    The band is what actually matters. A point forecast on a stock is close to
    meaningless on its own; the spread of plausible outcomes is the honest
    picture, and it is wide.
    """
    feats = build_features(prices, volumes)
    target = make_target(prices, horizon)
    train = feats.join(target.rename("y")).dropna()
    if len(train) < 80:
        return {"error": "not enough history to fit"}

    model = next((m for m in default_models() if m.name == model_name), None)
    if model is None:
        return {"error": f"unknown model {model_name!r}"}

    if model.needs_fit:
        model.fit(train.drop(columns="y").to_numpy(), train["y"].to_numpy())

    latest = feats.dropna()
    if latest.empty:
        return {"error": "no complete feature row"}

    x = latest.iloc[-1].to_numpy(dtype=float)
    last_close = float(prices.iloc[-1])
    pred_lr = model.predict_logret(x, prices, horizon)
    point = last_close * float(np.exp(pred_lr))

    # Uncertainty from recent daily volatility, widening with the square root
    # of time - the standard random-walk assumption.
    daily_vol = float(np.diff(np.log(prices.values))[-60:].std())
    sigma = daily_vol * np.sqrt(horizon)

    def band(z: float) -> tuple[float, float]:
        return (last_close * float(np.exp(pred_lr - z * sigma)),
                last_close * float(np.exp(pred_lr + z * sigma)))

    lo80, hi80 = band(1.2816)
    lo95, hi95 = band(1.9600)

    last_date = prices.index[-1]
    target_date = last_date + pd.tseries.offsets.BDay(horizon)

    return {
        "model": model_name,
        "horizon": horizon,
        "as_of": last_date.date().isoformat(),
        "target_date": target_date.date().isoformat(),
        "last_close": round(last_close, 2),
        "point": round(point, 2),
        "change_pct": round((point / last_close - 1) * 100, 2),
        "lo80": round(lo80, 2), "hi80": round(hi80, 2),
        "lo95": round(lo95, 2), "hi95": round(hi95, 2),
        "band_width_pct": round((hi80 - lo80) / last_close * 100, 1),
        "daily_vol_pct": round(daily_vol * 100, 2),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def load_symbol(symbol: str, years: int = config.HISTORY_YEARS
                ) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    with SessionLocal() as s:
        series = services.get_series(s, symbol.upper(),
                                     config.history_start(years), config.today())
    if not series or len(series["closes"]) < 120:
        return None, None
    idx = pd.to_datetime(series["dates"])
    return (pd.Series(series["closes"], index=idx, name=symbol),
            pd.Series(series["volumes"], index=idx, name="volume"))


def run_all(horizon: int = HORIZON, symbols: list[str] | None = None
            ) -> dict[str, Any]:
    """Backtest and project every top-10 company."""
    if symbols is None:
        with SessionLocal() as s:
            symbols = [c["symbol"]
                       for c in services.get_top_companies(s, "market_cap",
                                                           config.TOP_N)]

    results: dict[str, Any] = {}
    for sym in symbols:
        prices, volumes = load_symbol(sym)
        if prices is None:
            continue
        bt = backtest(prices, volumes, horizon=horizon)
        if "error" in bt:
            continue
        results[sym] = {
            "backtest": {k: v for k, v in bt.items() if k != "frame"},
            "projection": project(prices, volumes, horizon=horizon),
        }
    return {"horizon": horizon, "symbols": list(results), "results": results}


def aggregate(all_results: dict[str, Any]) -> dict[str, Any]:
    """Average each model's scores across companies - the headline verdict."""
    names = ["naive", "drift", "sma20", "ridge", "gbt"]
    agg: dict[str, dict[str, float]] = {}
    for name in names:
        rows = [r["backtest"]["scores"][name]
                for r in all_results["results"].values()
                if name in r["backtest"]["scores"]]
        if not rows:
            continue
        dirs = [r["direction_pct"] for r in rows
                if not (r["direction_pct"] is None or np.isnan(r["direction_pct"]))]
        agg[name] = {
            "mape": float(np.mean([r["mape"] for r in rows])),
            "rmse": float(np.mean([r["rmse"] for r in rows])),
            # The naive model never predicts a direction, so it has none.
            "direction_pct": float(np.mean(dirs)) if dirs else float("nan"),
            "skill_vs_naive": float(np.mean([r["skill_vs_naive"] for r in rows])),
            "beat_naive_count": int(sum(1 for r in rows
                                        if r["skill_vs_naive"] > 0)),
            "n": len(rows),
        }
    return agg


# --------------------------------------------------------------------------
# On-disk cache
# --------------------------------------------------------------------------
def cache_path(horizon: int) -> str:
    import os
    return os.path.join(config.DATA_DIR, f"forecast_{horizon}d.json")


def _json_safe(obj: Any) -> Any:
    """Replace NaN/Infinity with None, recursively.

    Python's json writes bare `NaN`, which is NOT valid JSON: the browser's
    JSON.parse rejects the whole document, so one non-finite number silently
    blanks every chart on the page. Nulls survive the round trip.
    """
    import math

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_results(res: dict[str, Any]) -> str:
    """Persist a run. The backtest takes minutes, so the web page reads this
    rather than recomputing inside a request."""
    import json
    import os

    payload = dict(res)
    payload["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    payload["aggregate"] = aggregate(res)
    path = cache_path(res["horizon"])
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        # allow_nan=False turns a stray non-finite value into a loud error here
        # rather than a silently broken page later.
        json.dump(_json_safe(payload), fh, indent=2, default=str,
                  allow_nan=False)
    return path


def load_results(horizon: int = HORIZON) -> dict[str, Any] | None:
    """The last saved run, or None if it has never been generated."""
    import json

    try:
        with open(cache_path(horizon), encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, ValueError):
        return None


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def print_report(res: dict[str, Any]) -> None:
    line = "=" * 92
    h = res["horizon"]
    print(line)
    print(f"  {h}-TRADING-DAY FORECAST BACKTEST - walk-forward, out of sample")
    print(line)

    agg = aggregate(res)
    print(f"\nAVERAGED ACROSS {len(res['symbols'])} COMPANIES\n")
    print(f"  {'MODEL':<10}{'MAPE':>9}{'RMSE':>10}{'DIRECTION':>12}"
          f"{'SKILL vs NAIVE':>17}{'BEAT NAIVE':>13}")
    print("  " + "-" * 88)
    for name, s in agg.items():
        d = ("       n/a" if np.isnan(s["direction_pct"])
             else f"{s['direction_pct']:>10.1f}%")
        print(f"  {name:<10}{s['mape']:>8.2f}%{s['rmse']:>10.2f}"
              f"{d}{s['skill_vs_naive']:>16.3f}"
              f"{s['beat_naive_count']:>8}/{s['n']}")

    print("\n  MAPE      average error as a % of the actual price")
    print("  DIRECTION how often the predicted up/down move was right (50% = chance)")
    print("  SKILL     1 - MSE/MSE_naive. Above 0 beats the random walk; below 0")
    print("            is worse than assuming the price does not change.")

    print("\n\nPER COMPANY (skill vs naive - positive means it beat the walk)\n")
    names = [n for n in ("drift", "sma20", "ridge", "gbt") if n in agg]
    print(f"  {'SYM':<7}" + "".join(f"{n:>10}" for n in names)
          + f"{'BEST':>10}{'DIRECTION':>12}")
    print("  " + "-" * 88)
    for sym, r in res["results"].items():
        sc = r["backtest"]["scores"]
        cells = "".join(f"{sc[n]['skill_vs_naive']:>10.3f}" for n in names)
        best = max(names, key=lambda n: sc[n]["skill_vs_naive"])
        best_label = best if sc[best]["skill_vs_naive"] > 0 else "naive"
        print(f"  {sym:<7}{cells}{best_label:>10}"
              f"{sc[best]['direction_pct']:>11.1f}%")

    print("\n\nFORWARD PROJECTION (ridge, for illustration)\n")
    print(f"  {'SYM':<7}{'LAST':>10}{'POINT':>10}{'CHANGE':>9}"
          f"{'80% RANGE':>22}{'BAND':>9}")
    print("  " + "-" * 88)
    for sym, r in res["results"].items():
        p = r["projection"]
        if "error" in p:
            continue
        rng = f"${p['lo80']:,.0f} - ${p['hi80']:,.0f}"
        print(f"  {sym:<7}{p['last_close']:>10,.2f}{p['point']:>10,.2f}"
              f"{p['change_pct']:>8.2f}%{rng:>22}{p['band_width_pct']:>8.1f}%")

    _print_verdict(agg)
    print("\n" + line)


def _print_verdict(agg: dict[str, Any]) -> None:
    print("\n\nWHAT THIS ACTUALLY SHOWS\n")
    beat = [n for n, s in agg.items()
            if n != "naive" and s["skill_vs_naive"] > 0]
    ranked = [(n, s["direction_pct"]) for n, s in agg.items()
              if n != "naive" and not np.isnan(s["direction_pct"])]
    best_name, best_dir = (max(ranked, key=lambda kv: kv[1])
                           if ranked else ("n/a", float("nan")))

    if not beat:
        print("  No model beat the naive random walk on average. That is the")
        print("  expected result, and it is the honest headline: on daily closing")
        print("  prices, 'tomorrow looks like today' is a hard benchmark.")
    else:
        print(f"  Models with positive average skill: {', '.join(beat)}.")
        print("  Treat a small positive skill score with suspicion - across ~10")
        print("  companies and one period it is well within noise.")

    print(f"\n  Best average directional accuracy: {best_dir:.1f}% ({best_name}).")
    print("  Anything within a few points of 50% is a coin flip. Beating a coin")
    print("  flip on direction is also not the same as making money: transaction")
    print("  costs and the size of the moves you get wrong both matter more.")
    print("\n  The width of the 80% band above is the useful output. It says what")
    print("  range is plausible - which is a real, usable statement, unlike a")
    print("  single predicted price.")
    print("\n  This is coursework, not a trading system. Do not put money on it.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest price forecasting models")
    ap.add_argument("symbol", nargs="?", help="one ticker (default: top 10)")
    ap.add_argument("--horizon", type=int, default=HORIZON,
                    help="trading days ahead (default 5)")
    ap.add_argument("--save", action="store_true",
                    help="write the results to data/ for the web page")
    args = ap.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else None
    res = run_all(horizon=args.horizon, symbols=symbols)
    if not res["results"]:
        raise SystemExit("No data. Run: python ingest.py --years 3")
    print_report(res)

    if args.save:
        if args.symbol:
            print("\nNot saving: --save expects a full run, not one symbol.")
        else:
            print(f"\nSaved {save_results(res)}")


if __name__ == "__main__":
    main()
