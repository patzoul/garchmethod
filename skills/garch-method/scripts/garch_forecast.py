# /// script
# requires-python = ">=3.10"
# dependencies = ["arch>=6.0", "pandas>=2.2", "numpy>=1.24", "matplotlib>=3.7", "yfinance>=0.2.40"]
# ///
"""
garch_forecast.py — walk-forward GARCH(1,1) volatility forecasting.

What this does:
  Fits a GARCH(1,1) model walk-forward (no lookahead) and produces a
  1-day-ahead volatility forecast for every day in the sample.

What this does NOT do:
  Predict direction. GARCH forecasts the MAGNITUDE of moves, not which
  way they go. Every output of this module carries that note on purpose.

Usage:
  uv run garch_forecast.py --csv prices.csv            # date + close columns
  uv run garch_forecast.py --ticker BTC-USD            # via yfinance (needs internet)
  uv run garch_forecast.py --csv prices.csv --json     # machine-readable output
"""

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

TRADING_DAYS_CRYPTO = 365
TRADING_DAYS_EQUITY = 252
MIN_TRAIN = 500          # days of history before first forecast
REFIT_EVERY = 21         # re-estimate params every N days (walk-forward)
REGIME_LOOKBACK = 365    # window for vol percentile / regime classification
# alpha + beta == 1 is IGARCH (the RiskMetrics EWMA case) — common on crypto and
# perfectly usable. Only persistence STRICTLY above 1 makes the recursion explode.
MAX_PERSISTENCE = 1.0

HONESTY_NOTE = "GARCH forecasts magnitude (volatility), not direction. It tells you how violent tomorrow is likely to be — not which way it goes."

DATE_ALIASES = ("date", "time", "timestamp", "datetime")
PRICE_ALIASES = ("close", "adj close", "adj_close", "adjclose", "price", "priceusd", "px_last")


class DataQualityError(ValueError):
    """Raised when the input price series is not fit to backtest on."""


def validate_prices(out: pd.DataFrame, source: str, strict: bool = True) -> pd.DataFrame:
    """
    Loader-boundary checks. A backtest is only as trustworthy as the series
    underneath it, so the failure modes that silently corrupt results get
    caught here rather than showing up later as a suspiciously good Sharpe.
    """
    problems, notes = [], []

    if out["date"].duplicated().any():
        dupes = out.loc[out["date"].duplicated(), "date"]
        problems.append(f"{len(dupes)} duplicate dates (first: {dupes.iloc[0].date()})")
    if not out["date"].is_monotonic_increasing:
        problems.append("dates are not monotonically increasing after sorting")
    if len(out) < MIN_TRAIN + 11:
        problems.append(f"only {len(out)} usable rows; need at least {MIN_TRAIN + 11}")
    if (out["close"] <= 0).any():
        problems.append("non-positive close prices")

    gaps = out["date"].diff().dt.days.dropna()
    if len(gaps) and gaps.max() > 10:
        big = gaps[gaps > 10]
        notes.append(f"{len(big)} gap(s) longer than 10 calendar days (max {int(gaps.max())}d)")

    stale = out["close"].diff() == 0
    if stale.any():
        run = int(stale.groupby((~stale).cumsum()).cumsum().max())
        if run >= 5:
            notes.append(f"run of {run} consecutive unchanged closes (stale or illiquid data)")

    rets = out["close"].pct_change().dropna()
    extreme = int((rets.abs() > 0.5).sum())
    if extreme:
        notes.append(f"{extreme} daily move(s) beyond +/-50% (check for splits or bad ticks)")

    for note in notes:
        print(f"  ! data warning [{source}]: {note}", file=sys.stderr)
    if problems:
        msg = f"price series [{source}] failed validation:\n    - " + "\n    - ".join(problems)
        if strict:
            raise DataQualityError(msg)
        print(f"  ! {msg}", file=sys.stderr)
    return out


def load_prices(csv=None, ticker=None, strict: bool = True):
    """Load a price series from CSV (date + close) or yfinance."""
    if csv:
        df = pd.read_csv(csv)
        cols = {str(c).lower().strip(): c for c in df.columns}
        date_col = next((cols[k] for k in DATE_ALIASES if k in cols), None)
        px_col = next((cols[k] for k in PRICE_ALIASES if k in cols), None)
        if date_col is None or px_col is None:
            raise DataQualityError(
                f"{csv}: could not identify date/close columns. Found {list(df.columns)}; "
                f"expected one of {DATE_ALIASES} and one of {PRICE_ALIASES}. "
                "Rename the columns rather than relying on position — guessing by column "
                "index is how you end up backtesting the volume column.")
        out = df[[date_col, px_col]].copy()
        out.columns = ["date", "close"]
        source = str(csv)
    elif ticker:
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("yfinance not installed. Use --csv, or: uv pip install yfinance")
        try:
            data = yf.download(ticker, period="max", auto_adjust=True,
                               progress=False, multi_level_index=False)
        except TypeError:  # older yfinance without multi_level_index
            data = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        if data is None or len(data) == 0:
            sys.exit(f"yfinance returned no rows for {ticker!r}. Check the symbol, or use --csv.")
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        out = data[["Date", "Close"]].copy()
        out.columns = ["date", "close"]
        source = str(ticker)
    else:
        sys.exit("Provide --csv or --ticker")

    out["date"] = pd.to_datetime(out["date"])
    if getattr(out["date"].dt, "tz", None) is not None:
        out["date"] = out["date"].dt.tz_localize(None)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna().sort_values("date").reset_index(drop=True)
    out = out[out["close"] > 0].reset_index(drop=True)
    return validate_prices(out, source, strict=strict)


def _fit_params(rets_slice: np.ndarray, prev, diag: dict = None):
    """
    Fit GARCH(1,1) on strictly prior data and return
    (mu, omega, alpha, beta, sigma2_next), where sigma2_next is the model's own
    1-step-ahead variance forecast for the first day NOT in the fit window.

    Falls back to the previous fit if the optimiser fails to converge, and caps
    persistence so a bad fit cannot make the recursion explode. Fit anomalies
    are counted into `diag` and summarised once, rather than logged per refit.
    """
    from arch import arch_model

    diag = diag if diag is not None else {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(rets_slice, vol="GARCH", p=1, q=1,
                         mean="Constant", dist="t").fit(disp="off", show_warning=False)

    if getattr(res, "convergence_flag", 0) != 0:
        if prev is not None:
            diag["non_converged"] = diag.get("non_converged", 0) + 1
            return prev
        raise RuntimeError(
            f"GARCH fit failed to converge on the first window (n={len(rets_slice)}) and "
            "there are no earlier parameters to fall back on. Check the price series.")

    p = res.params
    mu, omega, alpha, beta = (float(p["mu"]), float(p["omega"]),
                              float(p["alpha[1]"]), float(p["beta[1]"]))

    diag["fits"] = diag.get("fits", 0) + 1
    persistence = alpha + beta
    if persistence >= MAX_PERSISTENCE - 1e-6:
        diag["at_igarch_boundary"] = diag.get("at_igarch_boundary", 0) + 1
    if persistence > MAX_PERSISTENCE:
        # strictly explosive: rescale onto the IGARCH boundary
        scale = MAX_PERSISTENCE / persistence
        alpha, beta = alpha * scale, beta * scale
        diag["explosive"] = diag.get("explosive", 0) + 1

    # The model's variance forecast for the next day — i.e. the first day not in
    # the fit window. Seeding the recursion with conditional_volatility[-1]
    # instead would be one day stale and would bias every post-refit forecast.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sigma2_next = float(res.forecast(horizon=1, reindex=False).variance.iloc[0, 0])

    return mu, omega, alpha, beta, sigma2_next


def _report_fit_diagnostics(diag: dict):
    """One summary line per run, not one per refit."""
    fits = diag.get("fits", 0)
    if not fits:
        return
    bits = []
    if diag.get("at_igarch_boundary"):
        bits.append(f"{diag['at_igarch_boundary']}/{fits} fits hit the IGARCH boundary "
                    "(alpha+beta=1, the EWMA case — expected on high-persistence series)")
    if diag.get("explosive"):
        bits.append(f"{diag['explosive']} explosive fit(s) rescaled")
    if diag.get("non_converged"):
        bits.append(f"{diag['non_converged']} non-converged fit(s) carried previous params")
    if bits:
        print("  GARCH fit notes: " + "; ".join(bits), file=sys.stderr)


def walkforward_garch(prices: pd.DataFrame, periods_per_year: int = TRADING_DAYS_CRYPTO,
                      min_train: int = MIN_TRAIN, refit_every: int = REFIT_EVERY) -> pd.DataFrame:
    """
    Walk-forward GARCH(1,1). For each day t >= min_train, forecast the vol of
    day t+1 using ONLY data available at the close of day t.

    Params are re-estimated every `refit_every` days on an expanding window.
    Between refits, the GARCH recursion is rolled forward with the last fitted
    params — still zero lookahead, because params were estimated on strictly
    prior data.

    Returns a DataFrame indexed like `prices` with:
      ret          — daily % return
      fcast_vol    — 1-day-ahead conditional vol forecast (daily, %)
      fcast_vol_ann— annualized forecast vol (%)
      vol_pctile   — percentile of today's forecast vs trailing REGIME_LOOKBACK
      regime       — calm / normal / storm
    """
    px = prices["close"].to_numpy(dtype=float)
    rets = 100.0 * np.diff(px) / px[:-1]           # daily % returns, scaled for arch
    n = len(rets)
    if n < min_train + 10:
        sys.exit(f"Need at least {min_train + 10} days of prices; got {n + 1}.")

    fcast_var = np.full(n, np.nan)                  # forecast of NEXT day's variance, made at t
    params = None
    sigma2 = None                                   # Var(day t), entering iteration t
    diag = {}

    for t in range(min_train, n):
        if (t - min_train) % refit_every == 0:
            # fit on rets[0 .. t-1]; the fit's 1-step forecast IS Var(day t)
            params = _fit_params(rets[:t], params, diag)
            sigma2 = params[4]
        mu, omega, alpha, beta = params[:4]
        eps = rets[t] - mu                           # observed at the close of day t
        sigma2 = omega + alpha * eps ** 2 + beta * sigma2   # Var(day t) -> Var(day t+1)
        fcast_var[t] = sigma2                        # made at close of t, for day t+1

    _report_fit_diagnostics(diag)

    out = prices.iloc[1:].copy().reset_index(drop=True)
    out["ret"] = rets
    out["fcast_vol"] = np.sqrt(fcast_var)                                # daily %
    out["fcast_vol_ann"] = out["fcast_vol"] * np.sqrt(periods_per_year)  # annualized %
    pct = out["fcast_vol"].rolling(REGIME_LOOKBACK, min_periods=90).apply(
        lambda w: (w.iloc[:-1] < w.iloc[-1]).mean() * 100 if len(w) > 1 else np.nan, raw=False)
    out["vol_pctile"] = pct
    out["regime"] = pd.cut(out["vol_pctile"], bins=[-1, 33, 67, 101],
                           labels=["calm", "normal", "storm"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--ticker")
    ap.add_argument("--periods-per-year", type=int, default=TRADING_DAYS_CRYPTO,
                    help="365 for crypto (default), 252 for stocks")
    ap.add_argument("--json", action="store_true", help="print latest forecast as JSON")
    ap.add_argument("--out-csv", help="write full walk-forward series to CSV")
    ap.add_argument("--no-strict-data", action="store_true",
                    help="downgrade fatal data-quality problems to warnings")
    args = ap.parse_args()

    prices = load_prices(csv=args.csv, ticker=args.ticker, strict=not args.no_strict_data)
    res = walkforward_garch(prices, periods_per_year=args.periods_per_year)
    latest = res.dropna(subset=["fcast_vol"]).iloc[-1]
    regime = latest["regime"]

    payload = {
        "asset": args.ticker or args.csv,
        "as_of": str(latest["date"].date()),
        "forecast_vol_daily_pct": round(float(latest["fcast_vol"]), 3),
        "forecast_vol_annualized_pct": round(float(latest["fcast_vol_ann"]), 1),
        "vol_percentile_1y": round(float(latest["vol_pctile"]), 1) if pd.notna(latest["vol_pctile"]) else None,
        "regime": str(regime) if pd.notna(regime) else None,
        "note": HONESTY_NOTE,
    }
    if args.out_csv:
        res.to_csv(args.out_csv, index=False)
        payload["series_csv"] = args.out_csv
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"\n  {payload['asset']} — as of {payload['as_of']}")
        print(f"  1-day vol forecast : {payload['forecast_vol_daily_pct']}% daily "
              f"({payload['forecast_vol_annualized_pct']}% annualized)")
        print(f"  vol percentile (1y): {payload['vol_percentile_1y']}")
        print(f"  regime             : {(payload['regime'] or 'unknown').upper()}")
        print(f"\n  ⚠ {HONESTY_NOTE}\n")


if __name__ == "__main__":
    main()
