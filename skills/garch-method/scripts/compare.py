# /// script
# requires-python = ">=3.10"
# dependencies = ["arch>=6.0", "pandas>=2.2", "numpy>=1.24", "matplotlib>=3.7", "yfinance>=0.2.40"]
# ///
"""
compare.py — the honest test. Same strategy, two position-sizing rules.

Runs any signal series twice:
  A) FIXED SIZE   — every trade at 1x
  B) VOL-TARGETED — every trade sized by the GARCH forecast (target_vol / forecast_vol)

Same entries. Same exits. The ONLY difference is how much.
Then it shows both equity curves and the numbers, side by side, against
buy-and-hold — with trading costs charged and an error bar on the difference.

Why costs are not optional here:
  the fixed arm only trades when the signal flips; the vol-targeted arm
  re-sizes EVERY day. Its turnover is roughly double. Comparing them at zero
  cost is not a fair fight, and the gap it shows is partly an artefact.
  --cost-bps charges |change in position| x cost on both arms.

Signal timing convention (important):
  signal on date D means "known at the close of D", and it is applied to the
  return of the NEXT bar. If your CSV instead records the position you HELD
  during D, shift it forward by one row before passing it in, or you will be
  backtesting one day of lookahead.

Built-in demo strategy: EMA 9/21 crossover, long/flat.
Bring your own: --signals my_signals.csv (columns: date, signal in {-1, 0, 1})

Usage:
  uv run compare.py --csv prices.csv                          # EMA 9/21 demo
  uv run compare.py --csv prices.csv --target-vol 15 --cost-bps 10
  uv run compare.py --csv prices.csv --signals mine.csv       # your own strategy
  uv run compare.py --csv prices.csv --chart equity.png
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from garch_forecast import load_prices, walkforward_garch, HONESTY_NOTE
from vol_target import size_series, MAX_LEVERAGE, MIN_SIZE

DEFAULT_COST_BPS = 10.0     # round-trip-ish per unit of position turned over
BOOTSTRAP_REPS = 2000
BOOTSTRAP_BLOCK = 20        # days; preserves vol clustering in the resample


# ---------------------------------------------------------------- strategies
def ema_crossover_signals(close: pd.Series, fast: int = 9, slow: int = 21) -> pd.Series:
    """EMA fast/slow crossover, long/flat. Signal known at close of day t."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    return (ema_f > ema_s).astype(float)  # 1 = long, 0 = flat


def load_signals(path: str, dates: pd.Series) -> pd.Series:
    """
    Load a date,signal CSV. Signal on date D is taken as KNOWN AT THE CLOSE of D
    and is applied to the next bar's return — see the module docstring.
    """
    df = pd.read_csv(path)
    cols = {str(c).lower().strip(): c for c in df.columns}
    if "date" not in cols or "signal" not in cols:
        sys.exit(f"{path}: needs 'date' and 'signal' columns; found {list(df.columns)}")
    df = df.rename(columns={cols["date"]: "date", cols["signal"]: "signal"})
    df["date"] = pd.to_datetime(df["date"])
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
    if df["signal"].abs().max() > 1:
        sys.exit(f"{path}: signals must be in [-1, 1]; found max |signal| = "
                 f"{df['signal'].abs().max()}. Scale them or use --target-vol for sizing.")
    if df["date"].duplicated().any():
        sys.exit(f"{path}: duplicate dates in the signal file.")
    merged = pd.DataFrame({"date": dates}).merge(df[["date", "signal"]], on="date", how="left")
    matched = merged["signal"].notna().sum()
    if matched == 0:
        sys.exit(f"{path}: no signal dates overlap the price series.")
    if matched < 0.5 * len(dates):
        print(f"  ! only {matched}/{len(dates)} price dates have a signal; the rest are "
              "forward-filled from the last one", file=sys.stderr)
    return merged["signal"].ffill().fillna(0.0).clip(-1, 1)


# ------------------------------------------------------------------- metrics
def perf_stats(daily_ret: pd.Series, periods_per_year: int) -> dict:
    """`daily_ret` is in percent. Returns a dict of headline stats."""
    r = daily_ret.dropna() / 100.0
    if len(r) == 0:
        return {}
    equity = (1 + r).cumprod()
    if (equity <= 0).any():
        wiped = equity.le(0).idxmax()
        return {"CAGR_pct": None, "ann_vol_pct": None, "sharpe": None,
                "max_drawdown_pct": -100.0, "final_equity_x": 0.0,
                "ruined_at_index": int(wiped)}
    yrs = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year) / ann_vol if ann_vol > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    return {
        "CAGR_pct": round(100 * cagr, 1),
        "ann_vol_pct": round(100 * ann_vol, 1),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(100 * dd, 1),
        "final_equity_x": round(float(equity.iloc[-1]), 2),
    }


def worst_month(daily_ret: pd.Series, dates: pd.Series) -> float:
    r = pd.Series(daily_ret.values / 100.0, index=pd.to_datetime(dates.values))
    m = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return round(100 * m.min(), 1)


def sharpe_diff_ci(a: np.ndarray, b: np.ndarray, periods_per_year: int,
                   reps: int = BOOTSTRAP_REPS, block: int = BOOTSTRAP_BLOCK,
                   seed: int = 0) -> dict:
    """
    Moving-block bootstrap CI for Sharpe(a) - Sharpe(b) on paired daily returns.
    Blocks preserve the volatility clustering that makes an iid bootstrap lie.
    The two arms are resampled with the SAME indices, so the pairing survives.
    """
    n = len(a)
    if n < 2 * block:
        return {}

    def sh(x):
        s = x.std()
        return float(x.mean() / s * np.sqrt(periods_per_year)) if s > 0 else np.nan

    rng = np.random.default_rng(seed)
    n_blocks = n // block + 1
    diffs = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        diffs[i] = sh(a[idx]) - sh(b[idx])
    observed = sh(a) - sh(b)
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    # A CI entirely below zero is just as significant as one entirely above it —
    # it means the overlay measurably HURT. Reporting that as "not significant"
    # would bury the most actionable result the harness can produce.
    verdict = "improvement" if lo > 0 else "harm" if hi < 0 else "inconclusive"
    return {
        "sharpe_difference": round(observed, 3),
        "ci95_low": round(lo, 3),
        "ci95_high": round(hi, 3),
        "prob_no_improvement": round(float((diffs <= 0).mean()), 3),
        "verdict": verdict,
        "significant_at_95": verdict != "inconclusive",
    }


# ------------------------------------------------------------------ backtest
def run_comparison(prices: pd.DataFrame, signals: pd.Series = None,
                   target_vol: float = 15.0, periods_per_year: int = 365,
                   cost_bps: float = DEFAULT_COST_BPS,
                   max_leverage: float = MAX_LEVERAGE, min_size: float = MIN_SIZE,
                   min_train: int = None, refit_every: int = None):
    """
    Timing discipline (zero lookahead):
      signal known at close of t  ->  applied to return of t+1
      vol forecast made at close of t (for t+1)  ->  sizes the t+1 position

    Costs: cost_bps is charged on |position(t) - position(t-1)|, so an arm that
    re-sizes daily pays for it. Buy-and-hold is included as the benchmark that
    both arms have to beat to have earned their complexity.
    """
    wf_kwargs = {}
    if min_train is not None:
        wf_kwargs["min_train"] = min_train
    if refit_every is not None:
        wf_kwargs["refit_every"] = refit_every
    wf = walkforward_garch(prices, periods_per_year=periods_per_year, **wf_kwargs)
    close = wf["close"]

    sig = ema_crossover_signals(close) if signals is None else signals.reset_index(drop=True)

    next_ret = wf["ret"].shift(-1)                       # return of day t+1
    mult = size_series(wf["fcast_vol_ann"], target_vol,  # forecast made at t, for t+1
                       max_leverage=max_leverage, min_size=min_size)

    valid = wf["fcast_vol"].notna() & next_ret.notna()
    dates = wf.loc[valid, "date"].reset_index(drop=True)
    nr = next_ret[valid].reset_index(drop=True)

    positions = {
        "buy_hold": pd.Series(np.ones(len(nr))),
        "fixed_size": sig[valid].reset_index(drop=True),
        "vol_targeted": (sig * mult)[valid].reset_index(drop=True),
    }

    cost = cost_bps / 10000.0
    stats, net_returns, turnovers = {}, {}, {}
    yrs = len(nr) / periods_per_year
    for name, pos in positions.items():
        turn = pos.diff().abs().fillna(pos.iloc[0].__abs__())
        net = pos * nr - 100.0 * turn * cost          # both in percent
        net_returns[name] = net
        turnovers[name] = round(float(turn.sum() / yrs), 1) if yrs > 0 else None
        stats[name] = {**perf_stats(net, periods_per_year),
                       "worst_month_pct": worst_month(net, dates),
                       "annual_turnover_x": turnovers[name]}

    significance = sharpe_diff_ci(net_returns["vol_targeted"].to_numpy() / 100.0,
                                  net_returns["fixed_size"].to_numpy() / 100.0,
                                  periods_per_year)

    # Full-sample realized vol of the fixed arm — a HINT for setting --target-vol
    # so the two arms are risk-matched. Not used by the backtest itself.
    fixed_vol = float((net_returns["fixed_size"] / 100.0).std() * np.sqrt(periods_per_year) * 100)

    curves = pd.DataFrame({
        "date": dates.values,
        **{k: (1 + v.values / 100).cumprod() for k, v in net_returns.items()},
        "regime": wf.loc[valid, "regime"].values,
    })
    diagnostics = {
        "days": int(len(nr)),
        "years": round(yrs, 1),
        "cost_bps_per_unit_turnover": cost_bps,
        "risk_matched_target_vol_pct": round(fixed_vol, 1),
        "pct_days_at_min_size": round(float((mult[valid] <= min_size + 1e-9).mean() * 100), 1),
        "pct_days_at_max_leverage": round(float((mult[valid] >= max_leverage - 1e-9).mean() * 100), 1),
    }
    return stats, curves, significance, diagnostics


def plot_curves(curves: pd.DataFrame, out_path: str, title: str, subtitle: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="white")
    d = pd.to_datetime(curves["date"])
    ax.plot(d, curves["buy_hold"], lw=1.2, color="#b0b0b0", ls="--", label="Buy & hold")
    ax.plot(d, curves["fixed_size"], lw=1.6, color="#888888", label="Fixed size (1x every trade)")
    ax.plot(d, curves["vol_targeted"], lw=1.8, color="#0a7d38", label="Vol-targeted (GARCH sized)")

    storm = (curves["regime"] == "storm").to_numpy()
    ax.fill_between(d, 0, 1, where=storm, transform=ax.get_xaxis_transform(),
                    color="#d62728", alpha=0.07, label="Storm regime")

    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale, net of costs)")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555555")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--ticker")
    ap.add_argument("--signals", help="CSV with date,signal columns for your own strategy")
    ap.add_argument("--target-vol", type=float, default=15.0)
    ap.add_argument("--periods-per-year", type=int, default=365)
    ap.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                    help="bps charged per unit of position turned over (default 10). "
                         "Pass 0 only if you want the frictionless illustration.")
    ap.add_argument("--max-leverage", type=float, default=MAX_LEVERAGE)
    ap.add_argument("--min-size", type=float, default=MIN_SIZE)
    ap.add_argument("--chart", default="equity_comparison.png")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-strict-data", action="store_true",
                    help="downgrade fatal data-quality problems to warnings")
    args = ap.parse_args()

    prices = load_prices(csv=args.csv, ticker=args.ticker, strict=not args.no_strict_data)
    sig = None
    strategy_name = "EMA 9/21 crossover (long/flat)"
    if args.signals:
        wf_dates = prices["date"].iloc[1:].reset_index(drop=True)
        sig = load_signals(args.signals, wf_dates)
        strategy_name = f"custom signals ({args.signals})"

    stats, curves, sig_test, diag = run_comparison(
        prices, signals=sig, target_vol=args.target_vol,
        periods_per_year=args.periods_per_year, cost_bps=args.cost_bps,
        max_leverage=args.max_leverage, min_size=args.min_size)

    subtitle = (f"{diag['years']}y, costs {args.cost_bps:g} bps per unit turnover, "
                f"target vol {args.target_vol:g}%")
    chart = plot_curves(curves, args.chart,
                        f"Same strategy, two sizing rules — {strategy_name}", subtitle)

    payload = {"strategy": strategy_name, "target_vol_pct": args.target_vol,
               "results": stats, "significance": sig_test, "diagnostics": diag,
               "chart": chart, "note": HONESTY_NOTE}
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"\n  {strategy_name} — {subtitle}\n")
    cols = [("BUY & HOLD", "buy_hold"), ("FIXED", "fixed_size"), ("VOL-TARGETED", "vol_targeted")]
    print(f"  {'':22}" + "".join(f"{h:>14}" for h, _ in cols))
    rows = [("CAGR %", "CAGR_pct"), ("Ann vol %", "ann_vol_pct"), ("Sharpe", "sharpe"),
            ("Max drawdown %", "max_drawdown_pct"), ("Worst month %", "worst_month_pct"),
            ("Final equity (x)", "final_equity_x"), ("Turnover / yr", "annual_turnover_x")]
    for label, key in rows:
        cells = "".join(f"{str(stats[k].get(key, '-')):>14}" for _, k in cols)
        print(f"  {label:22}{cells}")

    if sig_test:
        print(f"\n  Sharpe(vol-targeted) - Sharpe(fixed) = {sig_test['sharpe_difference']:+}")
        print(f"  block-bootstrap 95% CI: [{sig_test['ci95_low']:+}, {sig_test['ci95_high']:+}]"
              f"   P(no improvement) = {sig_test['prob_no_improvement']}")
        verdict = {
            "improvement": "  → Vol targeting improved Sharpe, significantly at 95%.",
            "harm": "  → Vol targeting made this strategy WORSE, significantly at 95%. "
                    "The confidence interval lies entirely below zero.",
            "inconclusive": "  → NOT significant at 95%. On this sample, vol targeting has "
                            "not been shown to help.",
        }[sig_test["verdict"]]
        print(verdict)

    print(f"\n  risk-matched --target-vol would be ~{diag['risk_matched_target_vol_pct']}% "
          f"(fixed arm's own realized vol)")
    if diag["pct_days_at_min_size"] > 20 or diag["pct_days_at_max_leverage"] > 20:
        print(f"  ! sizing spent {diag['pct_days_at_min_size']}% of days at the floor and "
              f"{diag['pct_days_at_max_leverage']}% at the cap — the caps, not GARCH, are "
              "driving this result")
    print(f"\n  chart: {chart}")
    print(f"  ⚠ {HONESTY_NOTE}\n")


if __name__ == "__main__":
    main()
