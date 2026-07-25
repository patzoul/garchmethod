---
name: garch-method
description: Volatility forecasting and position sizing via walk-forward GARCH(1,1). Use whenever the user asks about volatility forecasts, position sizing, "how much should I put on", vol targeting, risk throttling, storm/calm regimes, or wants to test whether vol-targeted sizing improves an existing strategy. Works on any ticker (yfinance) or any CSV with date + close columns. Answers "how much" — never "which way".
---

# GARCH Method — volatility forecasting + position sizing

This skill answers the question retail never asks and every fund asks daily: **how much?**

It does NOT predict direction. GARCH forecasts the *magnitude* of moves — how violent tomorrow is likely to be, not which way it goes. Say this to the user whenever presenting results.

## Where the scripts are

All three scripts live next to this file, in `scripts/`. Set once per session:

```bash
SCRIPTS="${CLAUDE_PLUGIN_ROOT}/skills/garch-method/scripts"
```

If `CLAUDE_PLUGIN_ROOT` is not set (the skill was copied in manually rather than installed as a plugin), `scripts/` is the directory alongside this SKILL.md — resolve it from there. Never assume the current working directory.

Everything runs with `uv run` — dependencies (`arch`, `pandas`, `numpy`, `matplotlib`, `yfinance`) resolve automatically from PEP 723 inline metadata. Nothing to pip-install.

## The three tools

### 1. `garch_forecast.py` — the forecast
Walk-forward GARCH(1,1), zero lookahead. Params are re-estimated every 21 days on an expanding window; after each refit the recursion is seeded with the model's own 1-step-ahead forecast, and between refits it rolls forward using only past data.

```bash
uv run "$SCRIPTS/garch_forecast.py" --csv prices.csv --json
uv run "$SCRIPTS/garch_forecast.py" --ticker BTC-USD --json
```

Output: 1-day-ahead vol forecast (daily + annualized), vol percentile vs trailing year, regime (calm / normal / storm).

### 2. `vol_target.py` — the size
The entire idea: `size = target_vol / forecast_vol`, capped at [0.25x, 2.0x].

```bash
uv run "$SCRIPTS/vol_target.py" --csv prices.csv --target-vol 15 --json
uv run "$SCRIPTS/vol_target.py" --csv prices.csv --target-vol 15 --min-size 0   # true kill-switch
```

Output: position size multiplier. "Run 0.6x your baseline" — that's the answer.

### 3. `compare.py` — the honest test
Runs the same signals twice — fixed size vs vol-targeted — against buy-and-hold, **net of trading costs**, with a bootstrap error bar on the difference.

```bash
uv run "$SCRIPTS/compare.py" --csv prices.csv --target-vol 58 --cost-bps 10 --chart equity.png --json
uv run "$SCRIPTS/compare.py" --csv prices.csv --signals mine.csv
```

Output: CAGR, ann vol, Sharpe, max drawdown, worst month, final equity, annual turnover — for all three arms — plus the equity chart with storm regimes shaded.

## Reading the comparison honestly

These three things decide whether a result means anything. Do not report the table without them.

1. **Costs are not optional.** The fixed arm only trades when the signal flips; the vol-targeted arm re-sizes *every day*, so its turnover is roughly double. At `--cost-bps 0` the comparison flatters vol targeting by construction. The default is 10 bps. If the user wants the frictionless number, show it *next to* a costed one, never alone.
2. **Check the confidence interval, not the point estimate.** The output reports a block-bootstrap 95% CI on `Sharpe(vol-targeted) − Sharpe(fixed)`. If that interval straddles zero, the honest conclusion is "not shown to help on this sample" — say exactly that, even when the point estimate looks good.
3. **Compare against buy-and-hold.** If neither sizing rule beats simply holding the asset, the sizing question is moot and the user should hear so first.

Also watch the diagnostics line: if sizing spent most days pinned at the floor or the cap, the *caps* are driving the result, not GARCH. Re-run with a risk-matched `--target-vol` (the output prints the value to use).

## Signal timing convention

`--signals mine.csv` takes columns `date, signal` with signal in {-1, 0, 1}.

**A signal on date D means "known at the close of D"**, and it is applied to the **next** bar's return. If the user's file instead records the position they *held during* D, it must be shifted forward one row first — otherwise the backtest has a full day of lookahead and the results are worthless. Ask which convention their file uses before running it.

## JSON contract

Every script supports `--json`. Core output shape:

```json
{
  "as_of": "2026-05-23",
  "forecast_vol_annualized_pct": 41.2,
  "vol_percentile_1y": 78.0,
  "regime": "storm",
  "position_size_multiplier": 0.6,
  "note": "GARCH forecasts magnitude (volatility), not direction."
}
```

`compare.py --json` additionally returns `results` (three arms), `significance` (`sharpe_difference`, `ci95_low`, `ci95_high`, `prob_no_improvement`, `significant_at_95`) and `diagnostics` (`risk_matched_target_vol_pct`, `pct_days_at_min_size`, `pct_days_at_max_leverage`).

## Three composition patterns

**A. Sizing layer** — bolt onto any existing strategy. Your strategy decides *if*; this skill decides *how much*. Take the strategy's signal, multiply by `position_size_multiplier`, done.

**B. Risk throttle** — standalone. If `regime == "storm"`, cut exposure to the multiplier regardless of what your signals say. Note the default floor is 0.25x, so this throttles rather than exits; pass `--min-size 0` if you want it to be able to go flat.

**C. Comparison harness** — before trusting any strategy, run it through `compare.py` with realistic costs and check whether vol targeting improves its Sharpe / drawdown *beyond the error bar*. If sizing doesn't help, that is a real finding, not a failed run.

Composes cleanly with regime-direction skills (e.g. Markov-style bull/bear classifiers): their output answers *which way*, this answers *how much*. Multiply the two.

## Defaults & conventions

- Crypto: `--periods-per-year 365` (default). Stocks: `--periods-per-year 252`.
- Target vol: for a risk-matched comparison, set `--target-vol` to the strategy's own realized vol — `compare.py` prints that number for you. The 15% default is an equity-scale figure and is badly mismatched with the 365-day crypto default; set it explicitly.
- Costs: `--cost-bps 10` by default, charged on `|change in position|`. Raise it for illiquid assets or small accounts.
- Data: yfinance ticker (needs internet) or any CSV with date + close columns. The loader rejects duplicate dates, non-positive prices and unidentifiable columns rather than guessing; pass `--no-strict-data` to downgrade those to warnings.
- Minimum history: ~511 daily observations before the first forecast.

## Honesty rules (non-negotiable)

1. Never present GARCH output as a direction call.
2. Never hide the drawdown, worst-month, or turnover numbers when reporting a comparison.
3. Never report a Sharpe improvement without its confidence interval, and never report a comparison at zero cost without saying so.
4. If vol targeting does NOT improve the user's strategy, say so plainly — that result is just as valuable.
