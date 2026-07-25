# GARCH Method

Volatility forecasting and position sizing as a Claude Code skill. Fork of [milesdeutscher/garchmethod](https://github.com/milesdeutscher/garchmethod) (video 1 of the Quant Series), with packaging fixes, a corrected GARCH recursion, and a comparison harness that charges costs and reports error bars.

The model family is Engle's ARCH / Bollerslev's GARCH — the volatility machinery that won the 2003 Sveriges Riksbank Prize (the economics Nobel) and that real risk desks run daily.

**One sentence:** every other trading tool tries to answer *which way*. This one answers ***how much*** — and how much is the question that decides whether you survive.

---

## Install

In Claude Code:

```
/plugin marketplace add patzoul/garchmethod
/plugin install garch-method@garchmethod
```

Then just ask in plain English: *"what's the vol forecast on BTC-USD"*, *"how big should my SPY position be"*, *"test my strategy with and without vol targeting"* — Claude fires the skill automatically.

No API keys. No accounts. No sudo. Dependencies resolve on first run via `uv` (PEP 723 inline metadata) — nothing to pip-install yourself.

## What the skill does

It answers one question for **any asset**: how violent is this market, and how big should my position be?

- Fits **GARCH(1,1) walk-forward** — parameters re-estimated on an expanding window, every forecast made using only data that existed before that day. Zero lookahead, and there is a test that proves it.
- Produces a 1-day-ahead **volatility forecast** (daily + annualized)
- Classifies the **vol regime**: calm / normal / storm (percentile vs trailing year)
- Converts the forecast into a **position size**: `target_vol / forecast_vol`, capped [0.25x, 2.0x]
- Runs the **honest test** (`compare.py`): the same strategy signals sized two ways — fixed vs vol-targeted — against buy-and-hold, net of trading costs, with a bootstrap confidence interval on the Sharpe difference
- Ships with an **EMA 9/21 crossover demo** strategy; bring your own via a signals CSV (`date, signal in {-1,0,1}`)

It takes **either a ticker** (`--ticker BTC-USD`, fetched via `yfinance`) **or your own CSV** (`--csv my_prices.csv`, just a date + close column).

It's built to **compose**: use it as a sizing layer on a strategy you already have, a standalone risk throttle, or a comparison harness. See [`skills/garch-method/SKILL.md`](skills/garch-method/SKILL.md) for the JSON contract and three worked composition patterns.

**What it will never do:** predict direction. GARCH forecasts the *magnitude* of moves — the skill says this in its own output, every time.

## What this fork changes

Upstream is a solid demo. These are the changes that make it safe to run against real decisions:

**Packaging**
- Scripts moved to `skills/garch-method/scripts/` and referenced via `${CLAUDE_PLUGIN_ROOT}`. Upstream's `SKILL.md` pointed at a repo-root `scripts/` directory, which does not resolve once the plugin is installed — the skill could not find its own code.
- `yfinance` added to every script's PEP 723 dependency block. It was imported but never declared, so under `uv run` every `--ticker` invocation — including the README's own sanity check — died with `yfinance not installed`.
- `pandas` floor raised to 2.2 (`resample("ME")` is a 2.2 API).

**Correctness**
- **Fixed a one-day-stale seed after every refit.** The recursion was restarted from `conditional_volatility[-1]`, which is the variance of the last *in-sample* day, not of the first *out-of-sample* day. It now uses the fitted model's own 1-step-ahead forecast. On synthetic GARCH data the refit-day forecast was off by up to ~6% before the fix and matches `arch` exactly after it. Not lookahead — just wrong.
- Non-convergent fits carry the previous parameters forward instead of silently producing garbage; explosive fits (α+β > 1) are rescaled onto the IGARCH boundary. Both are counted and summarised in one line rather than swallowed by a blanket `warnings.filterwarnings("ignore")`.
- Loader-boundary validation: duplicate dates, non-monotonic dates, non-positive prices and unidentifiable columns are now errors rather than silent corruption. Upstream fell back to "column index 1" as the price column.
- No forecast now means size 0, not size 0.25.

**Honesty of the comparison**
- **`--cost-bps` (default 10)**, charged on `|change in position|`. This matters more than anything else here: the fixed arm only trades on signal flips while the vol-targeted arm re-sizes daily, so its turnover is roughly double (26.7x/yr vs 13.8x/yr on BTC). A zero-cost comparison is not a fair fight.
- **Buy-and-hold as a third column.** On SPY it beats both sizing rules, which is the single most useful thing the harness can tell you.
- **Block-bootstrap 95% CI on the Sharpe difference**, plus `P(no improvement)`. On both BTC and SPY the improvement from vol targeting is *not* significant at 95% once costs are charged — the tool now says so out loud instead of showing a flattering point estimate.
- Diagnostics for how often sizing was pinned at the floor or the cap, and the risk-matched `--target-vol` to use.
- Signal timing convention documented explicitly, and signals outside [-1, 1] or with duplicate dates are rejected.

**Tests** — 24 of them, offline, in `tests/`. The two that matter pin the refit seeding against `arch`'s own forecast, and assert that a one-day signal earns the *following* day's return.

## Results on real data

EMA 9/21 crossover, 10 bps per unit of turnover:

```
BTC-USD, 10.5y, target vol 58%
                    BUY & HOLD    FIXED   VOL-TARGETED
  CAGR %                  63.5     67.6           77.1
  Sharpe                  1.08     1.32            1.5
  Max drawdown %         -83.4    -73.4          -61.6
  Turnover / yr            0.1     13.8           26.7

  Sharpe(vol-targeted) - Sharpe(fixed) = +0.18
  95% CI: [-0.045, +0.435]   P(no improvement) = 0.059
  → NOT significant at 95%.
```

```
SPY, 31.4y, target vol 15%
                    BUY & HOLD    FIXED   VOL-TARGETED
  CAGR %                  11.1      4.7            4.8
  Sharpe                  0.65     0.46           0.45
  Max drawdown %         -55.2    -39.1          -35.7
  Turnover / yr            0.0     10.3           16.4

  Sharpe(vol-targeted) - Sharpe(fixed) = -0.006
  95% CI: [-0.128, +0.126]   P(no improvement) = 0.532
  → NOT significant at 95%.
```

Vol targeting reduces drawdowns in both cases — that part is real and consistent. What it does *not* do, on this evidence, is reliably improve risk-adjusted return once you pay for the turnover. Use the skill for the forecast and the drawdown control; don't expect free Sharpe.

## Running the tests

```bash
uv run --with pytest --with arch --with pandas --with numpy --with matplotlib pytest tests -q
```

## The zero-trust manual path

`garch-method.md` is the original one-shot onboarding prompt. Paste it into Claude Code (agent mode) and it builds the whole skill from scratch in front of you: detects your OS, installs `uv`, writes every file, runs the sanity check.

It fetches the script files from **this fork, pinned to a release tag** — upstream's version pointed at another account's `main`, so it both ignored any local changes and executed whatever happened to be on that branch at run time.

## Pine Script bonus — the Storm Gauge

`pine-script/storm-gauge.pine` — TradingView v5 indicator that puts the framework on a chart: vol bands on price, storm tint, and a corner gauge showing market violence vs your limit, its 1-year percentile, the weather, and — if you enter — your size, in dollars.

Honest label, on the panel itself: the gauge uses *realized* vol (Pine-native math). The walk-forward GARCH *forecast* lives in the skill.

Open TradingView → Pine Editor → paste the `.pine` → Save → Add to Chart. **Use the daily chart.** Inputs: 365 periods/year for crypto, 252 for stocks.

## Credit

- **Model family:** Robert Engle (ARCH, Nobel 2003) and Tim Bollerslev (GARCH, 1986). Read the originals — the math is theirs.
- **Original skill, installer and Pine script:** Miles Deutscher.
- **This fork's fixes:** see *What this fork changes* above.

## License

MIT.
