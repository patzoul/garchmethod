"""
Invariants for the comparison harness.

The headline one is test_costs_hit_the_vol_targeted_arm_harder: the vol-targeted
arm re-sizes every day while the fixed arm only trades on signal flips, so a
zero-cost comparison systematically flatters vol targeting. These tests pin the
fact that costs exist, are charged on turnover, and bite the higher-turnover arm
more.
"""

import numpy as np
import pandas as pd
import pytest

from compare import (ema_crossover_signals, load_signals, perf_stats,
                     run_comparison, sharpe_diff_ci)
from vol_target import size_from_vol, size_series

MIN_TRAIN = 250
REFIT = 21
ARMS = ("buy_hold", "fixed_size", "vol_targeted")


@pytest.fixture(scope="module")
def result(small_prices):
    return run_comparison(small_prices, target_vol=15.0, periods_per_year=252,
                          cost_bps=10.0, min_train=MIN_TRAIN, refit_every=REFIT)


def test_all_three_arms_reported(result):
    stats, curves, _, _ = result
    assert set(stats) == set(ARMS)
    for arm in ARMS:
        assert curves[arm].notna().all()


def test_buy_hold_matches_the_underlying(small_prices, result):
    """The benchmark arm must actually be the asset, not a re-derived curve."""
    _, curves, _, _ = result
    px = small_prices.set_index("date")["close"]
    span = px.loc[curves["date"].iloc[0]:curves["date"].iloc[-1]]
    # buy & hold is entered one bar before the first curve point and pays one
    # entry cost, so allow a little slack; the shape must match within 1%.
    assert curves["buy_hold"].iloc[-1] == pytest.approx(span.iloc[-1] / span.iloc[0], rel=0.01)


def test_costs_hit_the_vol_targeted_arm_harder(small_prices):
    free = run_comparison(small_prices, periods_per_year=252, cost_bps=0.0,
                          min_train=MIN_TRAIN, refit_every=REFIT)
    paid = run_comparison(small_prices, periods_per_year=252, cost_bps=25.0,
                          min_train=MIN_TRAIN, refit_every=REFIT)

    turn_fixed = free[0]["fixed_size"]["annual_turnover_x"]
    turn_volt = free[0]["vol_targeted"]["annual_turnover_x"]
    assert turn_volt > turn_fixed, "vol targeting should re-size more often than the signal flips"

    drag_fixed = free[0]["fixed_size"]["CAGR_pct"] - paid[0]["fixed_size"]["CAGR_pct"]
    drag_volt = free[0]["vol_targeted"]["CAGR_pct"] - paid[0]["vol_targeted"]["CAGR_pct"]
    assert drag_volt > drag_fixed, "higher turnover must pay more in costs"


def test_costs_are_monotonic(small_prices):
    cagrs = [run_comparison(small_prices, periods_per_year=252, cost_bps=c,
                            min_train=MIN_TRAIN, refit_every=REFIT)[0]["vol_targeted"]["CAGR_pct"]
             for c in (0.0, 10.0, 30.0)]
    assert cagrs[0] > cagrs[1] > cagrs[2]


def test_signal_applies_to_the_next_bar(small_prices):
    """
    One-day lookahead is the failure mode this harness exists to avoid.
    A signal that is long on exactly one day must earn the FOLLOWING day's
    return, never that day's own.
    """
    wf_dates = small_prices["date"].iloc[1:].reset_index(drop=True)
    sig = pd.Series(np.zeros(len(wf_dates)))
    day = 400
    sig.iloc[day] = 1.0

    stats, curves, _, _ = run_comparison(
        small_prices, signals=sig, periods_per_year=252, cost_bps=0.0,
        min_train=MIN_TRAIN, refit_every=REFIT)

    px = small_prices["close"].to_numpy(float)
    rets = 100.0 * np.diff(px) / px[:-1]
    total = curves["fixed_size"].iloc[-1] - 1.0
    assert total == pytest.approx(rets[day + 1] / 100.0, rel=1e-6)
    assert total != pytest.approx(rets[day] / 100.0, rel=1e-6)


def test_significance_is_reported_with_an_error_bar(result):
    _, _, sig, _ = result
    assert {"sharpe_difference", "ci95_low", "ci95_high", "prob_no_improvement",
            "verdict", "significant_at_95"} <= set(sig)
    assert sig["ci95_low"] <= sig["sharpe_difference"] <= sig["ci95_high"]


def test_significant_harm_is_not_reported_as_inconclusive():
    """
    A CI entirely below zero means the overlay measurably hurt. Reporting that
    as "not significant" would bury the most actionable result the harness has.
    """
    rng = np.random.default_rng(3)
    base = rng.standard_normal(3000) / 100
    worse = base - 0.0015                       # same shape, materially worse mean
    out = sharpe_diff_ci(worse, base, 252, reps=500)
    assert out["sharpe_difference"] < 0
    assert out["ci95_high"] < 0
    assert out["verdict"] == "harm"
    assert out["significant_at_95"] is True


def test_clear_improvement_is_flagged():
    rng = np.random.default_rng(4)
    base = rng.standard_normal(3000) / 100
    better = base + 0.0015
    out = sharpe_diff_ci(better, base, 252, reps=500)
    assert out["verdict"] == "improvement" and out["significant_at_95"] is True


def test_significance_flags_a_null_difference():
    """Two identical return streams must not read as a significant improvement."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(2000) / 100
    out = sharpe_diff_ci(a, a.copy(), 252, reps=300)
    assert out["sharpe_difference"] == pytest.approx(0.0, abs=1e-9)
    assert not out["significant_at_95"]


def test_diagnostics_expose_the_caps(result):
    _, _, _, diag = result
    for k in ("risk_matched_target_vol_pct", "pct_days_at_min_size",
              "pct_days_at_max_leverage", "cost_bps_per_unit_turnover"):
        assert k in diag
    assert 0 <= diag["pct_days_at_min_size"] <= 100


# ------------------------------------------------------------------- sizing
def test_no_forecast_means_no_position():
    assert size_from_vol(float("nan")) == 0.0
    assert size_from_vol(0.0) == 0.0
    s = size_series(pd.Series([np.nan, 30.0, np.nan]), target_vol_ann=15.0)
    assert s.iloc[0] == 0.0 and s.iloc[2] == 0.0
    assert s.iloc[1] == pytest.approx(0.5)


def test_size_respects_caps():
    assert size_from_vol(1.0, 15.0) == 2.0          # would be 15x, capped
    assert size_from_vol(600.0, 15.0) == 0.25       # would be 0.025x, floored
    assert size_from_vol(600.0, 15.0, min_size=0.0) == pytest.approx(0.025)


# -------------------------------------------------------------------- misc
def test_perf_stats_survives_ruin():
    r = pd.Series([-150.0, 10.0, 10.0])              # wiped out on day one
    out = perf_stats(r, 252)
    assert out["final_equity_x"] == 0.0 and out["max_drawdown_pct"] == -100.0


def test_load_signals_rejects_out_of_range(tmp_path):
    p = tmp_path / "sig.csv"
    pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=5),
                  "signal": [0, 1, 5, 0, 1]}).to_csv(p, index=False)
    with pytest.raises(SystemExit, match=r"\[-1, 1\]"):
        load_signals(str(p), pd.Series(pd.bdate_range("2020-01-01", periods=5)))


def test_ema_signal_is_long_flat(small_prices):
    s = ema_crossover_signals(small_prices["close"])
    assert set(np.unique(s)) <= {0.0, 1.0}
