"""
Invariants for the walk-forward GARCH engine.

The one that matters most is test_refit_seed_is_the_models_own_forecast: the
original implementation seeded the post-refit recursion with
`conditional_volatility[-1]`, which is the variance of the LAST IN-SAMPLE day,
not the variance of the first out-of-sample day. That is a one-day-stale seed,
and it biased every forecast for several days after each refit. This test pins
the corrected behaviour and fails loudly if the old form comes back.
"""

import warnings

import numpy as np
import pytest

from conftest import make_garch_prices
import garch_forecast as gf
from garch_forecast import DataQualityError, walkforward_garch, load_prices

MIN_TRAIN = 250
REFIT = 21


def _fit(rets_slice):
    from arch import arch_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return arch_model(rets_slice, vol="GARCH", p=1, q=1,
                          mean="Constant", dist="t").fit(disp="off", show_warning=False)


def test_refit_seed_is_the_models_own_forecast(small_prices):
    px = small_prices["close"].to_numpy(float)
    rets = 100.0 * np.diff(px) / px[:-1]
    wf = walkforward_garch(small_prices, periods_per_year=252,
                           min_train=MIN_TRAIN, refit_every=REFIT)

    for k in range(4):
        t = MIN_TRAIN + k * REFIT                      # a refit day
        res = _fit(rets[:t])
        p = res.params
        mu, om, al, be = (float(p["mu"]), float(p["omega"]),
                          float(p["alpha[1]"]), float(p["beta[1]"]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sigma2_t = float(res.forecast(horizon=1, reindex=False).variance.iloc[0, 0])

        correct = om + al * (rets[t] - mu) ** 2 + be * sigma2_t
        stale = om + al * (rets[t] - mu) ** 2 + be * float(res.conditional_volatility[-1]) ** 2
        got = float(wf["fcast_vol"].iloc[t]) ** 2

        assert got == pytest.approx(correct, rel=1e-6), f"refit day {t} does not use Var(day t)"
        if abs(correct - stale) / correct > 1e-4:
            assert abs(got - stale) > abs(got - correct), (
                f"refit day {t} looks like the old one-day-stale seed")


def test_no_lookahead(prices):
    """Changing the future must not move a single past forecast."""
    cut = 700
    base = walkforward_garch(prices, periods_per_year=252,
                             min_train=MIN_TRAIN, refit_every=REFIT)

    tampered = prices.copy()
    tampered.loc[cut:, "close"] = tampered.loc[cut:, "close"] * 3.0   # violent regime change
    after = walkforward_garch(tampered, periods_per_year=252,
                              min_train=MIN_TRAIN, refit_every=REFIT)

    # wf row t depends on prices rows 0..t+1, so rows up to cut-2 must be identical
    safe = cut - 2
    np.testing.assert_allclose(base["fcast_vol"].to_numpy()[:safe],
                               after["fcast_vol"].to_numpy()[:safe], rtol=1e-12)


def test_forecast_is_positive_and_finite(prices):
    wf = walkforward_garch(prices, periods_per_year=252, min_train=MIN_TRAIN, refit_every=REFIT)
    f = wf["fcast_vol"].dropna()
    assert len(f) == len(prices) - 1 - MIN_TRAIN
    assert np.isfinite(f).all() and (f > 0).all()


def test_annualisation(prices):
    wf = walkforward_garch(prices, periods_per_year=252, min_train=MIN_TRAIN, refit_every=REFIT)
    r = (wf["fcast_vol_ann"] / wf["fcast_vol"]).dropna()
    assert np.allclose(r, np.sqrt(252))


def test_regime_partitions_percentile(prices):
    wf = walkforward_garch(prices, periods_per_year=252, min_train=MIN_TRAIN, refit_every=REFIT)
    sub = wf.dropna(subset=["vol_pctile", "regime"])
    assert (sub.loc[sub["regime"] == "calm", "vol_pctile"] <= 33).all()
    assert (sub.loc[sub["regime"] == "storm", "vol_pctile"] > 67).all()


def test_persistence_never_exceeds_one(monkeypatch):
    """
    alpha+beta == 1 is IGARCH and allowed (that is the RiskMetrics EWMA case).
    Strictly above 1 is explosive and must be rescaled onto the boundary.
    """
    captured = []
    real = gf._fit_params

    def spy(rets_slice, prev, diag=None):
        out = real(rets_slice, prev, diag)
        captured.append(out[2] + out[3])
        return out

    monkeypatch.setattr(gf, "_fit_params", spy)
    gf.walkforward_garch(make_garch_prices(n=500), periods_per_year=252,
                         min_train=200, refit_every=REFIT)
    assert captured and all(p <= gf.MAX_PERSISTENCE + 1e-9 for p in captured)


def test_explosive_fit_is_rescaled():
    diag = {}
    rets = np.diff(np.log(make_garch_prices(n=400)["close"].to_numpy())) * 100
    mu, om, al, be, s2 = gf._fit_params(rets, None, diag)
    assert al + be <= gf.MAX_PERSISTENCE + 1e-9
    assert diag["fits"] == 1 and s2 > 0


# ------------------------------------------------------------------ loader
def test_loader_rejects_duplicate_dates(tmp_path):
    df = make_garch_prices(n=600)
    df = df.iloc[list(range(600)) + [300]]
    p = tmp_path / "dupes.csv"
    df.to_csv(p, index=False)
    with pytest.raises(DataQualityError, match="duplicate dates"):
        load_prices(csv=str(p))


def test_loader_refuses_to_guess_columns(tmp_path):
    df = make_garch_prices(n=600).rename(columns={"close": "volume"})
    p = tmp_path / "mystery.csv"
    df.to_csv(p, index=False)
    with pytest.raises(DataQualityError, match="could not identify"):
        load_prices(csv=str(p))


def test_loader_rejects_too_short(tmp_path):
    p = tmp_path / "short.csv"
    make_garch_prices(n=100).to_csv(p, index=False)
    with pytest.raises(DataQualityError, match="usable rows"):
        load_prices(csv=str(p))


def test_loader_accepts_clean_series(tmp_path):
    p = tmp_path / "clean.csv"
    make_garch_prices(n=600).to_csv(p, index=False)
    out = load_prices(csv=str(p))
    assert list(out.columns) == ["date", "close"]
    assert out["date"].is_monotonic_increasing and (out["close"] > 0).all()
