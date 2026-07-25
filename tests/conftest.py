import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "garch-method" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def make_garch_prices(n=900, omega=0.05, alpha=0.10, beta=0.87, seed=7, start=100.0):
    """
    Deterministic synthetic GARCH(1,1) price path with real volatility
    clustering. Offline by design — tests must never depend on yfinance.
    """
    rng = np.random.default_rng(seed)
    sigma2 = omega / (1 - alpha - beta)
    rets = np.empty(n)
    for i in range(n):
        eps = rng.standard_normal() * np.sqrt(sigma2)
        rets[i] = eps
        sigma2 = omega + alpha * eps ** 2 + beta * sigma2
    close = start * np.exp(np.cumsum(rets / 100.0))
    return pd.DataFrame({"date": pd.bdate_range("2018-01-01", periods=n), "close": close})


@pytest.fixture(scope="session")
def prices():
    return make_garch_prices()


@pytest.fixture(scope="session")
def small_prices():
    """Just enough rows to walk forward with a short warm-up, for fast tests."""
    return make_garch_prices(n=600)
