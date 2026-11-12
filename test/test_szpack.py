"""The compiled SZpack extension: known limits of the relativistic SZ spectrum."""
import numpy as np
import pytest
from scipy.optimize import brentq

from pymsz import sz_spectra as sp
from pymsz.SZpack_models import _szpack_options, szpack_dn, thermal_table

X = np.array([1., 2., 2.64, 5., 8.])  # avoid the tSZ null (x = 3.83)


def dn(S, x, Te, beta=0., mu=1., method='combo'):
    return szpack_dn(S, [float(v) for v in x], Te, beta, mu, method, _szpack_options(method))


@pytest.mark.parametrize('method', ['combo', 'asymptotic', '3D', 'nonrel'])
def test_low_temperature_limit(szpack, method):
    # Te -> 0: Delta n -> tau (kTe/me c^2) G(x) (x coth(x/2) - 4)
    theta = 0.01 / sp.ME_KEV
    assert np.allclose(dn(szpack, X, 0.01, method=method) / (theta * sp.tsz_nonrel(X)), 1., atol=1e-3)


@pytest.mark.parametrize('Te', [5., 10., 20.])
def test_combo_matches_exact_integration(szpack, Te):
    exact = dn(szpack, X, Te, method='3D')
    assert np.allclose(dn(szpack, X, Te, method='combo'), exact, rtol=1e-4, atol=0)


@pytest.mark.parametrize('Te', [5., 10., 20.])
def test_relativistic_null_shift(szpack, Te):
    # crossover frequency of the relativistic tSZ, fit of Itoh et al. (1998, ApJ 502, 7)
    theta = Te / sp.ME_KEV
    x0 = brentq(lambda x: dn(szpack, [x], Te)[0], 3.5, 4.5)
    assert x0 == pytest.approx(3.830 * (1. + 1.1674 * theta - 0.8533 * theta**2), abs=2e-3)


def test_relativistic_decrement_is_smaller(szpack):
    theta = 10. / sp.ME_KEV
    ratio = dn(szpack, X[:3], 10.) / (theta * sp.tsz_nonrel(X[:3]))
    assert np.all((ratio > 0.9) & (ratio < 0.97))


@pytest.mark.parametrize('mu', [1., -1.])
def test_kinematic_limit(szpack, mu):
    # cold gas moving with beta = v/c along mu (> 0 towards the observer): Delta n = tau beta mu G(x)
    beta = 1e-3
    k = dn(szpack, X, 0.01, beta, mu) - dn(szpack, X, 0.01)
    assert np.allclose(k / (beta * sp.G(X)), mu, rtol=1e-2)


def test_thermal_table(szpack):
    f = [220., 100., 150.]  # any order
    tab = thermal_table(f, [0.01, 10.])
    assert tab.shape == (2, 3)
    assert np.allclose(tab[0], sp.tsz_nonrel(sp.freq_to_x(f)), rtol=1e-3, atol=1e-4)
    assert np.allclose(tab[1], dn(szpack, sp.freq_to_x(f)[[1, 2, 0]], 10.)[[2, 0, 1]] / (10. / sp.ME_KEV))


def test_frequency_range_is_checked(szpack):
    with pytest.raises(ValueError, match='SZpack needs'):
        thermal_table([2.], [5.])
