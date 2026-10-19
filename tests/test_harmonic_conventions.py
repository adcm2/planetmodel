"""Coefficients of other conventions, converted to the library's.

The oracle is each convention's own statement: a field evaluated from
its coefficients in the source's basis, written out from the definition
with scipy's associated Legendre functions (which carry the
Condon-Shortley phase), equals the library's synthesis of the converted
coefficients at random points.
"""
from math import factorial

import numpy as np
import pytest
from scipy.special import lpmv

from planetmodel import synthesise
from planetmodel.harmonics import NORMALIZATIONS, from_convention, to_convention

LMAX = 8


def norm(normalization, l, m):
    """The factor before P_l^m(cos theta) cos(m phi), and sin(m phi), in each
    convention, from its definition."""
    ratio = factorial(l - m) / factorial(l + m)
    two = 1.0 if m == 0 else 2.0
    return {"ortho": np.sqrt(two * (2 * l + 1) / (4 * np.pi) * ratio),
            "4pi": np.sqrt(two * (2 * l + 1) * ratio),
            "schmidt": np.sqrt(two * ratio),
            "unnorm": 1.0,
            "woodhouse": np.sqrt((2 * l + 1) / (4 * np.pi) * ratio)}[normalization]


def source_field(C, normalization, csphase, theta, phi):
    """sum C P_l^m (cos, sin)(m phi) N in the source's own convention."""
    x = np.cos(theta)
    out = np.zeros_like(theta)
    for l in range(C.shape[1]):
        for m in range(l + 1):
            P = lpmv(m, l, x) * (1.0 if csphase == -1 else (-1.0) ** m)
            n = norm(normalization, l, m)
            out += n * P * (C[0, l, m] * np.cos(m * phi) + C[1, l, m] * np.sin(m * phi))
    return out


def coefficients(rng, *extra):
    C = rng.standard_normal((2, LMAX + 1, LMAX + 1) + extra)
    for l in range(LMAX + 1):
        C[:, l, l + 1:] = 0.0
    C[1, :, 0] = 0.0
    return C


@pytest.mark.parametrize("csphase", [1, -1])
@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_each_convention_is_its_own_statement(normalization, csphase):
    rng = np.random.default_rng(3)
    C = coefficients(rng)
    theta = np.arccos(rng.uniform(-1.0, 1.0, 200))
    phi = rng.uniform(0.0, 2 * np.pi, 200)
    want = source_field(C, normalization, csphase, theta, phi)
    c = from_convention(C, normalization=normalization, csphase=csphase)
    got = synthesise(c, theta, phi)
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


@pytest.mark.parametrize("csphase", [1, -1])
@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_conversion_round_trips_with_extra_axes(normalization, csphase):
    C = coefficients(np.random.default_rng(4), 3, 2)
    c = from_convention(C, normalization=normalization, csphase=csphase)
    assert c.shape == C.shape
    back = to_convention(c, normalization=normalization, csphase=csphase)
    assert np.allclose(back, C, rtol=1e-13, atol=0.0)


def test_the_library_convention_is_the_identity():
    C = coefficients(np.random.default_rng(5))
    assert np.array_equal(from_convention(C, normalization="ortho", csphase=1), C)


def test_a_4pi_mean_is_its_degree_zero_coefficient():
    # A shape model's C00 is its mean radius.
    C = np.zeros((2, 3, 3))
    C[0, 0, 0] = 3389500.0
    c = from_convention(C, normalization="4pi", csphase=1)
    assert synthesise(c, 0.3, 1.2) == pytest.approx(3389500.0, rel=1e-15)


def test_the_coefficient_files_convention():
    # The Woodhouse-Ritsema files: c_lm = (-1)^m a_lm / sqrt 2 for m > 0.
    a = coefficients(np.random.default_rng(6))
    c = from_convention(a, normalization="woodhouse", csphase=-1)
    m = np.arange(LMAX + 1)
    want = a * np.where(m == 0, 1.0, (-1.0) ** m / np.sqrt(2.0))
    assert np.allclose(c, want, rtol=1e-15, atol=0.0)


def test_entries_outside_the_harmonics_are_left_alone():
    C = np.zeros((2, 3, 3))
    C[0, 0, 2] = 7.0                                # order above the degree
    C[1, 1, 0] = 5.0                                # sine slot of order zero
    for normalization in NORMALIZATIONS:
        c = from_convention(C, normalization=normalization, csphase=-1)
        assert c[0, 0, 2] == 7.0 and c[1, 1, 0] == 5.0


def test_refusals_name_the_convention():
    C = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="normalization must be one of"):
        from_convention(C, normalization="orthonormal", csphase=1)
    with pytest.raises(ValueError, match="csphase must be 1"):
        to_convention(C, normalization="ortho", csphase=0)
    with pytest.raises(ValueError, match="shape"):
        from_convention(np.zeros((3, 2, 2)), normalization="ortho", csphase=1)
