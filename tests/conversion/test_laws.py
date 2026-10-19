"""The conversion laws, their profiles and their parameter sets: each law
against hand-worked numbers and its derivation, sl3d's spline against
Numerical Recipes' natural spline, Karato's anelastic sensitivity against
a difference of the dispersion law it comes from, and every shipped set
against the values its source states."""
import numpy as np
import pytest

from planetmodel.conversion import laws, parameters, profiles


def nr_natural_spline(x, y):
    """Numerical Recipes' `spline` with natural ends and `splint`, which
    clamps the bracketing interval to the first or the last: the oracle
    for sl3d's spline."""
    n = len(x)
    y2, u = np.zeros(n), np.zeros(n)
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = ((y[i + 1] - y[i]) / (x[i + 1] - x[i])
                - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
        u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p
    for k in range(n - 2, -1, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]

    def splint(t):
        klo = int(np.clip(np.searchsorted(x, t) - 1, 0, n - 2))
        khi = klo + 1
        h = x[khi] - x[klo]
        a, b = (x[khi] - t) / h, (t - x[klo]) / h
        return (a * y[klo] + b * y[khi]
                + ((a ** 3 - a) * y2[klo] + (b ** 3 - b) * y2[khi]) * h * h / 6.0)

    return splint


def test_profiles():
    r = np.array([[3.5e6, 5.0e6], [6.0e6, 6.371e6]])
    assert np.array_equal(profiles.constant(2.0)(r), np.full((2, 2), 2.0))
    assert profiles.as_profile(3.0)(1.0) == 3.0
    p = profiles.polynomial([1.0, -2.0, 0.5], scale=1e6)
    x = r / 1e6
    assert np.allclose(p(r), 1.0 - 2.0 * x + 0.5 * x * x, rtol=1e-15)
    t = profiles.tabulated([1.0, 2.0, 4.0], [10.0, 20.0, 0.0])
    assert np.allclose(t(np.array([0.0, 1.5, 3.0, 9.0])), [10.0, 15.0, 10.0, 0.0])
    knots, values = [1.0, 2.0, 3.5, 5.0], [0.0, 1.0, -1.0, 2.0]
    s = profiles.spline(knots, values)
    assert np.allclose(s(np.array(knots)), values, rtol=0, atol=1e-15)
    for bad in (([1.0, 1.0, 2.0], [0, 1, 2]), ([1.0, 2.0], [0, 1]),
                ([1.0, 2.0, np.nan], [0, 1, 2])):
        with pytest.raises(ValueError):
            profiles.spline(*bad)
    with pytest.raises(ValueError, match="positive"):
        profiles.polynomial([1.0], scale=0.0)
    assert "spline through 4 knots" in repr(s)


def test_sl3d_s_spline_is_numerical_recipes_natural_spline_inside_and_beyond():
    x = np.array([3870e3, 4524e3, 5043e3, 5537e3, 5870e3, 6203e3])
    y = np.array([0.1, 0.25, 0.337, 0.337, 0.25, 0.1])
    oracle = nr_natural_spline(x, y)
    r = np.linspace(3480e3, 6371e3, 301)
    got = parameters.SL3D_DENSITY_SCALING.scaling(r)
    assert np.allclose(got, [oracle(t) for t in r], rtol=0, atol=1e-14)
    # the end cubic continued: 0.0080 at the CMB, as the thesis draws it
    assert parameters.SL3D_DENSITY_SCALING.scaling(3480e3) == pytest.approx(0.0080,
                                                                            abs=5e-5)


def test_the_expansivities_have_their_sources_values():
    a = parameters.SL3D_EXPANSIVITY.expansivity
    # 1.0e-5 at the CMB, 2.5e-5 at 670 km, 3.58e-5 at the surface
    assert a(3480e3) == pytest.approx(1.0e-5, rel=1e-3)
    assert a(6371e3 - 670e3) == pytest.approx(2.5e-5, rel=1e-3)
    assert a(6371e3) == pytest.approx(3.58e-5, rel=1e-3)
    t = 5000e3 / 2891e3
    want = 2.5e-5 * (-1.04953003 + 2.86975098 * t - 2.09738922 * t ** 2
                     + 0.59281731 * t ** 3)
    assert a(5000e3) == pytest.approx(want, rel=1e-15)
    p = parameters.PAULSON_2005_EXPANSIVITY.expansivity
    depth = np.array([0.0, 670e3, 2891e3])
    rel = depth / 2891e3
    assert np.allclose(p(6371e3 - depth), (rel + 3.0 * (1.0 - rel)) * 1e-5, rtol=1e-14)


def test_the_shipped_sets_cite_and_hold_their_numbers():
    P = parameters
    assert P.SL3D_VISCOSITY.epsilon(5e6) == 0.01
    assert P.MILNE_2018_VISCOSITY.epsilon(5e6) == 0.04
    assert P.PAULSON_2005_DENSITY_SCALING.scaling(5e6) == 0.3
    assert P.MITROVICA_2011_DENSITY_SCALING.scaling(5e6) == 0.4
    assert P.GPLSPEC_DENSITY_SCALING.scaling(5e6) == 0.5
    assert P.GPLSPEC_DENSITY_SCALING.citation == ()
    assert P.SL3D_DENSITY_SCALING.citation[0].startswith("Crawford, O. (2019)")
    assert "doi:10.1093/gji/ggy189" in P.MILNE_2018_VISCOSITY.cite()
    for s in (P.SL3D_DENSITY_SCALING, P.SL3D_EXPANSIVITY, P.SL3D_VISCOSITY,
              P.MILNE_2018_VISCOSITY, P.PAULSON_2005_DENSITY_SCALING,
              P.PAULSON_2005_EXPANSIVITY, P.MITROVICA_2011_DENSITY_SCALING):
        assert s.cite().splitlines() == list(s.citation) and s.citation
    mine = P.BetaViscosity(0.7, 9e4, 2000.0, -1e-4, name="mine", citation=["Me."])
    assert mine.beta(1.0) == 0.7 and isinstance(mine.background, profiles.Profile)
    with pytest.raises(TypeError, match="not one string"):
        P.Expansivity(3e-5, name="x", citation="Me.")


def test_the_linear_laws():
    dlnvs = np.array([-0.02, 0.0, 0.01])
    dt = laws.temperature_from_velocity(dlnvs, sensitivity=-1e-4)
    assert np.allclose(dt, [200.0, 0.0, -100.0])
    drho = laws.density_from_velocity(dlnvs, scaling=0.3)
    assert np.allclose(drho, [-0.006, 0.0, 0.003])
    assert np.allclose(laws.density_from_temperature(dt, expansivity=2e-5),
                       [-0.004, 0.0, 0.002])
    back = laws.temperature_from_density(drho, expansivity=3e-5)
    assert np.allclose(laws.density_from_temperature(back, expansivity=3e-5), drho)
    assert np.allclose(laws.exponential_viscosity(dt, epsilon=0.01), [-2.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="nowhere zero"):
        laws.temperature_from_velocity(dlnvs, sensitivity=[-1e-4, 0.0, -1e-4])


def test_beta_scaling_is_wu_wang_and_steffen_s_printed_law():
    # log10(eta/eta0) = [-0.4343 beta / s] [a / T0^2] dVs/Vs, a = (E* + pV*)/R
    beta, a, t0, s = 0.65, 1.0e5, 2000.0, -1.5e-4
    dlnvs = np.array([-0.01, 0.0, 0.02])
    printed = -np.log10(np.e) * beta / s * a / t0 ** 2 * dlnvs
    got = laws.beta_viscosity(dlnvs, beta=beta, activation=a, background=t0,
                              sensitivity=s)
    assert np.allclose(got / np.log(10.0), printed, rtol=1e-14)
    assert got[2] > 0.0 > got[0]                      # fast is stiff
    # with beta = 1 it is the exponential law with eps = a / T0^2 after
    # the temperature anomaly of the same sensitivity
    eps = a / t0 ** 2
    dt = laws.temperature_from_velocity(dlnvs, sensitivity=s)
    assert np.allclose(laws.beta_viscosity(dlnvs, beta=1.0, activation=a, background=t0,
                                           sensitivity=s),
                       laws.exponential_viscosity(dt, epsilon=eps), rtol=1e-14)


def test_karato_s_anelastic_sensitivity_is_the_dispersion_law_s_derivative():
    # V = V_anh [1 - (1/2) cot(pi a / 2) Q^-1], Q^-1 = Q0^-1 exp(-a H / (R T))
    # scaled to Q at T; to first order in Q^-1, d ln V / dT is the law's
    a, h, t = 0.25, 4.0e5, 1600.0
    q = 120.0

    c = 0.5 / np.tan(0.5 * np.pi * a)

    def qinv(temp):
        return (1.0 / q) * np.exp(-a * h / laws.GAS_CONSTANT * (1.0 / temp - 1.0 / t))

    step = 1e-2
    got = laws.anelastic_sensitivity(t, q=q, enthalpy=h, exponent=a)
    first = -c * (qinv(t + step) - qinv(t - step)) / (2.0 * step)
    assert got == pytest.approx(first, rel=1e-8)            # the law is first order
    full = (np.log1p(-c * qinv(t + step)) - np.log1p(-c * qinv(t - step))) / (2 * step)
    assert got == pytest.approx(full, rel=2.0 * c / q)      # in Q^-1
    f = 0.5 * np.pi * a / np.tan(0.5 * np.pi * a)
    assert got == pytest.approx(-f * h / (np.pi * q * laws.GAS_CONSTANT * t * t),
                                rel=1e-15)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        laws.anelastic_sensitivity(t, q=q, enthalpy=h, exponent=1.0)


def test_ivins_and_sammis_s_table_1():
    P = parameters
    depth = np.array([350.0, 700.0, 1500.0, 2835.0]) * 1e3
    r = 6371e3 - depth
    assert np.allclose(P.IVINS_SAMMIS_1995_EXPANSIVITY.expansivity(r),
                       [1.725e-5, 1.463e-5, 1.177e-5, 0.65e-5], rtol=1e-14)
    whole = P.ivins_sammis_beta(0.65, -1e-4)
    assert np.allclose(whole.activation(r), [8.2e4, 8.3e4, 9.97e4, 11.7e4], rtol=1e-14)
    assert np.allclose(whole.background(r), [1990.0, 2230.0, 2470.0, 2675.0],
                       rtol=1e-14)
    layered = P.ivins_sammis_beta(0.65, -1e-4, convection="layered")
    assert np.allclose(layered.background(r), [2088.0, 2300.0, 2650.0, 3250.0],
                       rtol=1e-14)
    # linear between rows, held beyond them
    mid = 6371e3 - 1000e3
    assert whole.activation(mid) == pytest.approx(8.7e4 + (9.35e4 - 8.7e4) / 3.0)
    assert whole.activation(6371e3) == 8.2e4 and whole.activation(3480e3) == 11.7e4
    assert whole.beta(5e6) == 0.65 and whole.sensitivity(5e6) == -1e-4
    assert whole.citation[0].startswith("Wu, P.") and "tb06856" in whole.citation[1]
    with pytest.raises(ValueError, match="convection must be"):
        P.ivins_sammis_beta(0.65, -1e-4, convection="plume")
