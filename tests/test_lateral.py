"""The lateral rules: values between the nodes of an angular grid.

The local rules interpolate the nodes and the polar values, read a
longitude and its turns alike, and agree with scipy's FITPACK spline
and linear grid interpolator on a padded grid built here from the rule
as stated; they converge at second and fourth order on a smooth field.
The harmonic rule is exact on a band-limited field.  The accuracy
against oversampling is measured on the setup of the C++ figures that
the stage plan records, with a scalar field.
"""
import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

from planetmodel import (AngularGrid, driscoll_healy, equiangular, gauss_legendre,
                         synthesise)
from planetmodel.lateral import (GHOSTS, METHODS, LateralInterpolant, interpolate,
                                 polar_values)

LOCAL = ("linear", "cubic")


def smooth(theta, phi):
    """A smooth function on the sphere: one of the Cartesian coordinates."""
    st = np.sin(theta)
    return np.exp(st * np.cos(phi) + 0.5 * st * np.sin(phi) - 0.3 * np.cos(theta))


def on(grid, fn=smooth):
    """fn at the grid's nodes, and at its poles (north, south)."""
    v = fn(grid.colatitudes[:, None], grid.longitudes[None, :])
    return v, fn(np.array([0.0, np.pi]), np.zeros(2))


def offset_custom():
    """A custom grid, its longitudes unequal and starting away from zero."""
    phi = np.sort(np.random.default_rng(7).uniform(0.3, 6.1, 11))
    return AngularGrid(np.linspace(0.2, 2.9, 9), phi)


GRIDS = [gauss_legendre(6), driscoll_healy(5), equiangular(8, 14),
         equiangular(9, 14, poles=True), offset_custom()]


@pytest.mark.parametrize("method", LOCAL)
@pytest.mark.parametrize("grid", GRIDS, ids=repr)
def test_local_rules_interpolate_the_nodes_and_the_poles(grid, method):
    v, poles = on(grid)
    f = LateralInterpolant(v, grid, method=method, poles=poles)
    got = f(grid.colatitudes[:, None], grid.longitudes[None, :])
    assert np.allclose(got, v, rtol=0, atol=1e-12)
    phi = np.linspace(-7.0, 7.0, 29)
    assert np.allclose(f(0.0, phi), poles[0], rtol=0, atol=1e-12)
    assert np.allclose(f(np.pi, phi), poles[1], rtol=0, atol=1e-12)


@pytest.mark.parametrize("method", LOCAL)
@pytest.mark.parametrize("grid", GRIDS, ids=repr)
def test_a_longitude_and_its_turns_read_alike(grid, method):
    v, poles = on(grid)
    f = LateralInterpolant(v, grid, method=method, poles=poles)
    rng = np.random.default_rng(1)
    t, p = rng.uniform(0, np.pi, 50), rng.uniform(0, 2 * np.pi, 50)
    base = f(t, p)
    for k in (-2, -1, 1, 3):
        assert np.allclose(f(t, p + 2 * np.pi * k), base, rtol=0, atol=1e-12)
    # continuous across the seam at the first longitude
    phi0 = grid.longitudes[0]
    eps = 1e-9
    assert np.allclose(f(t, phi0 - eps), f(t, phi0 + eps), rtol=0, atol=1e-7)


def pad_as_stated(v, grid, poles):
    """The padded grid built from the rule in its own words: the wrap
    column, g = min(3, n - 1) ghost columns a period away at each end,
    and a constant row at every pole the grid lacks."""
    theta, phi = list(grid.colatitudes), list(grid.longitudes)
    n = len(phi)
    g = min(3, n - 1)
    cols = ([phi[j] - 2 * np.pi for j in range(n - g, n)] + phi
            + [phi[j] + 2 * np.pi for j in range(g + 1)])
    idx = list(range(n - g, n)) + list(range(n)) + list(range(g + 1))
    rows = v[:, idx]
    if not grid.pole_nodes[0]:
        theta = [0.0] + theta
        rows = np.vstack([np.full(len(cols), poles[0]), rows])
    if not grid.pole_nodes[1]:
        theta = theta + [np.pi]
        rows = np.vstack([rows, np.full(len(cols), poles[1])])
    return np.array(theta), np.array(cols), rows


@pytest.mark.parametrize("grid", GRIDS, ids=repr)
def test_the_rules_are_scipys_splines_on_the_padded_grid(grid):
    assert GHOSTS == 3
    v, poles = on(grid)
    T, P, V = pad_as_stated(v, grid, poles)
    rng = np.random.default_rng(2)
    t = rng.uniform(0, np.pi, 300)
    p = rng.uniform(0, 2 * np.pi, 300)
    q = grid.longitudes[0] + np.mod(p - grid.longitudes[0], 2 * np.pi)
    cubic = RectBivariateSpline(T, P, V, kx=3, ky=3, s=0).ev(t, q)
    linear = RegularGridInterpolator((T, P), V)(np.stack([t, q], -1))
    assert np.allclose(interpolate(v, grid, t, p, method="cubic", poles=poles), cubic,
                       rtol=0, atol=1e-12)
    assert np.allclose(interpolate(v, grid, t, p, method="linear", poles=poles),
                       linear, rtol=0, atol=1e-13)


@pytest.mark.parametrize("method, order", [("linear", 2), ("cubic", 4)])
def test_local_rules_converge_at_their_orders(method, order):
    rng = np.random.default_rng(3)
    t, p = rng.uniform(0, np.pi, 2000), rng.uniform(0, 2 * np.pi, 2000)
    exact = smooth(t, p)
    n = np.array([8, 16, 32, 64])
    err = []
    for lmax in n:
        g = gauss_legendre(int(lmax))
        v, poles = on(g)
        err.append(np.abs(interpolate(v, g, t, p, method=method, poles=poles)
                          - exact).max())
    rate = -np.polyfit(np.log(n), np.log(err), 1)[0]
    # measured: 2.0 for linear, 4.0 for cubic
    assert abs(rate - order) < 0.25, (rate, err)


def band16_field():
    """The setup of the C++ figures: band 16, coefficients N(0, 1) / (l + 1)."""
    L = 16
    c = np.random.default_rng(11).standard_normal((2, L + 1, L + 1))
    c /= (np.arange(L + 1) + 1.0)[None, :, None]
    for l in range(L + 1):
        c[0, l, l + 1:] = 0.0
        c[1, l, 0] = 0.0
        c[1, l, l + 1:] = 0.0
    return c


def test_accuracy_against_oversampling_beside_the_cpp_figures():
    """The largest error at 4000 random points relative to the largest
    value, on Gauss-Legendre grids of 1, 4 and 8 times the band.  Measured
    here for a scalar field (linear, cubic): 2.8e-1, 9.4e-2; 2.3e-2,
    4.6e-4; 5.2e-3, 3.4e-5.  The C++ figures for the bicubic rule on a
    spin-2 field: 13%, 7e-4, 4e-5."""
    c = band16_field()
    rng = np.random.default_rng(20260824)
    t, p = rng.uniform(0, np.pi, 4000), rng.uniform(0, 2 * np.pi, 4000)
    ref = synthesise(c, t, p)
    scale = np.abs(ref).max()
    poles = synthesise(c, np.array([0.0, np.pi]), np.zeros(2))
    bounds = {1: (0.35, 0.12), 4: (3e-2, 6e-4), 8: (7e-3, 5e-5)}
    for f, (lin, cub) in bounds.items():
        g = gauss_legendre(16 * f)
        v = synthesise(c, g.colatitudes[:, None], g.longitudes[None, :])
        for method, bound in (("linear", lin), ("cubic", cub)):
            got = interpolate(v, g, t, p, method=method, poles=poles)
            assert np.abs(got - ref).max() / scale < bound, (f, method)


def test_the_leading_shape_is_carried_through():
    g = gauss_legendre(5)
    v, poles = on(g)
    stack = np.stack([v * k for k in range(1, 7)]).reshape(3, 2, *v.shape)
    pstack = np.stack([poles * k for k in range(1, 7)]).reshape(3, 2, 2)
    f = LateralInterpolant(stack, g, method="cubic", poles=pstack)
    assert f.shape == (3, 2) and f.method == "cubic" and f.grid is g
    t = np.full((4, 5), 0.7)
    out = f(t, 1.1)
    assert out.shape == (3, 2, 4, 5)
    one = interpolate(v, g, 0.7, 1.1, method="cubic", poles=poles)
    assert one.shape == ()
    assert np.allclose(out[1, 0], 3.0 * one)


def test_refusals_say_what_is_wrong():
    g = gauss_legendre(4)
    v, poles = on(g)
    with pytest.raises(ValueError, match="method"):
        LateralInterpolant(v, g, method="quintic", poles=poles)
    assert set(METHODS) == {"linear", "cubic", "harmonic"}
    with pytest.raises(ValueError, match="trailing shape"):
        LateralInterpolant(v[:, :-1], g, method="linear", poles=poles)
    with pytest.raises(ValueError, match="shape"):
        LateralInterpolant(v, g, method="linear", poles=poles[:1])
    with pytest.raises(TypeError, match="real"):
        LateralInterpolant(v + 0j, g, method="linear", poles=poles)
    with pytest.raises(TypeError, match="AngularGrid"):
        LateralInterpolant(v, (g.colatitudes, g.longitudes), method="linear")
    custom = AngularGrid(g.colatitudes, g.longitudes)
    with pytest.raises(ValueError, match="poles="):
        LateralInterpolant(v, custom, method="linear")
    f = LateralInterpolant(v, g, method="linear", poles=poles)
    with pytest.raises(ValueError, match=r"\[0, pi\]"):
        f(3.2, 0.0)
    with pytest.raises(ValueError, match="finite"):
        f(np.nan, 0.0)
    tiny = gauss_legendre(0)
    with pytest.raises(ValueError, match="at least 4 rows"):
        LateralInterpolant(np.ones((1, 1)), tiny, method="cubic", poles=[1.0, 1.0])
    assert np.allclose(LateralInterpolant(np.ones((1, 1)), tiny, method="linear",
                                          poles=[1.0, 1.0])(1.0, 2.0), 1.0)


# -- through the expansion: the harmonics extra ---------------------------------

@pytest.mark.harmonics
@pytest.mark.parametrize("grid", [gauss_legendre(12), gauss_legendre(10, nphi=24),
                                  driscoll_healy(10, sampling=2),
                                  equiangular(14, 26, poles=True)], ids=repr)
def test_the_harmonic_rule_is_exact_on_a_band_limited_field(grid):
    pytest.importorskip("ducc0")
    c = band16_field()[:, :grid.lmax + 1, :grid.lmax + 1]
    v = synthesise(c, grid.colatitudes[:, None], grid.longitudes[None, :])
    rng = np.random.default_rng(5)
    t, p = rng.uniform(0, np.pi, 500), rng.uniform(-3, 9, 500)
    got = interpolate(v, grid, t, p, method="harmonic")
    assert np.allclose(got, synthesise(c, t, p), rtol=0, atol=1e-12)


@pytest.mark.harmonics
def test_polar_values_come_from_the_expansion():
    pytest.importorskip("ducc0")
    g = gauss_legendre(16)
    c = band16_field()
    v = synthesise(c, g.colatitudes[:, None], g.longitudes[None, :])
    exact = synthesise(c, np.array([0.0, np.pi]), np.zeros(2))
    assert np.allclose(polar_values(v, g), exact, rtol=0, atol=1e-12)
    stack = np.stack([v, 2 * v])
    assert polar_values(stack, g).shape == (2, 2)
    # a grid with a band and no poles given takes them from its expansion
    t = np.array([0.0, 0.01, np.pi])
    assert np.allclose(interpolate(v, g, t, 0.3, method="cubic"),
                       interpolate(v, g, t, 0.3, method="cubic", poles=exact),
                       rtol=0, atol=1e-12)


# -- the polar values: one source each -------------------------------------------

def test_a_pole_node_is_its_own_row_and_poles_must_agree_or_be_nan():
    g = driscoll_healy(4)                 # the north pole a node, the south not
    v, exact = on(g)
    assert np.allclose(polar_values(v, g, poles=[np.nan, exact[1]]), exact)
    assert np.allclose(polar_values(v, g, poles=exact), exact)
    with pytest.raises(ValueError, match="disagrees with its row"):
        polar_values(v, g, poles=[exact[0] + 1.0, exact[1]])
    with pytest.raises(ValueError, match="south pole, so its entry"):
        polar_values(v, g, poles=[exact[0], np.nan])
    f = LateralInterpolant(v, g, method="cubic", poles=[np.nan, exact[1]])
    assert np.allclose(f(0.0, np.linspace(0, 6, 7)), exact[0], rtol=0, atol=1e-12)


def test_a_varying_pole_row_is_refused():
    g = equiangular(7, 12, poles=True)
    v, _ = on(g)
    v[-1, 3] += 1e-3
    with pytest.raises(ValueError, match="south pole is a node"):
        LateralInterpolant(v, g, method="linear")
    v[-1, 3] -= 1e-3
    assert np.allclose(polar_values(v, g), on(g)[1])       # both from the rows
    with pytest.raises(ValueError, match="rtol"):
        v[0, 1] *= 1.0 + 1e-6
        polar_values(v, g)
    assert polar_values(v, g, rtol=1e-5).shape == (2,)


@pytest.mark.harmonics
def test_the_harmonic_rule_checks_poles_without_using_them():
    pytest.importorskip("ducc0")
    g = driscoll_healy(8)
    v, exact = on(g)
    with pytest.raises(ValueError, match="disagrees"):
        LateralInterpolant(v, g, method="harmonic", poles=[exact[0] + 1, exact[1]])
    a = LateralInterpolant(v, g, method="harmonic", poles=[np.nan, exact[1] + 5.0])
    b = LateralInterpolant(v, g, method="harmonic")
    t = np.array([0.0, 1.0, np.pi])
    assert np.array_equal(a(t, 0.4), b(t, 0.4))


@pytest.mark.parametrize("method, tol", [("linear", 0.2), ("cubic", 1.5e-4)])
def test_the_gradient_of_a_local_rule(method, tol):
    g = gauss_legendre(40)
    v, poles = on(g)
    f = LateralInterpolant(v, g, method=method, poles=poles)
    rng = np.random.default_rng(6)
    t, p = rng.uniform(0.1, 3.0, 300), rng.uniform(0, 2 * np.pi, 300)
    dt, dp = f.gradient(t, p)
    h = 1e-6
    st, ct = np.sin(t), np.cos(t)
    arg = st * np.cos(p) + 0.5 * st * np.sin(p) - 0.3 * ct
    exact_dt = np.exp(arg) * (ct * np.cos(p) + 0.5 * ct * np.sin(p) + 0.3 * st)
    exact_dp = np.exp(arg) * (-st * np.sin(p) + 0.5 * st * np.cos(p))
    assert np.abs(dt - exact_dt).max() < tol and np.abs(dp - exact_dp).max() < tol
    assert f.gradient(t[0], p[0])[0].shape == ()
    if method == "cubic":
        fd = (f(t + h, p) - f(t - h, p)) / (2 * h)
        assert np.allclose(dt, fd, atol=1e-7)
