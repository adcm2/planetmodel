"""Stored fields that vary with direction: GriddedField and GriddedFunction.

A gridded field passes the field contract for every rank and rule,
equals the field it was sampled from at every node, is single-valued
at the poles in the Cartesian frame, continues its end elements beyond
its interval, and sits in a model that samples and validates.  A
gridded function is a relief whose displacement and geometry pass
their contracts.
"""
import numpy as np
import pytest

from planetmodel import (AnalyticField, AngularGrid, Geometry, GriddedField,
                         GriddedFunction, RadialStretch, Skeleton, driscoll_healy,
                         equiangular, gauss_legendre, layer_linear, sample)
from planetmodel.catalogue import LayeredIsotropicElastic
from planetmodel.character import DENSITY, ELASTIC, SCALAR, STRESS, VECTOR
from planetmodel.fields import ComposedField
from planetmodel.mesh1d import Mesh1D
from planetmodel.testing import (check_displacement, check_field, check_geometry,
                                 check_sample)

INTERVAL = (0.5, 1.0)


def smooth_for(char):
    """A smooth field of the given character, its components Cartesian."""
    shape = char.voigt_shape or char.component_shape

    def fn(r, t, p):
        r, t, p = np.broadcast_arrays(r, t, p)
        st = np.sin(t)
        x, y, z = st * np.cos(p), st * np.sin(p), np.cos(t)
        base = (1.0 + r ** 2) * np.exp(0.3 * x - 0.2 * y + 0.1 * z)
        if not char.rank:
            return base
        k = int(np.prod(shape))
        comps = [base * (1.0 + 0.1 * i) + 0.05 * i * x * r for i in range(k)]
        return np.stack(comps, axis=-1).reshape(r.shape + shape)
    return AnalyticField(INTERVAL, fn, character=char, frame="cartesian")


def nodes_of(g):
    return (g.radii[:, :, None, None], g.grid.colatitudes[None, None, :, None],
            g.grid.longitudes[None, None, None, :])


LATERAL = ["linear", "cubic",
           pytest.param("harmonic", marks=pytest.mark.harmonics)]


@pytest.mark.parametrize("radial", ["polynomial", "linear"])
@pytest.mark.parametrize("lateral", LATERAL)
@pytest.mark.parametrize("char", [SCALAR, VECTOR, STRESS, ELASTIC], ids=str)
def test_the_field_contract_for_every_rank_and_rule(char, lateral, radial):
    if lateral == "harmonic":
        pytest.importorskip("ducc0")
    src = smooth_for(char)
    g = GriddedField.sampled(src, gauss_legendre(10), breaks=[0.5, 0.75, 1.0],
                             ngll=4, radial=radial, lateral=lateral)
    check_field(g)
    assert not g.is_radial and g.radial == radial and g.lateral == lateral


@pytest.mark.parametrize("grid", [gauss_legendre(8), driscoll_healy(6),
                                  equiangular(9, 16, poles=True),
                                  AngularGrid(np.linspace(0.3, 2.8, 7),
                                              np.sort(np.random.default_rng(0)
                                                      .uniform(0.2, 6.0, 9)))],
                         ids=repr)
@pytest.mark.parametrize("char", [SCALAR, VECTOR, STRESS], ids=str)
def test_sampled_equals_its_source_at_every_node(grid, char):
    src = smooth_for(char)
    g = GriddedField.sampled(src, grid, breaks=[0.5, 0.8, 1.0], ngll=3,
                             lateral="cubic", name="q")
    r, t, p = nodes_of(g)
    want = src.evaluate(r, t, p)
    assert np.allclose(g.values, want, rtol=0, atol=1e-14)
    assert np.allclose(g.evaluate(r, t, p), want, rtol=0, atol=1e-12)
    assert g.name == "q" and g.ngll == 3 and np.array_equal(g.breaks, [0.5, 0.8, 1.0])
    # at a pole the Cartesian components are one vector for every longitude
    phi = np.linspace(-3.0, 3.0, 11)
    for k, theta in enumerate((0.0, np.pi)):
        at = g.evaluate(0.63, theta, phi, frame="cartesian")
        exact = src.evaluate(0.63, theta, 0.0, frame="cartesian")
        assert np.allclose(at, exact, rtol=0, atol=1e-12)
    assert g.poles.shape == (2, 3, 2) + g.stored_shape


def test_a_table_of_depth_slices_is_straight_lines_in_radius():
    grid = gauss_legendre(4)
    slices = np.array([0.5, 0.6, 0.85, 1.0])
    vals = np.random.default_rng(1).standard_normal((4, grid.ntheta, grid.nphi))
    poles = np.random.default_rng(2).standard_normal((4, 2))
    # elements between consecutive slices, two nodes each
    values = np.stack([vals[:-1], vals[1:]], axis=1)
    g = GriddedField(INTERVAL, slices, grid, values, radial="linear",
                     poles=np.stack([poles[:-1], poles[1:]], axis=1))
    i, j = 2, 3
    t, p = grid.colatitudes[i], grid.longitudes[j]
    r = np.linspace(0.5, 1.0, 41)
    assert np.allclose(g(r, t, p), np.interp(r, slices, vals[:, i, j]), atol=1e-13)
    assert np.allclose(g(r, 0.0, 1.0), np.interp(r, slices, poles[:, 0]), atol=1e-13)


def test_on_interval_continues_the_end_elements():
    grid = gauss_legendre(6)
    g = GriddedField.sampled(smooth_for(SCALAR), grid, breaks=[0.5, 0.7, 1.0],
                             ngll=5, lateral="cubic")
    wide = g.on_interval(0.4, 1.1)
    assert wide.interval == (0.4, 1.1)
    with pytest.raises(ValueError, match="outside the interval"):
        g(1.05, 1.0, 1.0)
    i, j = 3, 4
    t, p = grid.colatitudes[i], grid.longitudes[j]
    pp = Mesh1D(g.breaks, ngll=5).to_ppoly(g.values[:, :, i, j])
    r = np.array([0.4, 0.45, 1.05, 1.1])
    assert np.allclose(wide(r, t, p), pp(r), atol=1e-12)
    narrow = g.on_interval(0.6, 0.9)
    assert np.allclose(narrow(0.65, t, p), g(0.65, t, p))
    assert g.renamed("x").name == "x" and g.name is None


def test_rescaled_scales_radius_and_value():
    grid = gauss_legendre(5)
    g = GriddedField.sampled(smooth_for(VECTOR), grid, breaks=[0.5, 1.0], ngll=4,
                             lateral="cubic")
    s = g.rescaled(k=2.0, v=3.0)
    assert s.interval == (1.0, 2.0) and np.allclose(s.breaks, 2.0 * g.breaks)
    rng = np.random.default_rng(3)
    r, t, p = (rng.uniform(0.5, 1.0, 20), rng.uniform(0, np.pi, 20),
               rng.uniform(0, 6, 20))
    assert np.allclose(s(2.0 * r, t, p), 3.0 * g(r, t, p), atol=1e-12)


def test_refusals_name_what_is_wrong():
    grid = gauss_legendre(4)
    v = np.ones((2, 3, grid.ntheta, grid.nphi))
    P = np.ones((2, 3, 2))
    with pytest.raises(ValueError, match="span"):
        GriddedField(INTERVAL, [0.5, 0.7, 0.9], grid, v, poles=P)
    with pytest.raises(ValueError, match="strictly increasing"):
        GriddedField(INTERVAL, [0.5, 1.0, 0.8], grid, v, poles=P)
    with pytest.raises(ValueError, match="shape"):
        GriddedField(INTERVAL, [0.5, 0.75, 1.0], grid, v[:, :, :, :-1], poles=P)
    with pytest.raises(ValueError, match="ngll >= 2"):
        GriddedField(INTERVAL, [0.5, 0.75, 1.0], grid, v[:, :1], poles=P[:, :1])
    with pytest.raises(ValueError, match="radial"):
        GriddedField(INTERVAL, [0.5, 0.75, 1.0], grid, v, poles=P, radial="spline")
    with pytest.raises(ValueError, match="lateral"):
        GriddedField(INTERVAL, [0.5, 0.75, 1.0], grid, v, poles=P, lateral="spline")
    with pytest.raises(TypeError, match="real"):
        GriddedField(INTERVAL, [0.5, 0.75, 1.0], grid, v + 0j, poles=P)
    with pytest.raises(ValueError, match="finite"):
        GriddedField(INTERVAL, [0.5, 0.75, 1.0], grid, v * np.nan, poles=P)
    with pytest.raises(ValueError, match="poles must have shape"):
        GriddedField(INTERVAL, [0.5, 0.75, 1.0], grid, v, poles=P[:, :, :1])
    custom = AngularGrid(grid.colatitudes, grid.longitudes)
    with pytest.raises(ValueError, match="poles="):
        GriddedField(INTERVAL, [0.5, 0.75, 1.0], custom, v)
    with pytest.raises(TypeError, match="AngularGrid"):
        GriddedField(INTERVAL, [0.5, 0.75, 1.0], (grid.colatitudes,), v, poles=P)


def test_a_pole_row_must_hold_one_value():
    grid = driscoll_healy(3)                      # the north pole is a node
    v = np.ones((1, 2, grid.ntheta, grid.nphi))
    v[0, 1, 0, 2] = 1.5                           # the north row varies
    with pytest.raises(ValueError, match="north pole is a node"):
        GriddedField(INTERVAL, INTERVAL, grid, v, poles=np.ones((1, 2, 2)))
    # a vector's spherical components at a pole node vary with longitude;
    # its Cartesian ones do not, and that is what is checked
    g = GriddedField.sampled(smooth_for(VECTOR), grid, ngll=2)
    assert np.ptp(g.values[0, 0, 0, :, 1]) > 0.01
    assert np.allclose(g.poles[0, 0, 0], g.evaluate(0.5, 0.0, 2.0, frame="cartesian"))


def test_a_gridded_field_in_a_model_samples_and_validates():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    grid = gauss_legendre(6)

    def rho3d(r, t, p):
        return 4.0 + 0.2 * r * np.sin(t) * np.cos(p)

    g = GriddedField.sampled(AnalyticField(INTERVAL, rho3d, character=DENSITY), grid,
                             breaks=[0.5, 0.75, 1.0], ngll=4, lateral="cubic")
    m = m.with_field(1, "rho3d", g)
    m = m.with_field(0, "rho3d", AnalyticField((0.0, 0.5), rho3d, character=DENSITY))
    s = sample(m, gauss_legendre(4), fields=["rho3d"], drmax=0.25)
    check_sample(s, m)
    assert np.allclose((g * 2.0)(0.7, 1.0, 2.0), 2.0 * g(0.7, 1.0, 2.0))
    q = GriddedField.sampled(smooth_for(SCALAR), grid, ngll=3)
    assert isinstance(q + 1.0, ComposedField)
    assert np.allclose((q + 1.0)(0.7, 1.0, 2.0), q(0.7, 1.0, 2.0) + 1.0)


# -- GriddedFunction ------------------------------------------------------------

def bump(t, p):
    """A relief of degree three, with a gradient at both poles."""
    return (0.02 * np.sin(t) ** 3 * np.cos(3.0 * p) + 0.01 * np.cos(t)
            + 0.005 * np.sin(t) * np.cos(p - 0.4))


def bump_gradient(t, p):
    """(d/dtheta, d/dphi) of `bump`."""
    st, ct = np.sin(t), np.cos(t)
    return (0.06 * st ** 2 * ct * np.cos(3 * p) - 0.01 * st
            + 0.005 * ct * np.cos(p - 0.4),
            -0.06 * st ** 3 * np.sin(3 * p) - 0.005 * st * np.sin(p - 0.4))


def test_a_gridded_function_is_a_relief():
    gf = GriddedFunction.sampled(bump, gauss_legendre(24))
    assert gf.lateral == "cubic" and gf.values.shape == (25, 49)
    assert np.allclose(gf.poles, [0.01, -0.01])
    sk = Skeleton([0.0, 0.5, 1.0])
    check_displacement(layer_linear(sk, [None, gf, gf]), sk)
    rng = np.random.default_rng(4)
    t, p = rng.uniform(0, np.pi, 200), rng.uniform(0, 2 * np.pi, 200)
    assert np.allclose(gf(t, p), bump(t, p), atol=1e-6)


@pytest.mark.harmonics
def test_a_harmonic_relief_gives_a_geometry_that_holds():
    pytest.importorskip("ducc0")
    gf = GriddedFunction.sampled(bump, gauss_legendre(8), lateral="harmonic")
    sk = Skeleton([0.0, 0.5, 1.0])
    h = layer_linear(sk, [None, gf, gf])
    check_displacement(h, sk)
    check_geometry(Geometry(sk, mapping=RadialStretch(h, rmax=1.0)))


@pytest.mark.parametrize("lateral", ["linear", "cubic",
                                     pytest.param("harmonic",
                                                  marks=pytest.mark.harmonics)])
def test_the_angular_gradient_is_the_rule_differentiated(lateral):
    if lateral == "harmonic":
        pytest.importorskip("ducc0")
    gf = GriddedFunction.sampled(bump, gauss_legendre(32), lateral=lateral)
    rng = np.random.default_rng(5)
    t, p = rng.uniform(0.2, 2.9, 100), rng.uniform(0, 2 * np.pi, 100)
    h = 1e-6
    dt, dp = gf.angular_gradient(t, p)
    if lateral == "linear":            # piecewise constant: away from cell edges
        g = gf.grid
        ok = ((np.abs(t[:, None] - g.colatitudes[None]).min(axis=1) > 10 * h)
              & (np.abs(p[:, None] - g.longitudes[None]).min(axis=1) > 10 * h))
    else:
        ok = np.ones(t.shape, dtype=bool)
    assert np.allclose(dt[ok], ((gf(t + h, p) - gf(t - h, p)) / (2 * h))[ok], atol=1e-7)
    assert np.allclose(dp[ok], ((gf(t, p + h) - gf(t, p - h)) / (2 * h))[ok], atol=1e-7)
    if lateral != "linear":
        exact_dt, exact_dp = bump_gradient(t, p)
        tol = 1e-12 if lateral == "harmonic" else 2e-5
        assert np.allclose(dt, exact_dt, rtol=0, atol=tol)
        assert np.allclose(dp, exact_dp, rtol=0, atol=tol)


def test_a_gridded_function_refuses_what_it_cannot_hold():
    grid = gauss_legendre(4)
    with pytest.raises(ValueError, match="ntheta, nphi"):
        GriddedFunction(grid, np.ones((1, grid.ntheta, grid.nphi)), poles=[1.0, 1.0])
    with pytest.raises(ValueError, match="lateral"):
        GriddedFunction(grid, np.ones((grid.ntheta, grid.nphi)), lateral="quintic",
                        poles=[1.0, 1.0])
    with pytest.raises(TypeError, match="real"):
        GriddedFunction(grid, np.ones((grid.ntheta, grid.nphi)) + 0j, poles=[1, 1])
    assert "cubic" in repr(GriddedFunction(grid, np.ones((grid.ntheta, grid.nphi)),
                                           poles=[1.0, 1.0]))
