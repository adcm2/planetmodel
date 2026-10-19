"""The grid transforms of `planetmodel.harmonics` against the scipy basis."""
import numpy as np
import pytest

from planetmodel import (AngularGrid, driscoll_healy, equiangular, gauss_legendre,
                         real_harmonics, synthesise)
from planetmodel.harmonics import analyse_grid, synthesise_grid, synthesise_points
from planetmodel.randomfield import SphericalGRF

pytestmark = pytest.mark.harmonics
pytest.importorskip("ducc0", reason="needs the planetmodel[harmonics] extra")


def random_coefficients(L, rng, *extra):
    c = rng.standard_normal((2, L + 1, L + 1) + extra)
    for l in range(L + 1):
        c[0, l, l + 1:] = 0.0
        c[1, l, 0] = 0.0
        c[1, l, l + 1:] = 0.0
    return c


EVERY_KIND = [gauss_legendre(7), gauss_legendre(6, nphi=18),
              gauss_legendre(5, phi0=0.4), driscoll_healy(6),
              driscoll_healy(5, sampling=2), driscoll_healy(4, phi0=-2.0),
              equiangular(9, 20), equiangular(9, 20, poles=True),
              equiangular(12, 24, phi0=0.1), equiangular(30, 13),
              equiangular(18, 36, phi0=np.radians(-175.0))]


@pytest.mark.parametrize("grid", EVERY_KIND, ids=repr)
def test_synthesis_on_every_kind_matches_the_basis(grid):
    rng = np.random.default_rng(0)
    c = random_coefficients(grid.lmax, rng)
    got = synthesise_grid(c, grid)
    assert got.shape == (grid.ntheta, grid.nphi)
    want = synthesise(c, grid.colatitudes[:, None], grid.longitudes[None, :])
    assert np.allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("grid", EVERY_KIND, ids=repr)
def test_analysis_on_every_kind_inverts_synthesis_at_its_band(grid):
    rng = np.random.default_rng(1)
    c = random_coefficients(grid.lmax, rng, 3)
    values = synthesise_grid(c, grid)
    assert values.shape == (3, grid.ntheta, grid.nphi)     # the radial axis first
    back = analyse_grid(values, grid)
    assert back.shape == c.shape                           # ... and last
    assert np.allclose(back, c, atol=1e-11)


def test_the_weights_are_ducc0s():
    from ducc0.sht import get_gridweights
    for rings, grid in [("GL", gauss_legendre(9)), ("DH", driscoll_healy(6)),
                        ("F1", equiangular(11, 8)),
                        ("CC", equiangular(11, 8, poles=True))]:
        want = get_gridweights(rings, grid.ntheta) / (2 * np.pi)
        assert np.allclose(grid.weights, want, rtol=0.0, atol=1e-15), rings


def test_synthesis_at_points_agrees_with_the_basis():
    L = 12
    rng = np.random.default_rng(4)
    c = random_coefficients(L, rng, 3)
    t, p = rng.uniform(0, np.pi, (5, 7)), rng.uniform(-9, 9, (5, 7))
    got = synthesise_points(c, t, p)
    assert got.shape == (3, 5, 7)
    want = np.einsum("slmk,slm...->k...", c, real_harmonics(L, t, p))
    assert np.allclose(got, want, rtol=0, atol=1e-12)
    assert synthesise_points(c[..., 0], np.array([0.0, np.pi]), 0.0).shape == (2,)
    assert synthesise_points(c, np.zeros(0), np.zeros(0)).shape == (3, 0)
    with pytest.raises(ValueError, match=r"\[0, pi\]"):
        synthesise_points(c, 3.5, 0.0)


def test_extra_axes_and_a_wider_grid():
    L = 5
    rng = np.random.default_rng(1)
    c = random_coefficients(L, rng, 3, 2)
    grid = gauss_legendre(9)                       # band above the coefficients
    got = synthesise_grid(c, grid)
    assert got.shape == (3, 2, grid.ntheta, grid.nphi)
    Y = real_harmonics(L, grid.colatitudes[:, None], grid.longitudes[None, :])
    want = np.einsum("slmab,slmtp->abtp", c, Y)
    assert np.allclose(got, want, atol=1e-12)


def test_analysis_stops_at_a_lower_degree():
    L = 8
    rng = np.random.default_rng(2)
    c = random_coefficients(L, rng, 4)
    grid = driscoll_healy(L)
    low = analyse_grid(synthesise_grid(c, grid), grid, lmax=3)
    assert low.shape == (2, 4, 4, 4) and np.allclose(low, c[:, :4, :4], atol=1e-11)


def test_refusals_name_the_grid():
    c = random_coefficients(4, np.random.default_rng(0))
    g = gauss_legendre(4)
    with pytest.raises(ValueError, match="shipped kind"):
        synthesise_grid(c, AngularGrid(g.colatitudes, g.longitudes, lmax=4))
    with pytest.raises(ValueError, match="colatitudes are not those"):
        synthesise_grid(c, AngularGrid(np.linspace(0.3, 2.8, 5), g.longitudes,
                                       kind="gauss_legendre", lmax=4))
    with pytest.raises(ValueError, match="resolves degrees up to 4"):
        synthesise_grid(c, AngularGrid(g.colatitudes, 2 * np.pi * np.arange(13) / 13,
                                       kind="gauss_legendre", lmax=6))
    with pytest.raises(ValueError, match="2 lmax \\+ 1"):
        synthesise_grid(c, AngularGrid(g.colatitudes, g.longitudes[:8],
                                       kind="gauss_legendre", lmax=4))
    with pytest.raises(ValueError, match="equally spaced"):
        synthesise_grid(c, AngularGrid(g.colatitudes, g.longitudes ** 1.01,
                                       kind="gauss_legendre", lmax=4))
    with pytest.raises(ValueError, match="cannot hold"):
        synthesise_grid(c, gauss_legendre(3))
    with pytest.raises(ValueError, match="shape"):
        synthesise_grid(c[0], gauss_legendre(4))
    with pytest.raises(ValueError, match="trailing shape"):
        analyse_grid(np.zeros((3, 4)), gauss_legendre(4))
    with pytest.raises(ValueError, match="lmax"):
        analyse_grid(np.zeros((5, 9)), gauss_legendre(4), lmax=6)


def test_shell_sample_on_a_grid_has_the_marginal_variance():
    g = SphericalGRF(0.7, 1.0, 2.0, 0.15, sigma=1.3, lmax=10)
    grid = gauss_legendre(g.lmax)
    values = g.sample_grid(grid, rng=5)
    assert values.shape == (g.r.size, grid.ntheta, grid.nphi)
    w = grid.weights[:, None] * (2 * np.pi / grid.nphi) / (4 * np.pi)
    mean_sq = np.sum(values[3] ** 2 * w)
    # one sample: the sphere average of u^2 estimates sigma^2 to within
    # the sampling scatter of ~ (2 / N_modes)^(1/2)
    assert abs(mean_sq / g.sigma[3] ** 2 - 1.0) < 0.5
    # against the point synthesis at the grid nodes
    c = g.sample(rng=5)
    want = synthesise_grid(c, grid)
    assert np.allclose(values, want)


def test_a_product_on_the_ball_through_the_bases():
    """Two fields of a ball multiplied as a consumer would: coefficient
    functions on the full mesh, values on a grid, the product, and back.
    Multiplying by the constant one returns the field, and the product of
    two fields is the projection of the product of their values."""
    from planetmodel.randomfield import (RadialOperatorFamily, SpectralBasis,
                                         SphericalBasis, padded_mesh, restriction)
    L = 3
    mesh = padded_mesh(0.0, 1.0, pad=0.3, ngll=5, drmax=0.2)
    fam = RadialOperatorFamily(mesh, kappa=0.05)
    R = restriction(mesh, 0.0, 1.0)
    sph = SphericalBasis([SpectralBasis(fam, l, restrict=R, nmodes=6)
                          for l in range(L + 1)])
    grid = gauss_legendre(2 * L)
    rng = np.random.default_rng(3)
    a = rng.standard_normal(sph.size)
    one = np.zeros(sph.size)
    one[sph.block(0, 0, 0)] = sph[0].analyse(np.full(mesh.nglob, np.sqrt(4 * np.pi)))

    def product(u, v):
        gu = synthesise_grid(sph.synthesise(u, physical=False), grid)
        gv = synthesise_grid(sph.synthesise(v, physical=False), grid)
        assert gu.shape == (mesh.nglob, grid.ntheta, grid.nphi)
        return sph.analyse(analyse_grid(gu * gv, grid, lmax=L))

    assert np.allclose(product(a, one), a, atol=1e-10)
    b = rng.standard_normal(sph.size)
    assert np.allclose(product(a, b), product(b, a), atol=1e-12)
    assert np.linalg.norm(product(a, b)) > 0.0
