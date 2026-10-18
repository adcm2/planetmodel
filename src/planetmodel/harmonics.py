"""Real orthonormal spherical harmonics, and synthesis from coefficients.

The basis is real and orthonormal on the unit sphere: with theta the
colatitude, phi the longitude and P_l^m the associated Legendre
functions without the Condon-Shortley phase,

    Y_l0        = N_l0 P_l(cos theta),
    Y_lm^cos    = sqrt 2 N_lm P_l^m(cos theta) cos(m phi),   m > 0,
    Y_lm^sin    = sqrt 2 N_lm P_l^m(cos theta) sin(m phi),   m > 0,
    N_lm        = sqrt((2l + 1) / 4 pi  (l - m)! / (l + m)!),

so that a field u = sum_lm c_lm Y_lm has sum_lm c_lm^2 = int u^2 dS.
Coefficient arrays are laid out as `coeffs[0, l, m]` for the cosine
harmonic of order m and `coeffs[1, l, m]` for the sine one, zero for
m = 0 in the sine slot and for m > l.  Scipy's complex harmonics carry
the Condon-Shortley phase; it is removed here.  The layout and the
convention are those of pyshtools under `normalization="ortho"` and
`csphase=1`, so a coefficient array here is a pyshtools `cilm` array
as it stands.  Of its 2 (lmax + 1)^2 entries, (lmax + 1)^2 belong to a
harmonic; `packing` lists them in the one order in which they are laid
out as a vector: by degree, then the cosine harmonics of orders
0 ... l, then the sine harmonics of orders 1 ... l.

Two routes.  `real_harmonics` and `synthesise` evaluate the basis at
arbitrary points through scipy, vectorised over the points: the
definition, and the oracle everything else is held to.  `synthesise_grid`
and `analyse_grid` go between coefficients and values on a grid of a
shipped kind -- Gauss-Legendre, Driscoll-Healy, or equiangular with or
without the poles, from `planetmodel.sampling` -- through ducc0's
transforms: fast, exact for a field band-limited to the grid's band,
and needing the `harmonics` extra.  The longitudes may start anywhere
and number anything from 2 lmax + 1 up, provided they are equally
spaced.  ducc0 holds complex coefficients a_lm for m >= 0 in healpy's
order, with the Condon-Shortley phase; the real ones here are

    a_l0 = c[0, l, 0],    a_lm = (-1)^m (c[0, l, m] - i c[1, l, m]) / sqrt 2,

and that is the whole of the conversion.
"""
from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import sph_harm_y

if TYPE_CHECKING:
    from .sampling import AngularGrid

__all__ = ["real_harmonics", "synthesise", "synthesise_grid", "analyse_grid",
           "packing"]


def real_harmonics(lmax: int, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """The basis up to degree `lmax` at colatitudes `theta` and longitudes
    `phi` (broadcast together): shape (2, lmax + 1, lmax + 1) + points."""
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    shape = np.broadcast_shapes(theta.shape, phi.shape)
    theta = np.broadcast_to(theta, shape)
    phi = np.broadcast_to(phi, shape)
    out = np.zeros((2, lmax + 1, lmax + 1) + shape)
    for l in range(lmax + 1):
        m = np.arange(l + 1)
        y = sph_harm_y(l, m.reshape((-1,) + (1,) * len(shape)), theta[None], phi[None])
        sign = ((-1.0) ** m).reshape((-1,) + (1,) * len(shape))
        y = y * sign
        out[0, l, 0] = y[0].real
        if l:
            out[0, l, 1:l + 1] = np.sqrt(2.0) * y[1:].real
            out[1, l, 1:l + 1] = np.sqrt(2.0) * y[1:].imag
    return out


def synthesise(coeffs: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """sum_lm c_lm Y_lm(theta, phi) for coefficients of shape
    (2, lmax + 1, lmax + 1), or that shape followed by the points' shape
    when the coefficients vary from point to point."""
    c = np.asarray(coeffs, dtype=float)
    if c.ndim < 3 or c.shape[0] != 2 or c.shape[1] != c.shape[2]:
        raise ValueError("coefficients must have shape (2, lmax + 1, lmax + 1, ...)")
    Y = real_harmonics(c.shape[1] - 1, theta, phi)
    if c.ndim == 3:
        c = np.broadcast_to(c.reshape(c.shape + (1,) * (Y.ndim - 3)), Y.shape)
    if c.shape != Y.shape:
        raise ValueError(f"coefficients of shape {c.shape} do not match points "
                         f"of shape {Y.shape[3:]}")
    return np.einsum("slm...,slm...->...", c, Y)


def packing(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index arrays (s, l, m) of the (lmax + 1)^2 entries of a coefficient
    array that belong to a harmonic, in the order of a vector of them: by
    degree l, then s = 0 with m = 0 ... l, then s = 1 with m = 1 ... l.
    So `coeffs[s, l, m]` is the vector of an array, and
    `out[s, l, m] = vector` puts one back."""
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    s, l, m = [], [], []
    for deg in range(lmax + 1):
        s += [0] * (deg + 1) + [1] * deg
        l += [deg] * (2 * deg + 1)
        m += list(range(deg + 1)) + list(range(1, deg + 1))
    return np.array(s), np.array(l), np.array(m)


# -- grids, through ducc0 ------------------------------------------------------

def _sht() -> ModuleType:
    """ducc0's spherical-harmonic transforms; ducc0 is imported here and
    nowhere else in planetmodel."""
    try:
        import ducc0
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ImportError(
            "the grid synthesis and analysis of planetmodel.harmonics need "
            "ducc0.  Install it with:\n"
            "    pip install 'planetmodel[harmonics]'      "
            "(or: poetry install --extras harmonics)"
        ) from exc
    return ducc0.sht


def _rings(grid: AngularGrid) -> str:
    """ducc0's name for the grid's ring layout, once its colatitudes are
    checked against its kind's own, or a ValueError naming what is wrong.
    An equiangular grid is Clenshaw-Curtis with its poles and Fejer's
    first rule without."""
    from .sampling import driscoll_healy, equiangular, gauss_legendre

    kinds = ("gauss_legendre", "driscoll_healy", "equiangular")
    if grid.kind not in kinds or grid.lmax is None:
        raise ValueError(
            f"grid synthesis and analysis take a grid of a shipped kind with a "
            f"band, one of {kinds}; got a {grid.kind!r} grid")
    nt = grid.ntheta
    if grid.kind == "gauss_legendre":
        name, own = "GL", gauss_legendre(nt - 1)
    elif grid.kind == "driscoll_healy":
        name, own = "DH", driscoll_healy(max(nt // 2 - 1, 0))
    elif grid.pole_nodes[0] and nt >= 2:
        name, own = "CC", equiangular(nt, grid.nphi, poles=True)
    else:
        name, own = "F1", equiangular(nt, grid.nphi)
    if own.ntheta != nt or not np.allclose(grid.colatitudes, own.colatitudes,
                                           rtol=0.0, atol=1e-12):
        raise ValueError(
            f"the {nt} colatitudes are not those of a {grid.kind!r} grid "
            f"({name} rings in ducc0's terms)")
    if grid.lmax > own.lmax:
        raise ValueError(
            f"a {grid.kind!r} grid of {nt} colatitudes resolves degrees up to "
            f"{own.lmax}, not the {grid.lmax} it claims")
    if grid.nphi < 2 * grid.lmax + 1:
        raise ValueError(
            f"a grid of band {grid.lmax} needs at least 2 lmax + 1 = "
            f"{2 * grid.lmax + 1} longitudes to resolve its orders; this grid "
            f"has {grid.nphi}")
    if not np.allclose(np.diff(grid.longitudes), 2.0 * np.pi / grid.nphi,
                       rtol=0.0, atol=1e-12):
        raise ValueError("the longitudes are not equally spaced around the circle")
    return name


def _healpy(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """(l, m) of each complex coefficient in healpy's order: m outermost,
    then l = m ... lmax."""
    m = np.concatenate([np.full(lmax + 1 - k, k) for k in range(lmax + 1)])
    l = np.concatenate([np.arange(k, lmax + 1) for k in range(lmax + 1)])
    return l, m


def _to_complex(c: np.ndarray) -> np.ndarray:
    """ducc0's coefficients of real ones of shape (2, L + 1, L + 1, K):
    shape (K, 1, nalm), one row per trailing index."""
    l, m = _healpy(c.shape[1] - 1)
    scale = np.where(m == 0, 1.0, (-1.0) ** m / np.sqrt(2.0))[:, None]
    a = scale * (c[0, l, m] - 1j * np.where(m == 0, 0.0, 1.0)[:, None] * c[1, l, m])
    return np.ascontiguousarray(a.T[:, None, :])


def _to_real(a: np.ndarray, lmax: int) -> np.ndarray:
    """Real coefficients (2, lmax + 1, lmax + 1, K) of ducc0's (K, nalm)."""
    l, m = _healpy(lmax)
    scale = np.where(m == 0, 1.0, (-1.0) ** m * np.sqrt(2.0))[:, None]
    c = np.zeros((2, lmax + 1, lmax + 1, a.shape[0]))
    c[0, l, m] = scale * a.T.real
    c[1, l, m] = np.where(m == 0, 0.0, -1.0)[:, None] * scale * a.T.imag
    return c


def synthesise_grid(coeffs: ArrayLike, grid: AngularGrid) -> np.ndarray:
    """Values on a grid of a shipped kind from coefficients, through ducc0.

    `coeffs` has shape (2, l + 1, l + 1) or that followed by any extra
    axes, a radial axis say, with l at most the grid's band; the result
    has the extra axes first and then (ntheta, nphi), the layout of a
    sample.  Exact for a field band-limited to the grid's degree.
    """
    sht = _sht()
    rings = _rings(grid)
    c = np.asarray(coeffs, dtype=float)
    if c.ndim < 3 or c.shape[0] != 2 or c.shape[1] != c.shape[2]:
        raise ValueError("coefficients must have shape (2, l + 1, l + 1, ...)")
    L = c.shape[1] - 1
    if L > grid.lmax:
        raise ValueError(
            f"coefficients to degree {L} on a grid of band {grid.lmax}: the grid "
            f"cannot hold them; use a grid of band at least {L}")
    extra = c.shape[3:]
    alm = _to_complex(c.reshape(c.shape[:3] + (-1,)))
    out = np.empty((alm.shape[0], grid.ntheta, grid.nphi))
    for k in range(alm.shape[0]):
        out[k] = sht.synthesis_2d(alm=alm[k], spin=0, lmax=L, geometry=rings,
                                  ntheta=grid.ntheta, nphi=grid.nphi,
                                  phi0=float(grid.longitudes[0]))[0]
    return out.reshape(extra + (grid.ntheta, grid.nphi))


def analyse_grid(values: ArrayLike, grid: AngularGrid, *,
                 lmax: int | None = None) -> np.ndarray:
    """Coefficients from values on a grid of a shipped kind, through ducc0.

    `values` has shape (..., ntheta, nphi), any extra axes first; the
    result is (2, lmax + 1, lmax + 1) followed by the extra axes, with
    `lmax` the grid's band by default or a smaller degree to stop at.
    The inverse of `synthesise_grid` for a field band-limited to the
    grid; a projection otherwise.
    """
    sht = _sht()
    rings = _rings(grid)
    band = grid.lmax
    lmax = band if lmax is None else int(lmax)
    if not 0 <= lmax <= band:
        raise ValueError(f"lmax must lie in 0..{band}, got {lmax}")
    v = np.asarray(values, dtype=float)
    if v.ndim < 2 or v.shape[-2:] != (grid.ntheta, grid.nphi):
        raise ValueError(
            f"values must have trailing shape {(grid.ntheta, grid.nphi)}, "
            f"got {v.shape}")
    extra = v.shape[:-2]
    flat = v.reshape((-1, 1) + v.shape[-2:])
    a = np.empty((flat.shape[0], (lmax + 1) * (lmax + 2) // 2), dtype=complex)
    for k in range(flat.shape[0]):
        a[k] = sht.analysis_2d(map=np.ascontiguousarray(flat[k]), spin=0,
                               lmax=lmax, geometry=rings,
                               phi0=float(grid.longitudes[0]))[0]
    return _to_real(a, lmax).reshape((2, lmax + 1, lmax + 1) + extra)
