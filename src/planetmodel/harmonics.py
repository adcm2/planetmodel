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
definition, and the oracle everything else is held to, to degree
`SCIPY_LMAX`, above which scipy's harmonics overflow and the route is
refused.  `synthesise_grid`
and `analyse_grid` go between coefficients and values on a grid of a
shipped kind -- Gauss-Legendre, Driscoll-Healy, or equiangular with or
without the poles, from `planetmodel.sampling` -- through ducc0's
transforms: fast, exact for a field band-limited to the grid's band,
and needing the `harmonics` extra.  The longitudes may start anywhere
and number anything from 2 lmax + 1 up, provided they are equally
spaced.  `synthesise_points` is the fast counterpart of `synthesise` at
arbitrary points, through ducc0's non-uniform synthesis, accurate to
about `POINT_EPSILON` of the field's largest value rather than to
rounding, and `synthesise_points_gradient` gives the two angular derivatives
there the same way.  ducc0 holds complex coefficients a_lm for m >= 0 in healpy's
order, with the Condon-Shortley phase; the real ones here are

    a_l0 = c[0, l, 0],    a_lm = (-1)^m (c[0, l, m] - i c[1, l, m]) / sqrt 2,

and that is the whole of the conversion.

**Other conventions.**  Published coefficients come in other
normalisations and with or without the Condon-Shortley phase;
`from_convention` gives the library's coefficients from them and
`to_convention` the reverse, in the layout above and in pyshtools'
terms where they have one.  `normalization` is one of `NORMALIZATIONS`:

    "ortho"      the library's own: int Y^2 dOmega = 1
    "4pi"        geodesy's: int Y^2 dOmega = 4 pi, so c = sqrt(4 pi) C
    "schmidt"    Schmidt's semi-normalised: int Y^2 dOmega = 4 pi / (2l + 1)
    "unnorm"     the bare P_l^m(cos theta) cos(m phi) and sin(m phi)
    "woodhouse"  N_lm P_l^m(cos theta) cos(m phi) and sin(m phi), N_lm as
                 above with no sqrt 2: the real and imaginary parts of the
                 complex orthonormal harmonics, int Y^2 dOmega = 1/2 for m > 0

and `csphase` is 1 where the source's Legendre functions leave out the
Condon-Shortley phase (-1)^m and -1 where they include it.  With
`csphase=-1`, "woodhouse" is the form of the Woodhouse-Ritsema
coefficient files.  Entries that belong to no harmonic are left as they
are; the bare functions overflow at high degree, as the convention
does.
"""
from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln, sph_harm_y_all

from .frames import wrapped

if TYPE_CHECKING:
    from .sampling import AngularGrid

__all__ = ["real_harmonics", "real_harmonics_gradient", "synthesise",
           "synthesise_gradient", "synthesise_grid", "analyse_grid",
           "synthesise_points", "synthesise_points_gradient", "packing",
           "from_convention", "to_convention", "NORMALIZATIONS",
           "POINT_EPSILON", "SCIPY_LMAX"]

#: The relative accuracy asked of ducc0's synthesis at arbitrary points.
POINT_EPSILON = 1e-13

#: The highest degree to which scipy's harmonics are finite: from degree
#: 646 they overflow at every colatitude and order.
SCIPY_LMAX = 645

#: The normalisations `from_convention` and `to_convention` take.
NORMALIZATIONS = ("ortho", "4pi", "schmidt", "unnorm", "woodhouse")

#: A basis of the real harmonics: (lmax, theta, phi) -> (2, L + 1, L + 1) + points.
type Basis = Callable[[int, np.ndarray, np.ndarray], np.ndarray]


def _complex_basis(lmax: int, theta: np.ndarray, phi: np.ndarray, *,
                   gradient: bool) -> tuple[np.ndarray, ...]:
    """scipy's complex harmonics of orders 0 ... lmax, the Condon-Shortley
    phase removed: (L + 1, L + 1) + points, with their (d/dtheta, d/dphi)
    when `gradient` asks."""
    if lmax > SCIPY_LMAX:
        raise ValueError(
            f"scipy's harmonics overflow above degree {SCIPY_LMAX}, and degree "
            f"{lmax} is asked for: synthesise through ducc0 (synthesise_points, "
            "synthesise_grid), the harmonics extra: pip install "
            "'planetmodel[harmonics]'")
    m = np.arange(lmax + 1).reshape((1, -1) + (1,) * theta.ndim)
    sign = (-1.0) ** m
    if not gradient:
        Y = sph_harm_y_all(lmax, lmax, theta, wrapped(phi))
        return (sign * Y[:, :lmax + 1],)
    Y, dY = sph_harm_y_all(lmax, lmax, theta, wrapped(phi), diff_n=1)
    return (sign * Y[:, :lmax + 1], sign * dY[:, :lmax + 1, ..., 0],
            sign * dY[:, :lmax + 1, ..., 1])


def _real(y: np.ndarray, lmax: int) -> np.ndarray:
    """The real basis (2, L + 1, L + 1) + points from the complex one."""
    out = np.zeros((2,) + y.shape)
    scale = np.full(lmax + 1, np.sqrt(2.0))
    scale[0] = 1.0
    scale = scale.reshape((1, -1) + (1,) * (y.ndim - 2))
    out[0] = scale * y.real
    out[1] = np.sqrt(2.0) * y.imag
    out[1][:, 0] = 0.0
    keep = np.arange(lmax + 1)[None, :] <= np.arange(lmax + 1)[:, None]
    return out * keep.reshape(keep.shape + (1,) * (y.ndim - 2))


def _points(lmax: int, theta: ArrayLike, phi: ArrayLike
            ) -> tuple[int, np.ndarray, np.ndarray]:
    """The degree and the points broadcast together, checked."""
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                     np.asarray(phi, dtype=float))
    return lmax, theta, phi


def real_harmonics(lmax: int, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """The basis up to degree `lmax` at colatitudes `theta` and longitudes
    `phi` (broadcast together): shape (2, lmax + 1, lmax + 1) + points."""
    lmax, theta, phi = _points(lmax, theta, phi)
    return _real(_complex_basis(lmax, theta, phi, gradient=False)[0], lmax)


def real_harmonics_gradient(lmax: int, theta: ArrayLike, phi: ArrayLike
                            ) -> tuple[np.ndarray, np.ndarray]:
    """(d/dtheta, d/dphi) of the basis up to degree `lmax`, each of the
    shape `real_harmonics` gives, through scipy's derivatives of the
    complex harmonics."""
    lmax, theta, phi = _points(lmax, theta, phi)
    _, dt, dp = _complex_basis(lmax, theta, phi, gradient=True)
    return _real(dt, lmax), _real(dp, lmax)


def _coefficients(coeffs: ArrayLike) -> np.ndarray:
    """A coefficient array (2, L + 1, L + 1, ...) as floats, or the error."""
    c = np.asarray(coeffs, dtype=float)
    if c.ndim < 3 or c.shape[0] != 2 or c.shape[1] != c.shape[2]:
        raise ValueError("coefficients must have shape (2, lmax + 1, lmax + 1, ...)")
    return c


def _contract(c: np.ndarray, theta: np.ndarray, phi: np.ndarray,
              basis: Basis, *, index: np.ndarray | None = None) -> np.ndarray:
    """sum_slm c[slm] B_slm at flat points, a block of points at a time so
    that the basis never holds more than `_BLOCK` entries; `c` is
    (2, L + 1, L + 1), or that followed by one column per point, or by
    columns that `index` picks for each point."""
    L = c.shape[1] - 1
    out = np.empty(theta.size)
    step = max(1, _BLOCK // (2 * (L + 1) ** 2))
    for i in range(0, theta.size, step):
        block = slice(i, i + step)
        B = basis(L, theta[block], phi[block])
        if c.ndim == 3:
            out[block] = np.einsum("slm,slmp->p", c, B)
        else:
            columns = c[..., block] if index is None else c[..., index[block]]
            out[block] = np.einsum("slmp,slmp->p", columns, B)
    return out


#: The basis entries evaluated at once by a synthesis at points.
_BLOCK = 1 << 22


def synthesise_gradient(coeffs: ArrayLike, theta: ArrayLike, phi: ArrayLike
                        ) -> tuple[np.ndarray, np.ndarray]:
    """(d/dtheta, d/dphi) of sum_lm c_lm Y_lm at points, for coefficients
    of shape (2, lmax + 1, lmax + 1): the derivatives of `synthesise`,
    through scipy, a block of points at a time."""
    c = _coefficients(coeffs)
    if c.ndim != 3:
        raise ValueError("coefficients must have shape (2, lmax + 1, lmax + 1)")
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                     np.asarray(phi, dtype=float))
    t, p, L = theta.ravel(), phi.ravel(), c.shape[1] - 1
    dt, dp = np.empty(t.size), np.empty(t.size)
    step = max(1, _BLOCK // (4 * (L + 1) ** 2))
    for i in range(0, t.size, step):
        block = slice(i, i + step)
        bt, bp = real_harmonics_gradient(L, t[block], p[block])
        dt[block] = np.einsum("slm,slmp->p", c, bt)
        dp[block] = np.einsum("slm,slmp->p", c, bp)
    return dt.reshape(theta.shape), dp.reshape(theta.shape)


def synthesise(coeffs: ArrayLike, theta: ArrayLike, phi: ArrayLike, *,
               index: ArrayLike | None = None) -> np.ndarray:
    """sum_lm c_lm Y_lm(theta, phi) for coefficients of shape
    (2, lmax + 1, lmax + 1), or that shape followed by the points' shape
    when the coefficients vary from point to point, or followed by one
    axis of n columns with `index`, of the points' shape, naming each
    point's column (so that coefficients shared by many points are held
    once).  Through scipy, a block of points at a time."""
    c = _coefficients(coeffs)
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                     np.asarray(phi, dtype=float))
    shape = theta.shape
    if index is not None:
        idx = np.broadcast_to(np.asarray(index, dtype=np.intp), shape).ravel()
        if c.ndim != 4 or (idx.size and not 0 <= idx.min() <= idx.max() < c.shape[3]):
            raise ValueError("with index, coefficients have shape (2, l + 1, l + 1, "
                             "n) and index picks columns in 0..n-1")
        return _contract(c, theta.ravel(), phi.ravel(), real_harmonics,
                         index=idx).reshape(shape)
    if c.ndim > 3:
        if c.shape[3:] != shape:
            raise ValueError(f"coefficients of shape {c.shape} do not match points "
                             f"of shape {shape}")
        c = c.reshape(c.shape[:3] + (-1,))
    return _contract(c, theta.ravel(), phi.ravel(), real_harmonics).reshape(shape)


def _convention_factors(lmax: int, normalization: str, csphase: int
                        ) -> np.ndarray:
    """c / C for every entry of a (2, L + 1, L + 1) array: a coefficient C
    of the convention times this is the library's; one where no harmonic
    is."""
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got "
                         f"{normalization!r}")
    if csphase not in (1, -1):
        raise ValueError(f"csphase must be 1 (without the Condon-Shortley phase) "
                         f"or -1 (with it), got {csphase!r}")
    l, m = np.meshgrid(np.arange(lmax + 1.0), np.arange(lmax + 1.0), indexing="ij")
    held = m <= l
    if normalization == "ortho":
        f = np.ones_like(l)
    elif normalization == "4pi":
        f = np.full_like(l, np.sqrt(4.0 * np.pi))
    elif normalization == "schmidt":
        f = np.sqrt(4.0 * np.pi / (2.0 * l + 1.0))
    elif normalization == "woodhouse":
        f = np.where(m == 0, 1.0, 1.0 / np.sqrt(2.0))
    else:                                           # "unnorm": 1 / (sqrt(2) N_lm)
        d = np.where(held, l - m, 0.0)
        log_n = 0.5 * (np.log((2.0 * l + 1.0) / (4.0 * np.pi))
                       + gammaln(d + 1.0) - gammaln(np.where(held, l + m, 0.0) + 1.0))
        f = np.exp(-log_n) / np.where(m == 0, 1.0, np.sqrt(2.0))
    if csphase == -1:
        f = f * (-1.0) ** m
    return np.stack([np.where(held, f, 1.0), np.where(held & (m > 0), f, 1.0)])


def from_convention(coefficients: ArrayLike, *, normalization: str,
                    csphase: int) -> np.ndarray:
    """The library's coefficients of the field whose coefficients in a
    source's convention are `coefficients`, of shape (2, L + 1, L + 1)
    followed by any extra axes: `normalization` one of `NORMALIZATIONS`,
    `csphase` 1 without the Condon-Shortley phase and -1 with it, as the
    module docstring states them."""
    c = _coefficients(coefficients)
    f = _convention_factors(c.shape[1] - 1, normalization, csphase)
    return c * f.reshape(f.shape + (1,) * (c.ndim - 3))


def to_convention(coefficients: ArrayLike, *, normalization: str,
                  csphase: int) -> np.ndarray:
    """The coefficients in a source's convention of the field whose
    library coefficients are `coefficients`: the inverse of
    `from_convention`."""
    c = _coefficients(coefficients)
    f = _convention_factors(c.shape[1] - 1, normalization, csphase)
    return c / f.reshape(f.shape + (1,) * (c.ndim - 3))


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


def _points_and_coefficients(coeffs: ArrayLike, theta: ArrayLike, phi: ArrayLike
                             ) -> tuple[np.ndarray, int, np.ndarray,
                                        tuple[int, ...], tuple[int, ...]]:
    """ducc0's coefficients (K, 1, nalm), their degree, and the points
    (P, 2) for a synthesis at arbitrary points, with the extra and point
    shapes to restore."""
    c = np.asarray(coeffs, dtype=float)
    if c.ndim < 3 or c.shape[0] != 2 or c.shape[1] != c.shape[2]:
        raise ValueError("coefficients must have shape (2, l + 1, l + 1, ...)")
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                     np.asarray(phi, dtype=float))
    if np.any(theta < 0.0) or np.any(theta > np.pi):
        raise ValueError("colatitudes must lie in [0, pi]")
    loc = np.ascontiguousarray(np.stack([theta.ravel(), wrapped(phi.ravel())],
                                        axis=-1))
    alm = _to_complex(c.reshape(c.shape[:3] + (-1,)))
    return alm, c.shape[1] - 1, loc, c.shape[3:], theta.shape


def synthesise_points(coeffs: ArrayLike, theta: ArrayLike, phi: ArrayLike
                      ) -> np.ndarray:
    """sum_lm c_lm Y_lm at arbitrary points, fast, through ducc0.

    `coeffs` has shape (2, l + 1, l + 1) or that followed by any extra
    axes; `theta` in [0, pi] and `phi`, in any range, broadcast
    together.  The result has the extra axes first and then the points'
    shape.  It agrees with `synthesise` to about `POINT_EPSILON` of the
    field's largest value, the accuracy asked of ducc0's non-uniform
    synthesis, at a cost nearly independent of the number of points.
    """
    sht = _sht()
    alm, L, loc, extra, points = _points_and_coefficients(coeffs, theta, phi)
    out = np.empty((alm.shape[0], loc.shape[0]))
    if loc.shape[0]:
        for k in range(alm.shape[0]):
            out[k] = sht.synthesis_general(alm=alm[k], spin=0, lmax=L, loc=loc,
                                           epsilon=POINT_EPSILON)[0]
    return out.reshape(extra + points)


def synthesise_points_gradient(coeffs: ArrayLike, theta: ArrayLike, phi: ArrayLike
                        ) -> tuple[np.ndarray, np.ndarray]:
    """(d/dtheta, d/dphi) of sum_lm c_lm Y_lm at arbitrary points, through
    ducc0, in the layout and to the accuracy of `synthesise_points`.

    ducc0 gives the gradient's components d/dtheta and (1 / sin theta)
    d/dphi; the second is multiplied by sin theta here, so that at a pole
    d/dphi is zero, as it is for any field smooth there.
    """
    sht = _sht()
    alm, L, loc, extra, points = _points_and_coefficients(coeffs, theta, phi)
    dt = np.zeros((alm.shape[0], loc.shape[0]))
    dp = np.zeros((alm.shape[0], loc.shape[0]))
    if loc.shape[0] and L > 0:
        for k in range(alm.shape[0]):
            g = sht.synthesis_general(alm=alm[k], spin=1, lmax=L, loc=loc,
                                      epsilon=POINT_EPSILON, mode="DERIV1")
            dt[k], dp[k] = g[0], g[1] * np.sin(loc[:, 0])
    return dt.reshape(extra + points), dp.reshape(extra + points)
