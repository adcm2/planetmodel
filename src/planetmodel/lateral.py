"""Values between the nodes of an angular grid: the lateral rules.

A field that varies with direction is stored at the nodes of an
`AngularGrid` and read between them by one of three rules, the same
here and in every reader of the library's files:

    linear     bilinear in (theta, phi) on the padded grid
    cubic      the tensor-product cubic spline on the padded grid,
               not-a-knot at its four edges
    harmonic   analysis on the grid and synthesis at the point, exact
               for a field band-limited to the grid's band (to the
               accuracy of `harmonics.synthesise_points`); only on a
               grid of a shipped kind with a band, and through ducc0
               (the `harmonics` extra)

**The padded grid.**  The local rules are rules on a rectangle of
(theta, phi), so the sphere is first laid out as one.  The longitudes
phi_0 < ... < phi_{n-1} gain the wrap column phi_0 + 2 pi, a copy of the
first, and g = min(3, n - 1) ghost columns beyond each end, copies of
the columns a period away: phi_{n-g} - 2 pi ... phi_{n-1} - 2 pi before
and phi_1 + 2 pi ... phi_g + 2 pi after.  The end conditions of the
spline in longitude then act at least g cells from any point that is
read, and the interpolant is periodic to that extent, though not
exactly.  The colatitudes gain a row at each pole the grid lacks as a
node, holding the field's value at that pole across every column; a
pole that is a node keeps its own row.  No rows are added beyond the
poles, so the not-a-knot conditions in colatitude act at the poles
themselves.  A point is read at its colatitude, which must lie in
[0, pi], and at its longitude reduced into [phi_0, phi_0 + 2 pi).  The
bilinear rule is the tensor-product spline of degree one on the same
grid, so both local rules interpolate the nodes and the polar values.

**The polar values.**  A grid without a pole among its nodes does not
hold the field there, and nothing between the nodes can say what it
is, so the value is given (`poles`, north then south): a model file
stores it, evaluated exactly by the writer.  Without it, a grid with a
band takes it from its own expansion, the value a field band-limited
to the grid has there; any other grid is refused rather than guessed
at.  A scalar has one value at a pole.  A field with components is
interpolated component by component in the Cartesian frame, where it
too is single-valued at a pole; the rotation into that frame belongs to
the field, not to this module.

**Accuracy.**  A local rule wants an oversampled grid.  For a field of
band 16 with coefficients falling as 1/(l + 1), on Gauss-Legendre
grids, the largest error relative to the field's largest value is
about 0.09 for the bicubic rule at the field's own band, 5e-4 at four
times it and 3e-5 at eight, and 0.3, 2e-2 and 5e-3 for the bilinear
one: fourth order against second (the cases of
`tests/test_lateral.py`).  Analysis on an oversampled grid is still
exact to the field's band, so a grid of about four times the band
serves a local reader and a spectral one alike.  Two improvements wait
for a use: a periodic spline condition in longitude in place of the
ghost columns, and smoothness through a pole, where the meridian
splines end and the gradient of the interpolant has a small kink.

`LateralInterpolant` builds a rule once over any array whose two
trailing axes are a grid's; `interpolate` builds one and evaluates it.
Everything is numbers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import NdBSpline, make_interp_spline

from .frames import wrapped
from .harmonics import analyse_grid, real_harmonics, synthesise_points

if TYPE_CHECKING:
    from .sampling import AngularGrid

__all__ = ["METHODS", "GHOSTS", "LateralInterpolant", "interpolate",
           "polar_values"]

#: The lateral rules, in the order of the module docstring.
METHODS = ("linear", "cubic", "harmonic")

#: The ghost columns added beyond each end of the longitudes, at most.
GHOSTS = 3

#: The degree of the tensor-product spline of each local rule.
_DEGREE = {"linear": 1, "cubic": 3}


def polar_values(values: ArrayLike, grid: AngularGrid) -> np.ndarray:
    """The values at the north and south poles from the grid's expansion.

    `values` has shape (..., ntheta, nphi); the result has shape
    (..., 2), north then south: the poles of the field band-limited to
    the grid's band that agrees with the values at the nodes, exact for
    a field of that band.  The grid must be of a shipped kind with a
    band, and the analysis needs the `harmonics` extra.
    """
    c = analyse_grid(values, grid)
    Y = real_harmonics(c.shape[1] - 1, np.array([0.0, np.pi]), np.zeros(2))
    return np.einsum("slm...,slmq->...q", c, Y)


def _padded(v: np.ndarray, grid: AngularGrid, poles: np.ndarray | None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(colatitudes, longitudes, values) of the padded grid of the module
    docstring, the values of shape (..., rows, columns)."""
    theta, phi = grid.colatitudes, grid.longitudes
    n = phi.size
    g = min(GHOSTS, n - 1)
    two_pi = 2.0 * np.pi
    phi_p = np.concatenate([phi[n - g:] - two_pi, phi, phi[:1] + two_pi,
                            phi[1:g + 1] + two_pi])
    v = np.concatenate([v[..., n - g:], v, v[..., :1], v[..., 1:g + 1]], axis=-1)
    north, south = grid.pole_nodes
    theta_p, rows = [theta], [v]
    if not north:
        theta_p.insert(0, [0.0])
        rows.insert(0, np.broadcast_to(poles[..., 0, None, None],
                                       v.shape[:-2] + (1, phi_p.size)))
    if not south:
        theta_p.append([np.pi])
        rows.append(np.broadcast_to(poles[..., 1, None, None],
                                    v.shape[:-2] + (1, phi_p.size)))
    return np.concatenate(theta_p), phi_p, np.concatenate(rows, axis=-2)


class LateralInterpolant:
    """One lateral rule, built once over an array of values on a grid.

    `values` has shape (..., ntheta, nphi), the grid's two axes last; the
    leading shape is `shape`, and calling the interpolant at
    colatitudes and longitudes that broadcast to a shape P gives an
    array of shape `shape + P`.  `method` is one of `METHODS`.  `poles`,
    of shape `shape + (2,)`, holds the values at the north and south
    poles for a local rule, consulted only for a pole the grid lacks as
    a node; when it is None a grid with a band takes them from its
    expansion (`polar_values`, needing the `harmonics` extra), and any
    other grid lacking a pole is refused.  The harmonic rule needs no
    polar values, since its expansion holds them, and `poles` is not
    consulted for it.  Values are real.
    """

    def __init__(self, values: ArrayLike, grid: AngularGrid, *, method: str,
                 poles: ArrayLike | None = None) -> None:
        from .sampling import AngularGrid
        if not isinstance(grid, AngularGrid):
            raise TypeError(f"grid must be an AngularGrid, got {type(grid).__name__}")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if np.iscomplexobj(values) or (poles is not None and np.iscomplexobj(poles)):
            raise TypeError("the lateral rules take real values")
        v = np.asarray(values, dtype=float)
        if v.ndim < 2 or v.shape[-2:] != (grid.ntheta, grid.nphi):
            raise ValueError(f"values must have trailing shape "
                             f"{(grid.ntheta, grid.nphi)}, got {v.shape}")
        self._grid, self._method, self._shape = grid, method, v.shape[:-2]
        if method == "harmonic":
            self._coeffs = analyse_grid(v, grid)
            return
        P = None if all(grid.pole_nodes) else self._polar(v, poles)
        theta_p, phi_p, V = _padded(v, grid, P)
        k = _DEGREE[method]
        if theta_p.size <= k or phi_p.size <= k:
            raise ValueError(
                f"the {method} rule needs at least {k + 1} rows and columns on "
                f"the padded grid; this one has {theta_p.size} x {phi_p.size}")
        V = np.moveaxis(V, (-2, -1), (0, 1))
        along_theta = make_interp_spline(theta_p, V, k=k, axis=0)
        along_phi = make_interp_spline(phi_p, np.swapaxes(along_theta.c, 0, 1),
                                       k=k, axis=0)
        self._spline = NdBSpline((along_theta.t, along_phi.t),
                                 np.swapaxes(along_phi.c, 0, 1), k)

    def _polar(self, v: np.ndarray, poles: ArrayLike | None) -> np.ndarray:
        """The polar values given, checked, or found from the grid's band."""
        grid, want = self._grid, self._shape + (2,)
        if poles is not None:
            P = np.asarray(poles, dtype=float)
            if P.shape != want:
                raise ValueError(f"poles must have shape {want}, the values' "
                                 f"leading shape and (north, south); got {P.shape}")
            return P
        if grid.kind == "custom" or grid.lmax is None:
            raise ValueError(
                "the grid lacks a pole among its nodes and has no band to find "
                "the field there from, so the polar values must be given: pass "
                f"poles=, the values at the north and south poles, shape {want}")
        return polar_values(v, grid)

    @property
    def grid(self) -> AngularGrid:
        """The angular grid the values are on."""
        return self._grid

    @property
    def method(self) -> str:
        """The lateral rule, one of `METHODS`."""
        return self._method

    @property
    def shape(self) -> tuple[int, ...]:
        """The leading shape of the values, carried through to every result."""
        return self._shape

    def __call__(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """The values at colatitudes `theta` in [0, pi] and longitudes `phi`
        (any range), broadcast together: shape `shape` followed by theirs."""
        theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                         np.asarray(phi, dtype=float))
        if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(phi))):
            raise ValueError("the points must be finite")
        if np.any(theta < 0.0) or np.any(theta > np.pi):
            raise ValueError("colatitudes must lie in [0, pi]")
        points = theta.shape
        t, p = theta.ravel(), phi.ravel()
        if self._method == "harmonic":
            out = synthesise_points(self._coeffs, t, p)
        else:
            p = wrapped(p, start=float(self._grid.longitudes[0]))
            out = np.moveaxis(self._spline(np.stack([t, p], axis=-1)), 0, -1)
        return out.reshape(self._shape + points)

    def __repr__(self) -> str:
        return f"LateralInterpolant({self._method}, {self._shape} on {self._grid!r})"


def interpolate(values: ArrayLike, grid: AngularGrid, theta: ArrayLike,
                phi: ArrayLike, *, method: str,
                poles: ArrayLike | None = None) -> np.ndarray:
    """Values on a grid read at points by a lateral rule.

    `LateralInterpolant(values, grid, method=method, poles=poles)`
    evaluated at (theta, phi): the leading shape of `values` followed by
    the broadcast shape of the points.  Build the interpolant itself to
    read the same values more than once.
    """
    return LateralInterpolant(values, grid, method=method, poles=poles)(theta, phi)
