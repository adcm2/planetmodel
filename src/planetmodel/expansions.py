"""Fields and surface functions given by spherical-harmonic expansions.

A `HarmonicField` is a scalar field on one interval written as

    f(r, theta, phi) = sum_lm c_lm(r) Y_lm(theta, phi),

with Y_lm the real orthonormal harmonics of `planetmodel.harmonics` and
the coefficients c_lm(r) one scipy piecewise polynomial of radius whose
values have shape (2, lmax + 1, lmax + 1), in that module's layout.  It
is exact in r on polynomial pieces and exact in angle to rounding: the
form of a model built on spherical harmonics and radial splines, the
oracle the gridded fields are held to, and what a spherical random
field is.  A `HarmonicFunction` is the same for a function of (theta,
phi) alone, a fixed coefficient array, with its two angular derivatives,
so that it serves as a relief of `layer_linear` and as a surface
quantity.

Both evaluate through scipy's harmonics (`harmonics.synthesise`), a
block of points at a time, at a cost of order (lmax + 1)^2 a point and
with the coefficients held once for each distinct radius; the points of
a radius shared by `FAST_POINTS` or more (a sphere, a grid at a node of
radius), and a harmonic function's `FAST_POINTS` or more, go through
ducc0's synthesis at points where the `harmonics` extra is installed,
to `harmonics.POINT_EPSILON` of the field's largest value and some
thousand times faster, scipy remaining the definition the tests hold it
to.  Above `harmonics.SCIPY_LMAX`, where scipy's harmonics overflow,
every point goes through ducc0, and without it the evaluation is
refused by name.  On the nodes of a grid, `on_grid` takes ducc0's grid
transforms (a shipped kind of band at least lmax) or its synthesis at
points (any other grid).  Entries of a coefficient array
that belong to no harmonic (the sine slot of order zero, orders above
the degree) must be zero.  Everything is numbers.
"""
from __future__ import annotations

import copy
from math import factorial
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import BSpline, PPoly

from .character import SCALAR, Character
from .fields import FieldBase, RadialField, radial_pieces_of
from .harmonics import SCIPY_LMAX, synthesise, synthesise_gradient
from .layerfunction import PolynomialLayer, as_interval

if TYPE_CHECKING:
    from .sampling import AngularGrid

__all__ = ["HarmonicField", "HarmonicFunction", "FAST_POINTS"]

#: The points sharing one radius from which ducc0 synthesises them, where
#: it is installed.
FAST_POINTS = 512


def _harmonic_entries(lmax: int) -> np.ndarray:
    """The (2, L + 1, L + 1) mask of the entries that belong to a harmonic."""
    l = np.arange(lmax + 1)
    keep = l[None, :] <= l[:, None]
    mask = np.stack([keep, keep.copy()])
    mask[1, :, 0] = False
    return mask


def _ppoly_of(spline: BSpline) -> PPoly:
    """A B-spline as a piecewise polynomial, any trailing shape kept: on
    each piece the Taylor coefficients at its left end, from the spline's
    derivatives there."""
    k = spline.k
    x = np.unique(spline.t[k:-k])
    c = np.stack([spline(x[:-1], nu=d) / factorial(d) for d in range(k, -1, -1)])
    return PPoly(c, x)


def _check_layout(c: np.ndarray, owner: str) -> int:
    """The degree of coefficients (..., 2, L + 1, L + 1) whose entries
    outside a harmonic are zero, or the error."""
    if c.ndim < 3 or c.shape[-3] != 2 or c.shape[-2] != c.shape[-1]:
        raise ValueError(f"{owner} coefficients must have values of shape "
                         f"(2, lmax + 1, lmax + 1), got {c.shape}")
    lmax = c.shape[-1] - 1
    if np.any(c[..., ~_harmonic_entries(lmax)]):
        raise ValueError(
            f"{owner} coefficients are non-zero in entries that belong to no "
            "harmonic (the sine slot of order zero, or an order above the "
            "degree)")
    return lmax


class HarmonicField(FieldBase):
    """sum_lm c_lm(r) Y_lm(theta, phi) on one interval, a scalar field.

    `coefficients` is a scipy `PPoly` or `BSpline` of radius whose values
    have shape (2, lmax + 1, lmax + 1), or such an array for coefficients
    constant in radius; beyond its breakpoints a piecewise polynomial
    continues its end pieces.  `character` must be of rank 0.  Values
    are real.
    """

    def __init__(self, interval: tuple[float, float],
                 coefficients: PPoly | BSpline | ArrayLike, *,
                 character: Character = SCALAR, name: str | None = None,
                 rtol: float = 1e-9) -> None:
        if character.rank:
            raise ValueError(f"a harmonic field is of rank 0, not {character}")
        self._interval = as_interval(interval)
        self._character = character
        self._name = name
        self._rtol = float(rtol)
        if isinstance(coefficients, BSpline):
            coefficients = _ppoly_of(coefficients)
        if isinstance(coefficients, PPoly):
            c = np.asarray(coefficients.c)
            if np.iscomplexobj(c):
                raise TypeError("a harmonic field holds real coefficients")
            pp = PPoly(c.astype(float), np.asarray(coefficients.x, dtype=float))
        else:
            c = np.asarray(coefficients)
            if np.iscomplexobj(c):
                raise TypeError("a harmonic field holds real coefficients")
            lo, hi = self._interval
            pp = PPoly(np.asarray(c, dtype=float)[None, None], np.array([lo, hi]))
        self._lmax = _check_layout(pp.c, "a harmonic field's")
        self._pp = pp

    @property
    def coefficients(self) -> PPoly:
        """The coefficients as a piecewise polynomial of radius, values of
        shape (2, lmax + 1, lmax + 1)."""
        return self._pp

    @property
    def lmax(self) -> int:
        """The highest degree of the expansion."""
        return self._lmax

    #: The rules a harmonic field states for itself: exact pieces in
    #: radius and the expansion between nodes.
    radial = "polynomial"
    lateral = "harmonic"

    def radial_pieces(self) -> tuple[np.ndarray, int]:
        """The coefficients' breakpoints in the interval, its ends included,
        and their degree."""
        return radial_pieces_of(self._interval, [self])

    @property
    def ppoly(self) -> PPoly:
        """The coefficients' piecewise polynomial, as `radial_pieces_of`
        reads a layer function."""
        return self._pp

    def on_grid(self, r: ArrayLike, grid: AngularGrid) -> np.ndarray:
        """The values at radii `r`, within the interval, on the nodes of
        `grid`, (r.size, ntheta, nphi): through ducc0 where it is
        installed (its grid transforms on a shipped kind of band at least
        `lmax`, its synthesis at points otherwise), through scipy's
        harmonics point by point without it."""
        r, _, _ = self._points(np.asarray(r, dtype=float).ravel(), 0.0, 0.0)
        c = self.coefficients_at(r)
        try:
            from .harmonics import synthesise_grid, synthesise_points
            if (grid.kind != "custom" and grid.lmax is not None
                    and grid.lmax >= self._lmax):
                return synthesise_grid(c, grid)
            return synthesise_points(c, grid.colatitudes[:, None],
                                     grid.longitudes[None, :])
        except ImportError:
            return self.evaluate(r[:, None, None], grid.colatitudes[None, :, None],
                                 grid.longitudes[None, None, :])

    def coefficients_at(self, r: ArrayLike) -> np.ndarray:
        """c_lm at the radii `r`: shape (2, lmax + 1, lmax + 1) + r's shape."""
        r = np.asarray(r, dtype=float)
        return np.moveaxis(self._pp(r), tuple(range(r.ndim)),
                           tuple(range(3, 3 + r.ndim)))

    def _values(self, r: np.ndarray, theta: np.ndarray | None,
                phi: np.ndarray | None) -> np.ndarray:
        radii, index = np.unique(r, return_inverse=True)
        index = index.reshape(r.shape)
        c = self.coefficients_at(radii)
        counts = np.bincount(index.ravel(), minlength=radii.size)
        many = np.flatnonzero((counts >= FAST_POINTS) | (self._lmax > SCIPY_LMAX))
        if many.size:
            from .harmonics import synthesise_points

            t, p = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                       np.asarray(phi, dtype=float))
            flat, tf, pf = index.ravel(), t.ravel(), p.ravel()
            out = np.empty(flat.size)
            try:
                for u in many:
                    at = flat == u
                    out[at] = synthesise_points(c[..., u], tf[at], pf[at])
            except ImportError:                       # without the harmonics extra
                return synthesise(c, theta, phi, index=index)
            rest = counts[flat] < FAST_POINTS
            if rest.any():
                out[rest] = synthesise(c, tf[rest], pf[rest], index=flat[rest])
            return out.reshape(r.shape)
        return synthesise(c, theta, phi, index=index)

    def spherical_mean(self, *, grid: AngularGrid | None = None) -> RadialField:
        """c_00(r) / sqrt(4 pi), exact polynomial pieces; `grid` is not
        needed."""
        pp = PPoly(self._pp.c[..., 0, 0, 0] / np.sqrt(4.0 * np.pi), self._pp.x)
        return RadialField(self._interval, PolynomialLayer(self._interval, pp),
                           character=self._character, name=self._name)

    def on_interval(self, lo: float, hi: float) -> "HarmonicField":
        """The same expansion on [lo, hi], the end pieces continuing."""
        out = copy.copy(self)
        out._interval = as_interval((lo, hi))
        return out

    def rescaled(self, *, k: float, v: float) -> "HarmonicField":
        """v f(r / k, theta, phi): the breakpoints and interval scaled by k,
        each piece's coefficients by v / k^p for its power p."""
        k, v = float(k), float(v)
        if k <= 0.0:
            raise ValueError(f"the coordinate factor must be positive, got {k}")
        c = self._pp.c
        powers = (c.shape[0] - 1) - np.arange(c.shape[0])
        factor = (v / k ** powers).reshape((-1,) + (1,) * (c.ndim - 1))
        lo, hi = self._interval
        return HarmonicField((k * lo, k * hi), PPoly(c * factor, self._pp.x * k),
                             character=self._character, name=self._name,
                             rtol=self._rtol)

    def renamed(self, name: str | None) -> "HarmonicField":
        """The same field under `name`."""
        out = copy.copy(self)
        out._name = name
        return out

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        lo, hi = self._interval
        return (f"HarmonicField({nm}{self._character} on [{lo:g}, {hi:g}], "
                f"lmax={self._lmax}, {self._pp.x.size - 1} radial pieces)")


class HarmonicFunction:
    """sum_lm c_lm Y_lm(theta, phi): a function of the angles alone.

    `coefficients` has shape (2, lmax + 1, lmax + 1), in the layout of
    `planetmodel.harmonics`.  Calling it gives the values at points;
    `angular_gradient` gives (d/dtheta, d/dphi), so that it is a relief
    for `layer_linear`; `on_grid` the values on a grid's nodes, and
    `mean` the spherical mean.  Values are real.
    """

    def __init__(self, coefficients: ArrayLike) -> None:
        if np.iscomplexobj(coefficients):
            raise TypeError("a harmonic function holds real coefficients")
        c = np.array(coefficients, dtype=float)
        if c.ndim != 3:
            raise ValueError(f"coefficients must have shape (2, lmax + 1, "
                             f"lmax + 1), got {c.shape}")
        self._lmax = _check_layout(c, "a harmonic function's")
        c.setflags(write=False)
        self._c = c

    @property
    def coefficients(self) -> np.ndarray:
        """The coefficient array, (2, lmax + 1, lmax + 1)."""
        return self._c

    @property
    def lmax(self) -> int:
        """The highest degree of the expansion."""
        return self._lmax

    def __call__(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """The values at (theta, phi), broadcast together, by the route of
        the module docstring."""
        t, p = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                   np.asarray(phi, dtype=float))
        if t.size >= FAST_POINTS or self._lmax > SCIPY_LMAX:
            try:
                from .harmonics import synthesise_points
                return synthesise_points(self._c, t, p)
            except ImportError:                       # without the harmonics extra
                pass
        return synthesise(self._c, t, p)

    def angular_gradient(self, theta: ArrayLike, phi: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(d/dtheta, d/dphi) at (theta, phi), broadcast together, by the
        route of the module docstring."""
        t, p = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                   np.asarray(phi, dtype=float))
        if t.size >= FAST_POINTS or self._lmax > SCIPY_LMAX:
            try:
                from .harmonics import synthesise_points_gradient
                return synthesise_points_gradient(self._c, t, p)
            except ImportError:                       # without the harmonics extra
                pass
        return synthesise_gradient(self._c, t, p)

    def on_grid(self, grid: AngularGrid) -> np.ndarray:
        """The values on the nodes of `grid`, (ntheta, nphi), through
        ducc0 where it is installed (its grid transforms on a shipped kind
        of band at least `lmax`, its synthesis at points otherwise)."""
        try:
            from .harmonics import synthesise_grid
            if (grid.kind != "custom" and grid.lmax is not None
                    and grid.lmax >= self._lmax):
                return synthesise_grid(self._c, grid)
        except ImportError:                           # without the harmonics extra
            pass
        return self(grid.colatitudes[:, None], grid.longitudes[None, :])

    def mean(self) -> float:
        """The mean over the sphere, c_00 / sqrt(4 pi)."""
        return float(self._c[0, 0, 0] / np.sqrt(4.0 * np.pi))

    def __repr__(self) -> str:
        return f"HarmonicFunction(lmax={self._lmax})"
