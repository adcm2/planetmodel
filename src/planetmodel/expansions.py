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
block of points at a time, at a cost of order (lmax + 1)^2 a point; on
the nodes of a grid of a shipped kind the grid transforms of
`planetmodel.harmonics` are the fast route, through the `harmonics`
extra.  Entries of a coefficient array that belong to no harmonic (the
sine slot of order zero, orders above the degree) must be zero.
Everything is numbers.
"""
from __future__ import annotations

import copy
from math import factorial
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import BSpline, PPoly

from .character import SCALAR, Character
from .fields import FieldBase, RadialField
from .harmonics import synthesise, synthesise_gradient
from .layerfunction import PolynomialLayer, as_interval

if TYPE_CHECKING:
    from .sampling import AngularGrid

__all__ = ["HarmonicField", "HarmonicFunction"]


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

    def coefficients_at(self, r: ArrayLike) -> np.ndarray:
        """c_lm at the radii `r`: shape (2, lmax + 1, lmax + 1) + r's shape."""
        r = np.asarray(r, dtype=float)
        return np.moveaxis(self._pp(r), tuple(range(r.ndim)),
                           tuple(range(3, 3 + r.ndim)))

    def _values(self, r: np.ndarray, theta: np.ndarray | None,
                phi: np.ndarray | None) -> np.ndarray:
        return synthesise(self.coefficients_at(r), theta, phi)

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
    for `layer_linear`.  Values are real.
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
        """The values at (theta, phi), broadcast together."""
        return synthesise(self._c, theta, phi)

    def angular_gradient(self, theta: ArrayLike, phi: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(d/dtheta, d/dphi) at (theta, phi), broadcast together."""
        return synthesise_gradient(self._c, theta, phi)

    def __repr__(self) -> str:
        return f"HarmonicFunction(lmax={self._lmax})"
