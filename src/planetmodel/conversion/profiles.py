"""Radial profiles: the parameters of a law that vary with radius.

A profile is a function of the reference radius in metres, in SI like
the laws it feeds (`planetmodel.conversion.laws`): an expansivity
alpha(r), a velocity-to-density scaling d ln rho / d ln Vs (r), a
sensitivity d ln Vs / dT (r), a law's exponent.  A law takes numbers;
whoever applies it evaluates the profile at the radii of its points.

- `constant(value)`;
- `polynomial(coefficients, *, scale)`, sum_k c_k (r / scale)^k from the
  constant term up, as an expansivity cubic in r / 2891 km is written;
- `spline(radii, values)`, the natural cubic spline through the knots
  (zero second derivative at both ends), continued beyond them by its
  end cubics, which is what Numerical Recipes' `spline` with natural
  ends and `splint` give, the end interval serving outside the knots;
- `tabulated(radii, values)`, linear between the knots and holding the
  end values beyond them.

`as_profile` makes a number a constant profile, so a law's parameter may
be either (`ProfileLike`).  A profile states what it is in its
`description`, which a parameter set's repr shows.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

__all__ = ["Profile", "ProfileLike", "as_profile", "constant", "polynomial", "spline",
           "tabulated"]


class Profile:
    """A function of the reference radius in metres, its values broadcast
    to the radii's shape as float64; `description` says what it is."""

    def __init__(self, fn: Callable[[np.ndarray], ArrayLike], *,
                 description: str) -> None:
        if not callable(fn):
            raise TypeError(f"a profile is a callable of the radius, got {fn!r}")
        self._fn = fn
        self._description = str(description)

    @property
    def description(self) -> str:
        """What the profile is, in a few words."""
        return self._description

    def __call__(self, r: ArrayLike) -> np.ndarray:
        """The values at radii `r` in metres, of their shape."""
        r = np.asarray(r, dtype=float)
        return np.broadcast_to(np.asarray(self._fn(r), dtype=float), r.shape).copy()

    def __repr__(self) -> str:
        return f"Profile({self._description})"


#: A law's parameter: a number, or a profile of the radius.
type ProfileLike = float | Profile


def as_profile(value: ProfileLike) -> Profile:
    """A profile as it stands; a number as a constant profile."""
    if isinstance(value, Profile):
        return value
    return constant(value)


def constant(value: float) -> Profile:
    """The same value at every radius."""
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"a constant profile is a finite number, got {value!r}")
    return Profile(lambda r: np.full(r.shape, v), description=f"constant {v:g}")


def polynomial(coefficients: Sequence[float], *, scale: float = 1.0) -> Profile:
    """sum_k c_k (r / scale)^k, the coefficients from the constant term up."""
    c = np.asarray(coefficients, dtype=float)
    if c.ndim != 1 or not c.size or not np.all(np.isfinite(c)):
        raise ValueError("a polynomial profile takes finite coefficients, constant "
                         "term first")
    if not scale > 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    s = float(scale)
    return Profile(lambda r: np.polynomial.polynomial.polyval(r / s, c),
                   description=f"polynomial of degree {c.size - 1} in r / {s:g} m")


def _knots(radii: Sequence[float], values: Sequence[float], least: int
           ) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(radii, dtype=float)
    v = np.asarray(values, dtype=float)
    if r.ndim != 1 or r.shape != v.shape or r.size < least:
        raise ValueError(f"radii and values are two sequences of the same length, "
                         f"at least {least}")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
        raise ValueError("radii and values must be finite")
    if not np.all(np.diff(r) > 0.0):
        raise ValueError("the radii must increase")
    return r, v


def spline(radii: Sequence[float], values: Sequence[float]) -> Profile:
    """The natural cubic spline through (radii, values), continued beyond
    the end knots by the end intervals' cubics (the module docstring)."""
    r, v = _knots(radii, values, 3)
    cs = CubicSpline(r, v, bc_type="natural", extrapolate=True)
    return Profile(cs, description=f"natural cubic spline through {r.size} knots "
                                   f"from {r[0]:g} to {r[-1]:g} m")


def tabulated(radii: Sequence[float], values: Sequence[float]) -> Profile:
    """Linear between the knots, the end values held beyond them."""
    r, v = _knots(radii, values, 2)
    return Profile(lambda x: np.interp(x, r, v),
                   description=f"linear through {r.size} knots from {r[0]:g} to "
                               f"{r[-1]:g} m")
