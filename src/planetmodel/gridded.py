"""Stored fields that vary with direction: values on a grid at GLL nodes.

A `GriddedField` is a field on one interval held as numbers: the
interval is cut into radial elements at `breaks`, each element carries
`ngll` Gauss-Lobatto-Legendre nodes, and at every node the field is
given on one `AngularGrid`, so that `values` has shape

    (nelem, ngll, ntheta, nphi) + stored shape,

the stored components in the spherical frame at each node (Voigt for
the ranks that have it), which is a `Sample`'s layout cut to one layer.
Between the nodes it is read radially by one of `mesh1d.RADIAL_METHODS`
within the element holding the radius (the Lagrange polynomial through
its GLL nodes, or straight lines between consecutive nodes), and
laterally by one of the rules of `planetmodel.lateral` at every node of
that element.  A table of depth slices joined by straight
lines is `ngll = 2` with the breaks at the slices.  The end elements
continue beyond the interval, as a polynomial layer's end pieces do,
which is what `on_interval` asks of them.

Components of rank 1 and above are read in the Cartesian frame, where
a field is single-valued at a pole, and turned back into the frame
asked for at the point.  The values at the poles follow the rule of
`lateral.polar_values` and have one source each: a pole that is a node
of the grid is its own row, and a pole the grid lacks is given in
`poles`, of shape (nelem, ngll, 2) + stored shape with the components
Cartesian, or found from the grid's expansion when the grid has a band;
`poles` holds both poles once resolved.  `GriddedField.sampled` builds
one from any field, evaluating its values at the poles exactly.

A `GriddedFunction` is the same thing for a function of (theta, phi)
alone: values on a grid, its values at the poles, a lateral rule, and
the two angular derivatives, so that it serves as a relief of
`layer_linear` and as a surface quantity such as the thickness of a
lithosphere.

Everything is numbers and every stored array is read-only.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .character import SCALAR, Character
from .displacement import Relief
from .fields import Field, FieldBase, RadialField, rotate_stored, stored_shape
from .frames import spherical_frame
from .layerfunction import PolynomialLayer, as_interval
from .lateral import METHODS, LateralInterpolant, polar_values
from .mesh1d.mesh import RADIAL_METHODS, Mesh1D

if TYPE_CHECKING:
    from .sampling import AngularGrid

__all__ = ["GriddedField", "GriddedFunction"]


def _readonly(a: np.ndarray) -> np.ndarray:
    """A read-only float64 copy."""
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


def _grid_frames(grid: AngularGrid) -> np.ndarray:
    """The spherical frame at every node of the grid, (ntheta, nphi, 3, 3)."""
    return spherical_frame(grid.colatitudes[:, None], grid.longitudes[None, :])


class GriddedField(FieldBase):
    """A field held on an angular grid at the GLL nodes of radial elements.

    `breaks` are the element boundaries, strictly increasing and
    spanning `interval` (within `rtol` of its width at each end), and
    `values` has shape `(nelem, ngll, ntheta, nphi) + stored shape`, the
    components in the spherical frame at the nodes of the elements, with
    `ngll` at least two.  `radial` is one of `mesh1d.RADIAL_METHODS` and
    `lateral` one of `lateral.METHODS`.  `poles`, of shape
    `(nelem, ngll, 2) + stored shape` with components in the Cartesian
    frame, gives the values at the poles the grid lacks, by the rule of
    `lateral.polar_values` with `rtol` of the values' largest; when it is
    None they are found from the grid's expansion, which needs a band
    and the `harmonics` extra, and refused otherwise.  Values are real.
    """

    def __init__(self, interval: tuple[float, float], breaks: ArrayLike,
                 grid: AngularGrid, values: ArrayLike, *,
                 character: Character = SCALAR, name: str | None = None,
                 radial: str = "polynomial", lateral: str = "linear",
                 poles: ArrayLike | None = None, rtol: float = 1e-9) -> None:
        from .sampling import AngularGrid
        if not isinstance(grid, AngularGrid):
            raise TypeError(f"grid must be an AngularGrid, got {type(grid).__name__}")
        if radial not in RADIAL_METHODS:
            raise ValueError(f"radial must be one of {RADIAL_METHODS}, got {radial!r}")
        if lateral not in METHODS:
            raise ValueError(f"lateral must be one of {METHODS}, got {lateral!r}")
        self._interval = as_interval(interval)
        self._character = character
        self._name = name
        self._rtol = float(rtol)
        lo, hi = self._interval
        b = np.array(breaks, dtype=float)
        if b.ndim != 1 or b.size < 2 or not np.all(np.diff(b) > 0.0):
            raise ValueError("breaks must be a strictly increasing 1-d array of at "
                             "least two radii")
        tol = self._rtol * (hi - lo)
        if abs(b[0] - lo) > tol or abs(b[-1] - hi) > tol:
            raise ValueError(f"breaks from {b[0]:g} to {b[-1]:g} do not span the "
                             f"interval [{lo:g}, {hi:g}]")
        if np.iscomplexobj(values) or (poles is not None and np.iscomplexobj(poles)):
            raise TypeError("a gridded field holds real values")
        v = np.asarray(values, dtype=float)
        c = stored_shape(character)
        lead = (b.size - 1,)
        if v.ndim != 4 + len(c) or v.shape[:1] != lead or v.shape[1] < 2 \
                or v.shape[2:] != (grid.ntheta, grid.nphi) + c:
            raise ValueError(
                f"values must have shape (nelem, ngll, ntheta, nphi) + {c} = "
                f"{lead + ('ngll', grid.ntheta, grid.nphi) + c} with ngll >= 2, "
                f"got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("a gridded field holds finite values")
        self._breaks = _readonly(b)
        self._grid = grid
        self._data = _readonly(v)
        self._radial, self._lateral = radial, lateral
        self._mesh = Mesh1D(b, ngll=v.shape[1])
        # the components laid out for the lateral rules: Cartesian, with
        # the grid's two axes last
        cart = v if not character.rank else rotate_stored(v, _grid_frames(grid),
                                                          character)
        self._cartesian = np.moveaxis(cart, (2, 3), (-2, -1))
        if poles is not None:
            want = lead + (v.shape[1], 2) + c
            p = np.asarray(poles, dtype=float)
            if p.shape != want:
                raise ValueError(f"poles must have shape (nelem, ngll, 2) + {c} = "
                                 f"{want}, got {p.shape}")
            poles = np.moveaxis(p, 2, -1)
        resolved = polar_values(self._cartesian, grid, poles=poles, rtol=self._rtol)
        self._poles = _readonly(np.moveaxis(resolved, -1, 2))
        self._interpolants: dict[int, LateralInterpolant] = {}

    @classmethod
    def sampled(cls, field: Field, grid: AngularGrid, *,
                breaks: ArrayLike | None = None, ngll: int = 5,
                radial: str = "polynomial", lateral: str = "linear",
                name: str | None = None) -> "GriddedField":
        """`field` evaluated on `grid` at the GLL nodes of elements with
        `breaks` (the ends of its interval by default) and `ngll` nodes
        each, with its values at both poles evaluated exactly: a gridded
        field that equals `field` at every node.  The name is the field's
        unless `name` is given."""
        lo, hi = (float(x) for x in field.interval)
        b = np.array([lo, hi] if breaks is None else breaks, dtype=float)
        mesh = Mesh1D(b, ngll=int(ngll))
        r = mesh.r[:, :, None, None]
        theta = grid.colatitudes[None, None, :, None]
        phi = grid.longitudes[None, None, None, :]
        values = field.evaluate(r, theta, phi)
        poles = field.evaluate(mesh.r[:, :, None], np.array([0.0, np.pi]),
                               np.zeros(2), frame="cartesian")
        return cls(field.interval, b, grid, values, character=field.character,
                   name=field.name if name is None else name, radial=radial,
                   lateral=lateral, poles=poles, rtol=getattr(field, "rtol", 1e-9))

    @property
    def breaks(self) -> np.ndarray:
        """The radii of the element boundaries."""
        return self._breaks

    @property
    def grid(self) -> AngularGrid:
        """The angular grid at every node."""
        return self._grid

    @property
    def values(self) -> np.ndarray:
        """The stored components at the nodes, (nelem, ngll, ntheta, nphi) +
        stored shape, in the spherical frame."""
        return self._data

    @property
    def poles(self) -> np.ndarray:
        """The values at the north and south poles at every node,
        (nelem, ngll, 2) + stored shape, components in the Cartesian frame."""
        return self._poles

    @property
    def radial(self) -> str:
        """The radial rule, one of `mesh1d.RADIAL_METHODS`."""
        return self._radial

    @property
    def lateral(self) -> str:
        """The lateral rule, one of `lateral.METHODS`."""
        return self._lateral

    @property
    def ngll(self) -> int:
        """The GLL nodes per element."""
        return self._data.shape[1]

    @property
    def radii(self) -> np.ndarray:
        """The radius of every node, (nelem, ngll)."""
        return self._mesh.r

    def _interpolant(self, e: int) -> LateralInterpolant:
        """The lateral rule over the nodes of element e, built once."""
        f = self._interpolants.get(e)
        if f is None:
            f = LateralInterpolant(self._cartesian[e], self._grid,
                                   method=self._lateral,
                                   poles=np.moveaxis(self._poles[e], 1, -1),
                                   rtol=self._rtol)
            self._interpolants[e] = f
        return f

    def _values_at(self, r: np.ndarray, theta: np.ndarray,
                   phi: np.ndarray) -> np.ndarray:
        """The Cartesian components at flat points, (P,) + stored shape."""
        e, W = self._mesh.interpolation_weights(r, method=self._radial)
        out = np.empty((r.size,) + self.stored_shape)
        for k in np.unique(e):
            m = e == k
            lat = self._interpolant(int(k))(theta[m], phi[m])   # (ngll, *c, Pk)
            out[m] = np.einsum("pi,i...p->p...", W[m], lat)
        return out

    def _values(self, r: np.ndarray, theta: np.ndarray | None,
                phi: np.ndarray | None) -> np.ndarray:
        shape = r.shape
        out = self._values_at(r.ravel(), theta.ravel(), phi.ravel())
        out = out.reshape(shape + self.stored_shape)
        if self._character.rank:
            R = spherical_frame(theta, phi)
            out = rotate_stored(out, np.swapaxes(R, -1, -2), self._character)
        return out

    def spherical_mean(self, *, grid: AngularGrid | None = None) -> RadialField:
        """The spherical mean as exact polynomial pieces: at every node the
        exact mean of the lateral rule's interpolant, joined by the radial
        rule, so the mean of the field as its rules define it; `grid` is
        not needed."""
        if self._character.rank:
            raise ValueError(f"a spherical mean is taken of a field of rank 0, "
                             f"not {self._character}")
        nodal = np.stack([self._interpolant(e).mean()
                          for e in range(self._breaks.size - 1)])
        pp = self._mesh.to_ppoly(nodal, method=self._radial)
        return RadialField(self._interval, PolynomialLayer(self._interval, pp),
                           character=self._character, name=self._name)

    def on_interval(self, lo: float, hi: float) -> "GriddedField":
        """The same elements on [lo, hi]: beyond the breaks the end
        elements continue."""
        out = copy.copy(self)
        out._interval = as_interval((lo, hi))
        out._interpolants = self._interpolants
        return out

    def rescaled(self, *, k: float, v: float) -> "GriddedField":
        """v f(r / k, theta, phi): the breaks and interval scaled by k, the
        values and the polar values by v."""
        k, v = float(k), float(v)
        lo, hi = self._interval
        return GriddedField((k * lo, k * hi), k * self._breaks, self._grid,
                            v * self._data, character=self._character,
                            name=self._name, radial=self._radial,
                            lateral=self._lateral, poles=v * self._poles,
                            rtol=self._rtol)

    def renamed(self, name: str | None) -> "GriddedField":
        """The same field under `name`."""
        out = copy.copy(self)
        out._name = name
        return out

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        lo, hi = self._interval
        return (f"GriddedField({nm}{self._character} on [{lo:g}, {hi:g}]: "
                f"{self._breaks.size - 1} x {self.ngll} nodes, {self._radial} x "
                f"{self._lateral} on {self._grid!r})")


class GriddedFunction:
    """A function of (theta, phi) held on an angular grid.

    `values` has shape (ntheta, nphi); `lateral` is one of
    `lateral.METHODS`, the bicubic one by default; `poles`, shape (2,),
    gives the values at the poles the grid lacks, by the rule of
    `lateral.polar_values` with `rtol`, and when it is None they are
    found from the grid's expansion or refused.  Calling it gives the
    values at points; `angular_gradient` gives (d/dtheta, d/dphi), so
    that it is a relief for `layer_linear`.
    """

    def __init__(self, grid: AngularGrid, values: ArrayLike, *,
                 lateral: str = "cubic", poles: ArrayLike | None = None,
                 rtol: float = 1e-9) -> None:
        if np.iscomplexobj(values):
            raise TypeError("a gridded function holds real values")
        v = np.asarray(values, dtype=float)
        if v.ndim != 2:
            raise ValueError(f"values must have shape (ntheta, nphi), got {v.shape}")
        if lateral not in METHODS:
            raise ValueError(f"lateral must be one of {METHODS}, got {lateral!r}")
        self._poles = _readonly(polar_values(v, grid, poles=poles, rtol=rtol))
        self._f = LateralInterpolant(v, grid, method=lateral, poles=self._poles,
                                     rtol=rtol)
        self._data = _readonly(v)

    @classmethod
    def sampled(cls, fn: Relief, grid: AngularGrid, *,
                lateral: str = "cubic") -> "GriddedFunction":
        """`fn(theta, phi)` evaluated on `grid` and, exactly, at both poles."""
        values = np.asarray(fn(grid.colatitudes[:, None], grid.longitudes[None, :]),
                            dtype=float)
        poles = np.asarray(fn(np.array([0.0, np.pi]), np.zeros(2)), dtype=float)
        return cls(grid, np.broadcast_to(values, (grid.ntheta, grid.nphi)),
                   lateral=lateral, poles=poles)

    @property
    def grid(self) -> AngularGrid:
        """The angular grid the values are on."""
        return self._f.grid

    @property
    def values(self) -> np.ndarray:
        """The values at the nodes, (ntheta, nphi)."""
        return self._data

    @property
    def poles(self) -> np.ndarray:
        """The values at the north and south poles."""
        return self._poles

    @property
    def lateral(self) -> str:
        """The lateral rule, one of `lateral.METHODS`."""
        return self._f.method

    def __call__(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """The values at (theta, phi), broadcast together."""
        return self._f(theta, phi)

    def angular_gradient(self, theta: ArrayLike, phi: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(d/dtheta, d/dphi) at (theta, phi), broadcast together."""
        return self._f.gradient(theta, phi)

    def mean(self) -> float:
        """The mean over the sphere of the function as its rule defines it."""
        return float(self._f.mean())

    def __repr__(self) -> str:
        return f"GriddedFunction({self.lateral} on {self.grid!r})"
