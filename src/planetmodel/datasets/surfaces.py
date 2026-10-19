"""Surfaces: functions of the angles that place a surface in a body.

A published surface (a planet's shape, its topography, the radius of
its crust-mantle interface, an ice sheet's thickness) is a function of
(theta, phi) in metres together with what its values measure, which
`Surface` holds: `"radius"`, the surface's radius itself; `"elevation"`,
its height above a datum; `"depth"`, its distance below a datum; or
`"thickness"`, the thickness of what lies between it and a datum above,
which places it as a depth does.  The datum is a radius, or another
`Surface` (an ice sheet's base is its thickness below the ice's top);
a surface measuring a radius has none, and a surface whose datum is not
yet known (a thickness published alone) is given one with
`with_datum`.  `radius` gives where the surface is; calling it gives
its values as published.  The function is a `GriddedFunction`, a
`HarmonicFunction` or any function of the angles; `sampled` holds a
surface on a grid, the internal form, which a harmonic function reaches
through ducc0's transforms, and `mean` gives the spherical mean of its
values.

`CoefficientReader` is the reader of a registry entry whose files are
coefficient tables (`datasets.coefficients`): it reads one of them into
a `Surface` of the measure and datum the entry states.
`GridSurfaceReader` is the reader of one whose file is a global netCDF
grid (`grids.read_grid`): one variable of it becomes the surface, on
the grid's own nodes, bilinear between them, a pole the grid lacks the
mean of its nearest ring.
"""
from __future__ import annotations

import os
from dataclasses import KW_ONLY, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..units import LENGTH, parse_unit
from .coefficients import FORMATS, read_coefficients
from .grids import read_grid, ring_poles

if TYPE_CHECKING:
    from ..displacement import Relief
    from ..sampling import AngularGrid
    from .registry import Dataset

__all__ = ["MEASURES", "Datum", "Surface", "CoefficientReader", "GridSurfaceReader"]

#: What a surface's values measure.
MEASURES = ("radius", "elevation", "depth", "thickness")

#: What elevations, depths and thicknesses are measured from: a radius,
#: or another surface.
type Datum = float | Surface


@dataclass(frozen=True, eq=False)
class Surface:
    """A surface as a function of the angles, by the rules of the module
    docstring: `function` its values in metres, `measures` one of
    `MEASURES`, `datum` a radius or a `Surface` (none for a radius),
    `name` what it is."""

    function: Relief
    measures: str
    _: KW_ONLY
    datum: Datum | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.measures not in MEASURES:
            raise ValueError(f"measures must be one of {MEASURES}, got "
                             f"{self.measures!r}")
        if not callable(self.function):
            raise TypeError("a surface's function is a function of (theta, phi)")
        if self.measures == "radius" and self.datum is not None:
            raise ValueError("a surface measuring its radius has no datum")
        if self.datum is not None and not isinstance(self.datum, Surface):
            object.__setattr__(self, "datum", float(self.datum))

    def __call__(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """The values at (theta, phi), broadcast together, as published."""
        t, p = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                   np.asarray(phi, dtype=float))
        return np.broadcast_to(np.asarray(self.function(t, p), dtype=float), t.shape)

    def radius(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """The surface's radius at (theta, phi), broadcast together: the
        values themselves, the datum's radius plus an elevation, or less a
        depth or a thickness; refused for a surface without its datum."""
        values = self(theta, phi)
        if self.measures == "radius":
            return values
        if self.datum is None:
            raise ValueError(
                f"{self._label()} measures {self.measures} from a datum it does not "
                "have: give it one with with_datum(radius or surface)")
        base = (self.datum.radius(theta, phi) if isinstance(self.datum, Surface)
                else self.datum)
        return base + values if self.measures == "elevation" else base - values

    def with_datum(self, datum: Datum) -> Surface:
        """The same surface measured from `datum`."""
        return replace(self, datum=datum)

    def mean(self, *, grid: AngularGrid | None = None) -> float:
        """The spherical mean of the values: the function's own where it
        has one (a harmonic function's c_00 term, a gridded function's by
        its rule), by quadrature on `grid`, which has weights, otherwise."""
        own = getattr(self.function, "mean", None)
        if own is not None:
            return float(own())
        if grid is None or grid.weights is None:
            raise ValueError(f"{self._label()}'s function has no mean of its own: "
                             "pass grid=, an angular grid with weights")
        return float(grid.mean(self(grid.colatitudes[:, None],
                                    grid.longitudes[None, :])))

    def sampled(self, grid: AngularGrid, *, lateral: str = "cubic") -> Surface:
        """The surface held on `grid` as a `GriddedFunction` read by
        `lateral`, its values there (through the function's `on_grid`
        where it has one) and at both poles; the measure, datum and name
        kept."""
        from ..gridded import GriddedFunction
        on_grid = getattr(self.function, "on_grid", None)
        values = (np.asarray(on_grid(grid), dtype=float) if on_grid is not None
                  else self(grid.colatitudes[:, None], grid.longitudes[None, :]))
        poles = self(np.array([0.0, np.pi]), np.zeros(2))
        return replace(self, function=GriddedFunction(
            grid, np.broadcast_to(values, (grid.ntheta, grid.nphi)), lateral=lateral,
            poles=poles))

    def _label(self) -> str:
        return f"the surface {self.name!r}" if self.name else "the surface"

    def __repr__(self) -> str:
        datum = ("" if self.datum is None else
                 f" from {self.datum._label()}" if isinstance(self.datum, Surface)
                 else f" from radius {self.datum:g}")
        return (f"Surface({self.name or 'unnamed'}: {self.measures}{datum}, "
                f"{self.function!r})")


class CoefficientReader:
    """The reader of a coefficient table into a `Surface`: `format` and
    the source's `normalization` and `csphase` as `read_coefficients`
    takes them, the `measures`, `datum` and `name` (the entry's by
    default) of what is read.  `read(paths, dataset, *, lmax=None,
    which=0)` reads the table `which` (an index into the entry's files,
    or a file's name) to `lmax`."""

    def __init__(self, *, format: str, measures: str, normalization: str = "4pi",
                 csphase: int = 1, datum: Datum | None = None,
                 name: str | None = None) -> None:
        if format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
        if measures not in MEASURES:
            raise ValueError(f"measures must be one of {MEASURES}, got {measures!r}")
        self.format, self.measures = format, measures
        self.normalization, self.csphase = normalization, csphase
        self.datum, self.name = datum, name

    def read(self, paths: tuple[Path, ...], dataset: Dataset, *,
             lmax: int | None = None, which: int | str = 0) -> Surface:
        """The surface table `which` holds, to `lmax`."""
        names = [os.path.basename(os.fspath(p)) for p in paths]
        if isinstance(which, str):
            if which not in names:
                raise KeyError(f"no file {which!r}; the files are {names}")
            which = names.index(which)
        if not 0 <= int(which) < len(paths):
            raise IndexError(f"which must name one of the {len(paths)} files, got "
                             f"{which}")
        function = read_coefficients(paths[int(which)], format=self.format,
                                     normalization=self.normalization,
                                     csphase=self.csphase, lmax=lmax)
        name = self.name or dataset.name
        if len(paths) > 1:
            name = f"{name}, {names[int(which)]}"
        return Surface(function, self.measures, datum=self.datum, name=name)

    def __repr__(self) -> str:
        return (f"CoefficientReader({self.format}, {self.measures}, "
                f"{self.normalization}, csphase={self.csphase})")


class GridSurfaceReader:
    """The reader of a surface held as the variable `variable` of a global
    netCDF grid, by the rules of the module docstring: its unit the
    file's, or `units` where the file's is wrong or absent, a length;
    measuring `measures` from `datum`; `regular` rebuilding glitched
    coordinates as `read_grid` does; named `name`, the entry's by
    default.  netCDF4 is the `netcdf` extra."""

    def __init__(self, variable: str, *, measures: str, datum: Datum | None = None,
                 units: str | None = None, regular: bool = False,
                 name: str | None = None) -> None:
        if measures not in MEASURES:
            raise ValueError(f"measures must be one of {MEASURES}, got {measures!r}")
        self.variable, self.measures, self.datum = variable, measures, datum
        self.units, self.regular, self.name = units, regular, name

    def read(self, paths: tuple[Path, ...], dataset: Dataset) -> Surface:
        """The surface the grid's variable is."""
        from ..gridded import GriddedFunction
        data = read_grid(paths[0], variables=(self.variable,), regular=self.regular)
        if data.grid is None or data.radius is not None:
            raise ValueError(f"{paths[0]}: {self.variable} is not on a global grid "
                             "of the angles alone")
        unit = self.units or data.units.get(self.variable, "")
        dims, factor = parse_unit(unit)
        if dims != LENGTH:
            raise ValueError(f"{paths[0]}: {self.variable} is in {unit!r}, not a "
                             "length")
        v = factor * np.asarray(data.values[self.variable], dtype=float)
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{paths[0]}: {self.variable} is missing at "
                             f"{int(np.sum(~np.isfinite(v)))} nodes")
        function = GriddedFunction(data.grid, v, lateral="linear",
                                   poles=ring_poles(v, data.grid))
        return Surface(function, self.measures, datum=self.datum,
                       name=self.name or dataset.name)

    def __repr__(self) -> str:
        return f"GridSurfaceReader({self.variable}, {self.measures})"
