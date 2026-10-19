"""Gridded sources: netCDF grids of latitude and longitude, in the library's form.

The Earth Model Collaboration's files, and COARDS grids such as GMT's,
hold fields on a grid of latitude and longitude, with or without a
vertical axis, in conventions that vary from file to file.  `read_grid`
reads one into a `GridData` in the library's form:

- the axes are found by their coordinate variables' names (latitude,
  lat; longitude, lon; depth; radius; elevation, height), or failing
  those by `standard_name`, units and `positive`, in any order of
  dimensions; a projected grid of x and y is refused here;
- a vertical axis of depth (positive down), radius or elevation becomes
  radius in metres on the source's surface radius (its global
  `radius_in_meters`, else `surface=`, the Earth's 6371 km by default),
  increasing; two slices within `VERTICAL_RTOL` of the surface radius of
  each other, and the pairs `pairs=` names, are made one radius: a
  discontinuity, the two slices its sides, as a repeated radius is in a
  deck;
- latitudes become colatitudes increasing from the north, longitudes
  are reduced into one turn and sorted, a seam column repeated at the
  end made one with the first; equally spaced nodes are
  snapped to their exact positions, and `regular=True` rebuilds a
  glitched coordinate as the regular one of its count and ends, refused
  where a node is off by more than half a step;
- missing values (`_FillValue`, `missing_value`, NaN) become NaN, and
  `valid_range` is ignored, being in some files the rounded range of
  the data;
- a seam column repeated at the end, and a pole row, are checked to hold
  one value to `SAME_RTOL` of the field's largest (the precision of a file
  written to five figures) and made exactly that; one that does not is
  refused, or averaged where the entry says so (`seam="mean"`,
  `pole_rows="mean"`: a source whose own gridding left two columns on
  one meridian);
- `stride=k` keeps every k-th latitude and longitude of the file's own
  order, for a grid too fine to hold whole, and `vertical=(lo, hi)` the
  slices whose vertical coordinate lies in that window (in the file's
  own units, either end open as None), for a source whose part beyond it
  is not wanted; both are applied as the file is read, and the file's
  own precision kept until the grid is in its order, so that a large
  grid is copied only once it is cut down.

A grid whose longitudes go round the circle and whose latitudes reach
within a step and a half of each pole is global, on an `AngularGrid`:
the shipped equiangular kind where its nodes are one (from wherever its
longitudes start), a custom grid otherwise.  Any other is regional,
kept on its own coordinates.  Values stay in the file's units, beside
its unit strings.

`grid_model` makes the model of a global grid with a vertical axis:
each variable under the vocabulary name it is mapped to, in SI by its
unit string (`units.parse_unit`; a mislabel corrected by `units=`), as
values (kind None) or as a perturbation of one of `perturbation.KINDS`
over a reference; depth slices missing everywhere dropped from the ends
of the support and refused inside it, scattered missing values refused
unless `fill=` gives them; `GriddedField`s with `ngll = 2` between the
slices, radially linear, laterally `lateral` (linear by default), one
per segment between discontinuities; at a pole the grid lacks, the mean
of its nearest ring, the reader's rule, right to second order in the
ring's distance from the pole.  `RegionalGrid` holds a regional grid's
values in SI and evaluates them inside its box.  `GridReader` is the
reader of a data set of one such file: a model of a global grid, a
`RegionalGrid` of a regional one.  netCDF4 is imported where a file is
read (the `netcdf` extra).
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator

from ..character import SCALAR
from ..gridded import GriddedField
from ..model import Model
from ..perturbation import KINDS
from ..sampling import AngularGrid, equiangular
from ..units import DIMENSIONLESS, LENGTH, Scales, parse_unit
from ..vocabulary import spec
from .models import over_reference, reference_of, standalone

if TYPE_CHECKING:
    from .registry import Dataset

__all__ = ["VERTICAL_RTOL", "SAME_RTOL", "EARTH_RADIUS", "Variables", "GridData",
           "read_grid", "grid_model", "RegionalGrid", "GridReader"]

#: Two slices this close, relative to the surface radius, are one radius.
VERTICAL_RTOL = 1e-5

#: A pole row, or a seam column and its repeat, hold one value to this
#: fraction of the field's largest: the precision of five figures.
SAME_RTOL = 1e-4

#: The radius of a depth of zero unless a file or its entry says otherwise.
EARTH_RADIUS = 6371.0e3

#: How the variables of a file are read: each file name to the vocabulary
#: name it is and its kind, None for values, else a perturbation's kind.
type Variables = Mapping[str, tuple[str, str | None]]

_ROLES = {"latitude": ("latitude", "lat"), "longitude": ("longitude", "lon"),
          "depth": ("depth",), "radius": ("radius",),
          "elevation": ("elevation", "height", "altitude")}
_VERTICAL = ("depth", "radius", "elevation")


@dataclass(frozen=True, eq=False)
class GridData:
    """A gridded source in the library's form: `theta` colatitudes
    increasing and `phi` longitudes (radians; in one turn and increasing
    for a global grid, contiguous for a regional one), `radius` (metres,
    non-decreasing, a repeat a discontinuity; None for a surface),
    `values` by the file's names, shaped (radius, theta, phi) or (theta,
    phi), in the file's `units`, NaN where missing; a global grid's
    `grid`."""

    theta: np.ndarray
    phi: np.ndarray
    radius: np.ndarray | None
    values: Mapping[str, np.ndarray]
    units: Mapping[str, str]
    _: KW_ONLY
    grid: AngularGrid | None = None
    surface: float = EARTH_RADIUS
    long_names: Mapping[str, str] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    @property
    def is_global(self) -> bool:
        """Whether the grid covers the sphere, and so has an `AngularGrid`."""
        return self.grid is not None


def _roles(ds: Any) -> dict[str, str]:
    """The coordinate variable of each axis a file has, by its role."""
    found: dict[str, str] = {}
    for name, var in ds.variables.items():
        if var.ndim != 1 or var.dimensions != (name,):
            continue
        low = name.lower()
        std = str(getattr(var, "standard_name", "")).lower()
        unit = str(getattr(var, "units", "")).strip().lower()
        role = next((r for r, names in _ROLES.items() if low in names or std in names),
                    None)
        if role is None and unit.startswith(("degrees_north", "degree_north")):
            role = "latitude"
        elif role is None and unit.startswith(("degrees_east", "degree_east")):
            role = "longitude"
        if role is not None and role not in found:
            found[role] = name
    if "latitude" not in found or "longitude" not in found:
        projected = [n for n in ("x", "y", "X", "Y") if n in ds.variables]
        raise ValueError(
            "the file has no latitude and longitude axes"
            + (f" (it is projected, on {projected})" if projected else ""))
    vertical = [r for r in _VERTICAL if r in found]
    if len(vertical) > 1:
        raise ValueError(f"the file has more than one vertical axis: {vertical}")
    return found


def _regular(values: np.ndarray, what: str) -> np.ndarray:
    """A coordinate rebuilt as the regular one of its count and ends,
    refused where a node is off by more than half a step."""
    rebuilt = np.linspace(values[0], values[-1], values.size)
    step = abs(rebuilt[1] - rebuilt[0]) if values.size > 1 else 1.0
    if np.max(np.abs(values - rebuilt)) > 0.5 * step:
        raise ValueError(f"the {what} are not a regular coordinate with glitches: a "
                         "node is off by more than half a step")
    return rebuilt


def _unwrapped(reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted longitudes of one turn, in degrees, made contiguous and
    increasing from after the widest gap around the circle; with the
    permutation that takes them there."""
    gaps = np.diff(np.concatenate([reduced, [reduced[0] + 360.0]]))
    start = (int(np.argmax(gaps)) + 1) % reduced.size
    order = np.concatenate([np.arange(start, reduced.size), np.arange(start)])
    return np.concatenate([reduced[start:], reduced[:start] + 360.0]), order


def read_grid(source: str | os.PathLike[str], *,
              variables: tuple[str, ...] | None = None,
              surface: float | None = None, vertical_units: str | None = None,
              pairs: tuple[tuple[float, float], ...] = (), regular: bool = False,
              seam: str = "refuse", pole_rows: str = "refuse", stride: int = 1,
              vertical: tuple[float | None, float | None] | None = None) -> GridData:
    """A netCDF grid in the library's form, by the rules of the module
    docstring: `variables` the names to read (every field on the grid's
    axes by default), `surface` the radius of a depth of zero in metres
    where the file states none, `vertical_units` the vertical axis's unit
    where it states none, `pairs` the values of the vertical coordinate
    of slices that are the two sides of one discontinuity."""
    import netCDF4

    for what, rule in (("seam", seam), ("pole_rows", pole_rows)):
        if rule not in ("refuse", "mean"):
            raise ValueError(f"{what} must be 'refuse' or 'mean', got {rule!r}")
    path = os.fspath(source)
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        roles = _roles(ds)
        axes = [r for r in _VERTICAL if r in roles] + ["latitude", "longitude"]
        dims = [roles[r] for r in axes]
        names = (tuple(n for n, v in ds.variables.items()
                       if n not in dims and set(v.dimensions) == set(dims))
                 if variables is None else tuple(variables))
        if not names:
            raise ValueError(f"{path}: no field lies on the grid's axes {dims}")
        coords = {r: np.asarray(ds.variables[roles[r]][:], dtype=float) for r in axes}
        vunit = {r: str(getattr(ds.variables[roles[r]], "units", "")) for r in axes}
        attributes = {k: ds.getncattr(k) for k in ds.ncattrs()}
        axis = next((r for r in _VERTICAL if r in coords), None)
        stride = int(stride)
        if stride < 1:
            raise ValueError(f"stride must be a positive integer, got {stride}")
        take = {roles["latitude"]: slice(None, None, stride),
                roles["longitude"]: slice(None, None, stride)}
        if vertical is not None:
            if axis is None:
                raise ValueError(f"{path}: vertical= cuts a grid with a vertical axis")
            lo, hi = vertical
            c = coords[axis]
            keep = np.ones(c.shape, dtype=bool)
            if lo is not None:
                keep &= c >= lo
            if hi is not None:
                keep &= c <= hi
            if keep.sum() < 2:
                raise ValueError(f"{path}: the window {vertical} keeps fewer than "
                                 "two slices")
            first, last = int(np.argmax(keep)), int(keep.size - np.argmax(keep[::-1]))
            take[roles[axis]] = slice(first, last)
        for r in axes:
            coords[r] = coords[r][take.get(roles[r], slice(None))]
        values, units, long_names = {}, {}, {}
        for n in names:
            var = ds.variables[n]
            if set(var.dimensions) != set(dims):
                raise ValueError(f"{path}: {n!r} lies on {var.dimensions}, not the "
                                 f"grid's axes {dims}")
            a = np.asarray(var[tuple(take.get(d, slice(None)) for d in var.dimensions)])
            a = a.astype(np.float32 if a.dtype == np.float32 else float, copy=False)
            for attr in ("_FillValue", "missing_value"):
                if attr in var.ncattrs():
                    fill = np.asarray(var.getncattr(attr), dtype=var.dtype).ravel()[0]
                    a[a == fill] = np.nan
            values[n] = np.transpose(a, [var.dimensions.index(d) for d in dims])
            units[n] = str(getattr(var, "units", ""))
            long_names[n] = str(getattr(var, "long_name", ""))
    lat, lon = coords["latitude"], coords["longitude"]
    if regular:
        lat, lon = _regular(lat, "latitudes"), _regular(lon, "longitudes")
    # colatitude increasing from the north
    if np.any(np.abs(lat) > 90.0 + 1e-6):
        raise ValueError(f"{path}: latitudes beyond +-90 degrees")
    lat = np.clip(lat, -90.0, 90.0)
    order = np.argsort(-lat, kind="stable")
    lat = lat[order]
    values = {n: np.take(v, order, axis=-2) for n, v in values.items()}
    # longitudes: the seam dropped, one turn, sorted
    reduced = np.mod(lon, 360.0)
    reduced[np.isclose(reduced, 360.0, rtol=0.0, atol=1e-9)] = 0.0
    keep = np.ones(lon.size, dtype=bool)
    for i in range(lon.size):
        same = np.flatnonzero(np.isclose(reduced[:i], reduced[i], rtol=0.0, atol=1e-6))
        if same.size:
            j = int(same[0])
            for n, v in values.items():
                scale = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 1.0
                worst = float(np.nanmax(np.abs(v[..., i] - v[..., j]), initial=0.0))
                if seam == "refuse" and worst > SAME_RTOL * scale:
                    raise ValueError(
                        f"{path}: {n!r} differs by {worst:.3g} between the two columns "
                        f"at longitude {reduced[i]:g}, the seam repeated; seam='mean' "
                        "averages them")
                v[..., j] = 0.5 * (v[..., i] + v[..., j])
            keep[i] = False
    lon, reduced = lon[keep], reduced[keep]
    values = {n: v[..., keep] for n, v in values.items()}
    order = np.argsort(reduced, kind="stable")
    reduced = reduced[order]
    values = {n: v[..., order] for n, v in values.items()}
    values = {n: np.array(v, dtype=float) for n, v in values.items()}
    theta = np.radians(90.0 - lat)
    # global or regional
    lon_gaps = np.diff(np.concatenate([reduced, [reduced[0] + 360.0]]))
    lat_step = np.median(np.abs(np.diff(lat))) if lat.size > 1 else 180.0
    reaches = 90.0 - lat[0] <= 1.5 * lat_step and lat[-1] + 90.0 <= 1.5 * lat_step
    is_global = bool(lon_gaps.max() <= 1.5 * np.median(lon_gaps) and reaches)
    grid = None
    if is_global:
        n, m = theta.size, reduced.size
        phi0 = np.radians(reduced[0])
        even_lon = np.allclose(lon_gaps, 360.0 / m, rtol=0.0, atol=1e-6)
        for poles in (False, True):
            if not even_lon or (poles and n < 2):
                continue
            candidate = equiangular(n, m, poles=poles, phi0=phi0)
            if (np.allclose(candidate.colatitudes, theta, rtol=0.0, atol=1e-8)
                    and np.allclose(candidate.longitudes, np.radians(reduced),
                                    rtol=0.0, atol=1e-8)):
                grid = candidate
                break
        if grid is None:
            phi = (np.sort(np.radians(reduced) % (2 * np.pi)) if not even_lon else
                   np.radians(reduced[0]) + 2 * np.pi * np.arange(m) / m)
            grid = AngularGrid(theta, phi)
        theta, phi = grid.colatitudes, grid.longitudes
        # the pole rows, one value each
        for k, at in ((0, 0.0), (-1, np.pi)):
            if abs(theta[k] - at) > 1e-12:
                continue
            for name, v in values.items():
                row = v[..., k, :]
                with warnings.catch_warnings():         # a slice missing everywhere
                    warnings.simplefilter("ignore", RuntimeWarning)
                    spread = np.nanmax(row, axis=-1) - np.nanmin(row, axis=-1)
                    mean = np.nanmean(row, axis=-1)
                scale = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 1.0
                worst = float(np.nanmax(spread, initial=0.0))
                if pole_rows == "refuse" and worst > SAME_RTOL * scale:
                    raise ValueError(
                        f"{path}: {name!r} varies by {worst:.3g} along the "
                        f"pole row at colatitude {at:g}; pole_rows='mean' averages it")
                v[..., k, :] = mean[..., None]
    else:
        contiguous, order = _unwrapped(reduced)
        phi = np.radians(contiguous)
        values = {n: v[..., order] for n, v in values.items()}
    radius = None
    vertical = axis
    if vertical is not None:
        unit = vertical_units or vunit[vertical]
        if not unit.strip():
            raise ValueError(f"{path}: the {vertical} axis states no unit; give "
                             "vertical_units=")
        dims_, factor = parse_unit(unit)
        if dims_ != LENGTH:
            raise ValueError(f"{path}: the {vertical} axis's unit {unit!r} is not a "
                             "length")
        top = float(attributes.get("radius_in_meters", surface or EARTH_RADIUS))
        c = coords[vertical]
        r = {"depth": top - c * factor, "radius": c * factor,
             "elevation": top + c * factor}[vertical]
        order = np.argsort(r, kind="stable")
        r, c = r[order], c[order]
        values = {n: v[order] for n, v in values.items()}
        for a, b in pairs:
            i = np.flatnonzero(np.isclose(c, a, rtol=0.0, atol=1e-9 * max(1.0, abs(a))))
            j = np.flatnonzero(np.isclose(c, b, rtol=0.0, atol=1e-9 * max(1.0, abs(b))))
            if i.size != 1 or j.size != 1 or abs(int(i[0]) - int(j[0])) != 1:
                raise ValueError(f"{path}: the pair ({a:g}, {b:g}) is not two adjacent "
                                 f"slices of the {vertical} axis")
            lo, hi = sorted((int(i[0]), int(j[0])))
            r[lo] = r[hi] = 0.5 * (r[lo] + r[hi])
        close = np.flatnonzero(np.diff(r) < VERTICAL_RTOL * top)
        if np.any(np.diff(close) == 1):
            raise ValueError(f"{path}: three slices meet at one radius")
        for i in close:
            r[i] = r[i + 1] = 0.5 * (r[i] + r[i + 1])
        radius = r
        surface = top
    for v in values.values():
        v.setflags(write=False)
    return GridData(theta, phi, radius, MappingProxyType(values),
                    MappingProxyType(units), grid=grid,
                    surface=float(surface or EARTH_RADIUS),
                    long_names=MappingProxyType(long_names),
                    attributes=MappingProxyType(attributes), source=path)


def _si(data: GridData, file_name: str, name: str, kind: str | None,
        units: Mapping[str, str] | None) -> np.ndarray:
    """A variable's values in SI, its unit checked against what it is."""
    unit = (units or {}).get(file_name, data.units[file_name])
    dims, factor = parse_unit(unit)
    if kind in ("relative", "log"):
        want = DIMENSIONLESS
    else:
        want = spec(name).dimensions
    if dims != want:
        raise ValueError(f"{file_name!r} is in {unit!r}, of dimensions {dims}; as "
                         f"{name!r} ({'values' if kind is None else kind}) it must be "
                         f"of {want}; units= corrects a mislabel")
    return data.values[file_name] * factor


def _support(v: np.ndarray, name: str) -> slice:
    """The slices of a variable not missing everywhere, refused where a
    missing one lies inside them."""
    present = np.isfinite(v).any(axis=(-2, -1))
    if not present.any():
        raise ValueError(f"{name!r} is missing everywhere")
    first, last = int(np.argmax(present)), int(present.size - np.argmax(present[::-1]))
    if not present[first:last].all():
        raise ValueError(f"{name!r} is missing on a whole slice inside its support")
    return slice(first, last)


def _segments(radius: np.ndarray) -> list[slice]:
    """The runs of slices between discontinuities (repeated radii), each of
    at least two slices; a lone slice at an end, the far side of a boundary
    the source stops at, is left out."""
    cuts = np.flatnonzero(np.diff(radius) == 0.0)
    starts = np.concatenate([[0], cuts + 1])
    stops = np.concatenate([cuts + 1, [radius.size]])
    out = [slice(int(a), int(b)) for a, b in zip(starts, stops)]
    while len(out) > 1 and out[0].stop - out[0].start < 2:
        out.pop(0)
    while len(out) > 1 and out[-1].stop - out[-1].start < 2:
        out.pop()
    for s in out:
        if s.stop - s.start < 2:
            raise ValueError(f"the segment at radius {radius[s.start]:g} m has one "
                             "slice; a segment between discontinuities needs two")
    return out


def _ring_poles(v: np.ndarray, grid: AngularGrid) -> np.ndarray:
    """(slice, 2) values at the poles: a pole node's own, else the mean of
    the nearest ring."""
    with warnings.catch_warnings():                     # a slice missing everywhere
        warnings.simplefilter("ignore", RuntimeWarning)
        north = np.nanmean(v[..., 0, :], axis=-1)
        south = np.nanmean(v[..., -1, :], axis=-1)
    return np.stack([north, south], axis=-1)


def grid_model(data: GridData, variables: Variables, *, reference: Model | None = None,
               units: Mapping[str, str] | None = None, lateral: str = "linear",
               fill: float | None = None, name: str | None = None) -> Model:
    """The model of a global grid with a vertical axis, by the rules of
    the module docstring: `variables` maps the file's names to (name,
    kind); values make a model of their own, perturbations need
    `reference`."""
    if not data.is_global or data.radius is None:
        raise ValueError("a model is made of a global grid with a vertical axis; "
                         "this grid is " + ("regional" if not data.is_global
                                            else "a surface"))
    kinds = {k for _, k in variables.values()}
    if len(kinds) != 1:
        raise ValueError(f"the variables of one model share a kind, got {kinds}")
    (kind,) = kinds
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be None or one of {KINDS}, got {kind!r}")
    grid = data.grid
    converted = {}
    support = None
    for file_name, (vocab, _) in variables.items():
        v = _si(data, file_name, vocab, kind, units)
        here = _support(v, file_name)
        if support is not None and here != support:
            raise ValueError("the variables are missing on different slices")
        support = here
        converted[vocab] = v
    radius = data.radius[support]
    segments = _segments(radius)
    fields: dict[str, list[GriddedField]] = {}
    for vocab, v in converted.items():
        v = np.array(v[support])
        missing = ~np.isfinite(v)
        if missing.any():
            if fill is None:
                raise ValueError(f"{vocab!r} is missing at {int(missing.sum())} nodes "
                                 "inside its support; fill= says what they are")
            v[missing] = fill
        character = (spec(vocab).character if kind in (None, "absolute")
                     else SCALAR)
        poles = _ring_poles(v, grid)
        for k, node in enumerate(grid.pole_nodes):
            if node:
                poles[..., k] = np.nan
        segs = []
        for s in segments:
            r = radius[s]
            vals = np.stack([v[s][:-1], v[s][1:]], axis=1)
            pls = np.stack([poles[s][:-1], poles[s][1:]], axis=1)
            segs.append(GriddedField((float(r[0]), float(r[-1])), r, grid, vals,
                                     character=character, name=vocab,
                                     radial="linear", lateral=lateral, poles=pls))
        fields[vocab] = segs
    if kind is None:
        return standalone(fields, surface=data.surface, name=name)
    if reference is None:
        raise ValueError(f"perturbations of kind {kind!r} need reference=")
    return over_reference(reference, fields, kind=kind, name=name)


class RegionalGrid:
    """A regional grid's variables in SI by their vocabulary names, each
    with its kind, evaluated inside its box: linear in radius between the
    slices of each segment (a radius on a discontinuity reads the side
    above) and in latitude and longitude, NaN outside the box or the
    support.  `reference` names what a perturbation is relative to."""

    def __init__(self, data: GridData, variables: Variables, *,
                 units: Mapping[str, str] | None = None,
                 reference: str | None = None) -> None:
        if data.is_global or data.radius is None:
            raise ValueError("a RegionalGrid holds a regional grid with a vertical "
                             "axis")
        self._data = data
        self._kinds = {vocab: kind for vocab, kind in variables.values()}
        self._values = {vocab: _si(data, f, vocab, kind, units)
                        for f, (vocab, kind) in variables.items()}
        self._segments = _segments(data.radius)
        self.reference = reference
        self._interp: dict[tuple[str, int], RegularGridInterpolator] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """The vocabulary names held."""
        return tuple(self._values)

    @property
    def kinds(self) -> Mapping[str, str | None]:
        """Each name's kind: None for values, else a perturbation's."""
        return MappingProxyType(self._kinds)

    @property
    def extent(self) -> tuple[tuple[float, float], tuple[float, float],
                              tuple[float, float]]:
        """The box: radii (m), colatitudes and longitudes (radians), each
        (low, high), the longitudes contiguous."""
        d = self._data
        return ((float(d.radius[0]), float(d.radius[-1])),
                (float(d.theta[0]), float(d.theta[-1])),
                (float(d.phi[0]), float(d.phi[-1])))

    def coordinates(self, theta: ArrayLike, phi: ArrayLike
                    ) -> tuple[np.ndarray, np.ndarray]:
        """The grid's own coordinates of points: colatitude, and longitude
        reduced into the turn the box's longitudes start."""
        start = float(self._data.phi[0])
        phi = start + np.mod(np.asarray(phi, dtype=float) - start, 2 * np.pi)
        return np.asarray(theta, dtype=float), phi

    def contains(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """Whether points lie inside the box."""
        (_, _), (t0, t1), (p0, p1) = self.extent
        t, p = self.coordinates(theta, phi)
        return (t >= t0) & (t <= t1) & (p >= p0) & (p <= p1)

    def __call__(self, name: str, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
                 ) -> np.ndarray:
        """The variable `name` at points, NaN outside the box or support."""
        if name not in self._values:
            raise KeyError(f"the grid holds {self.names}, not {name!r}")
        t, p = self.coordinates(theta, phi)
        r, t, p = np.broadcast_arrays(np.asarray(r, dtype=float), t, p)
        out = np.full(r.shape, np.nan)
        radius = self._data.radius
        for k, s in enumerate(self._segments):
            lo, hi = radius[s][0], radius[s][-1]
            top = k == len(self._segments) - 1
            inside = (r >= lo) & ((r <= hi) if top else (r < hi))
            if not inside.any():
                continue
            key = (name, k)
            if key not in self._interp:
                self._interp[key] = RegularGridInterpolator(
                    (radius[s], self._data.theta, self._data.phi),
                    self._values[name][s], bounds_error=False, fill_value=np.nan)
            out[inside] = self._interp[key](np.stack([r[inside], t[inside], p[inside]],
                                                     axis=-1))
        return out


@dataclass(frozen=True)
class GridReader:
    """The reader of a data set of one gridded file: `variables` maps its
    names to (vocabulary name, kind); the other fields go to `read_grid`
    and `grid_model` as the entry knows them (a mislabelled unit, the
    surface radius, discontinuity pairs, a glitched coordinate, a seam or
    pole rows to average, a fill, the lateral rule, a stride, a window of
    the vertical axis)."""

    variables: Variables
    _: KW_ONLY
    units: Mapping[str, str] | None = None
    surface: float | None = None
    vertical_units: str | None = None
    pairs: tuple[tuple[float, float], ...] = ()
    regular: bool = False
    seam: str = "refuse"
    pole_rows: str = "refuse"
    fill: float | None = None
    lateral: str = "linear"
    stride: int = 1
    vertical: tuple[float | None, float | None] | None = None

    def read(self, paths: tuple[Path, ...], dataset: Dataset, *,
             reference: Model | None = None, scales: Scales | None = None,
             stride: int | None = None) -> Model | RegionalGrid:
        """A global grid's model (over its reference when it holds
        perturbations), in SI or `scales`; a regional grid's
        `RegionalGrid`."""
        if len(paths) != 1:
            raise ValueError(f"{dataset.name}: a grid is one file, given {len(paths)}")
        data = read_grid(paths[0], variables=tuple(self.variables),
                         surface=self.surface, vertical_units=self.vertical_units,
                         pairs=self.pairs, regular=self.regular, seam=self.seam,
                         pole_rows=self.pole_rows,
                         stride=self.stride if stride is None else stride,
                         vertical=self.vertical)
        if not data.is_global:
            return RegionalGrid(data, self.variables, units=self.units,
                                reference=dataset.reference)
        kinds = {k for _, k in self.variables.values()}
        ref = None if kinds == {None} else reference_of(dataset, reference=reference)
        model = grid_model(data, self.variables, reference=ref, units=self.units,
                           lateral=self.lateral, fill=self.fill, name=dataset.name)
        return model if scales is None else model.converted(scales)
