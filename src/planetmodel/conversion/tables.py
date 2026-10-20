"""Tables of properties against temperature and pressure.

A `PropertyTable` holds properties in SI on a regular grid of
temperature T (K, increasing) and pressure P (Pa, increasing), each an
array (T, P): a table of Vs, density or anything else that an offline
generator (BurnMan, Perple_X, VBRc), a mineral-physics code or a paper's
own law wrote. It is read bilinearly (`__call__`), a point outside the
grid refused by name, and a property monotone in temperature is inverted
for the temperature that gives a value at a pressure
(`temperature_of`): at a fixed pressure the bilinear surface is
piecewise linear in temperature, so the inversion is exact, the value
read between the two tabulated temperatures that bracket it, refused
where the property is not strictly monotone there or no temperature of
the table gives it.

On disk a table is a netCDF file of this form (`write`, `read`):
dimensions `temperature` and `pressure`; the coordinate variables
`temperature(temperature)` and `pressure(pressure)`; one variable
`name(temperature, pressure)` per property, its `units` attribute read
by `units.parse_unit` and converted to SI; the global attributes `name`
and, a work a line, `citation`.  netCDF4 is the `netcdf` extra.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator

from ..model import cited
from ..units import parse_unit

__all__ = ["PropertyTable"]


@dataclass(frozen=True, eq=False)
class PropertyTable:
    """Properties on a grid of `temperature` (K) and `pressure` (Pa), each
    of `properties` an array (temperature, pressure) in SI, by the rule
    of the module docstring; `units` the SI unit of each property, as a
    file states it, `name` what the table is, `citation` its works."""

    temperature: ArrayLike
    pressure: ArrayLike
    properties: Mapping[str, ArrayLike]
    _: KW_ONLY
    name: str
    units: Mapping[str, str] | None = None
    citation: Iterable[str] = ()

    def __post_init__(self) -> None:
        t = np.array(self.temperature, dtype=float)
        p = np.array(self.pressure, dtype=float)
        for axis, what in ((t, "temperature"), (p, "pressure")):
            if axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0.0):
                raise ValueError(f"the {what} axis increases, two values or more")
        props = {}
        for key, values in self.properties.items():
            v = np.array(values, dtype=float)
            if v.shape != (t.size, p.size):
                raise ValueError(f"{key!r} has shape {v.shape}, not (temperature, "
                                 f"pressure) = {(t.size, p.size)}")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"{key!r} is not finite throughout")
            v.flags.writeable = False
            props[key] = v
        if not props:
            raise ValueError("a table holds at least one property")
        t.flags.writeable = False
        p.flags.writeable = False
        object.__setattr__(self, "temperature", t)
        object.__setattr__(self, "pressure", p)
        object.__setattr__(self, "properties", MappingProxyType(props))
        object.__setattr__(self, "units", MappingProxyType(dict(self.units or {})))
        object.__setattr__(self, "citation", cited((), self.citation))

    def _values(self, name: str) -> np.ndarray:
        if name not in self.properties:
            raise KeyError(f"{self.name} holds no {name!r}; it holds "
                           f"{sorted(self.properties)}")
        return self.properties[name]

    def _within(self, p: np.ndarray) -> None:
        lo, hi = self.pressure[0], self.pressure[-1]
        if np.any((p < lo) | (p > hi)) or np.any(np.isnan(p)):
            raise ValueError(f"a pressure lies outside {self.name}'s {lo:g} to "
                             f"{hi:g} Pa")

    def __call__(self, name: str, temperature: ArrayLike, pressure: ArrayLike
                 ) -> np.ndarray:
        """The property `name` at temperatures and pressures, broadcast
        together, read bilinearly."""
        t, p = np.broadcast_arrays(np.asarray(temperature, dtype=float),
                                   np.asarray(pressure, dtype=float))
        lo, hi = self.temperature[0], self.temperature[-1]
        if np.any((t < lo) | (t > hi)) or np.any(np.isnan(t)):
            raise ValueError(f"a temperature lies outside {self.name}'s {lo:g} to "
                             f"{hi:g} K")
        self._within(p)
        f = RegularGridInterpolator((self.temperature, self.pressure),
                                    self._values(name), method="linear")
        return f(np.stack([t.ravel(), p.ravel()], axis=-1)).reshape(t.shape)

    def temperature_of(self, name: str, value: ArrayLike, pressure: ArrayLike
                       ) -> np.ndarray:
        """The temperature at which the property `name` takes `value` at
        `pressure`, broadcast together, by the rule of the module
        docstring."""
        v, p = np.broadcast_arrays(np.asarray(value, dtype=float),
                                   np.asarray(pressure, dtype=float))
        shape = v.shape
        v, p = v.ravel(), p.ravel()
        self._within(p)
        table, axis = self._values(name), self.pressure
        j = np.clip(np.searchsorted(axis, p, side="right") - 1, 0, axis.size - 2)
        w = (p - axis[j]) / (axis[j + 1] - axis[j])
        col = (1.0 - w)[:, None] * table[:, j].T + w[:, None] * table[:, j + 1].T
        sign = np.sign(col[:, -1] - col[:, 0])
        c, x = col * sign[:, None], v * sign
        if np.any(sign == 0.0) or np.any(np.diff(c, axis=1) <= 0.0):
            raise ValueError(f"{name!r} of {self.name} is not strictly monotone in "
                             "temperature at every pressure asked for")
        if np.any((x < c[:, 0]) | (x > c[:, -1])):
            raise ValueError(f"a value of {name!r} lies outside what {self.name} "
                             "gives at its pressure")
        k = np.clip(np.sum(c <= x[:, None], axis=1) - 1, 0, c.shape[1] - 2)
        rows = np.arange(c.shape[0])
        lo, hi = c[rows, k], c[rows, k + 1]
        t = self.temperature
        out = t[k] + (x - lo) / (hi - lo) * (t[k + 1] - t[k])
        return out.reshape(shape)

    def cite(self) -> str:
        """The works the table rests on, one a line."""
        return "\n".join(self.citation)

    def write(self, path: str | os.PathLike[str]) -> Path:
        """The table written to `path` in the form of the module docstring."""
        import netCDF4
        path = Path(path)
        with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            ds.setncattr("name", self.name)
            if self.citation:
                ds.setncattr("citation", "\n".join(self.citation))
            for axis, unit in (("temperature", "K"), ("pressure", "Pa")):
                values = getattr(self, axis)
                ds.createDimension(axis, values.size)
                var = ds.createVariable(axis, "f8", (axis,))
                var.units = unit
                var[:] = values
            for key, values in self.properties.items():
                var = ds.createVariable(key, "f8", ("temperature", "pressure"))
                var.units = self.units.get(key, "1")
                var[:] = values
        return path

    @classmethod
    def read(cls, path: str | os.PathLike[str]) -> PropertyTable:
        """A table from a file of the form of the module docstring."""
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            axes = {}
            for axis, unit in (("temperature", "K"), ("pressure", "Pa")):
                var = ds.variables[axis]
                dims, factor = parse_unit(str(getattr(var, "units", unit)))
                if dims != parse_unit(unit)[0]:
                    raise ValueError(f"{path}: {axis} is in {var.units!r}, not a "
                                     f"{axis}")
                axes[axis] = factor * np.asarray(var[:], dtype=float)
            props, units = {}, {}
            for key, var in ds.variables.items():
                if key in axes:
                    continue
                if var.dimensions != ("temperature", "pressure"):
                    raise ValueError(f"{path}: {key} lies on {var.dimensions}, not "
                                     "(temperature, pressure)")
                stated = str(getattr(var, "units", "1"))
                dims, factor = parse_unit(stated)
                props[key] = factor * np.asarray(var[:], dtype=float)
                units[key] = dims.unit_string()
            attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        return cls(axes["temperature"], axes["pressure"], props,
                   name=str(attrs.get("name", Path(path).stem)), units=units,
                   citation=tuple(str(attrs.get("citation", "")).splitlines()))
