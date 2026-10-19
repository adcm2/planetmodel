"""Maps: a sphere slice drawn on cartopy's projections.

`sphere_map` draws a `slices.SphereSlice` on a cartopy `GeoAxes`, of any
projection, through the `maps` extra; other axes are refused by name.
Each value fills the cell around its node: the cell edges in longitude
lie midway between neighbouring longitudes, the last cell closing on the
first across the seam, and in colatitude midway between neighbouring
colatitudes, the end rows running to the poles, so that no kind of grid
leaves a gap (`cell_edges`).  The data are handed to cartopy in
longitude and latitude (`PlateCarree`), which reprojects and wraps them.
The colour scale is `style.colour_scale`'s, signed fields drawn
symmetric about zero with `diverging=True` and the limits put at a
percentile with `clip=`; `normalised=True` draws the slice divided by
its largest absolute value (`slices.normalised`); `coastlines=True` adds
cartopy's Natural Earth coastlines, which it fetches on first drawing.
cartopy is imported where a map is drawn.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..slices import SphereSlice, normalised as _normalised
from .style import colour_extend, colour_scale

if TYPE_CHECKING:
    from ..sampling import AngularGrid

__all__ = ["sphere_map", "cell_edges"]


def cell_edges(grid: AngularGrid) -> tuple[np.ndarray, np.ndarray]:
    """The edges of the cells around a grid's nodes, in degrees: the
    longitude edges (nphi + 1), midway between neighbours and spanning one
    whole turn, and the latitude edges (ntheta + 1), midway between
    neighbours, from 90 at the north pole to -90 at the south."""
    phi = np.degrees(grid.longitudes)
    closed = np.append(phi, phi[0] + 360.0)
    mid = 0.5 * (closed[:-1] + closed[1:])
    lon = np.append(mid[-1] - 360.0, mid)
    lat = 90.0 - np.degrees(grid.colatitudes)
    inner = 0.5 * (lat[:-1] + lat[1:])
    return lon, np.concatenate([[90.0], inner, [-90.0]])


def sphere_map(ax: Any, slice_: SphereSlice, *, cmap: str | None = None,
               vmin: float | None = None, vmax: float | None = None,
               diverging: bool = False, clip: float | None = None,
               normalised: bool = False, coastlines: bool = False,
               colorbar: bool = False, **style: Any) -> Any:
    """Draw a sphere slice on a cartopy `GeoAxes`, by the rules of the
    module docstring.

    `cmap`, `vmin` and `vmax` override the colour scale, `diverging` and
    `clip` choose it, `normalised` divides by the largest absolute value;
    `coastlines` adds the coastlines; `colorbar` a colour bar below the
    map, labelled with what the slice shows, with arrows at the ends the
    values run beyond; `style` goes to `pcolormesh`.  The whole sphere is
    shown.  Returns the mesh drawn.
    """
    if getattr(ax, "projection", None) is None or not hasattr(ax, "set_global"):
        raise ValueError("a map is drawn on a cartopy GeoAxes, made with "
                         "subplot_kw={'projection': cartopy.crs.Robinson()} say; "
                         "cartopy is the maps extra: pip install 'planetmodel[maps]'")
    import cartopy.crs as ccrs
    if normalised:
        slice_ = _normalised(slice_)
    lon, lat = cell_edges(slice_.grid)
    cmap, lo, hi = colour_scale([slice_.values], slice_.shown, cmap=cmap, vmin=vmin,
                                vmax=vmax, diverging=diverging, clip=clip)
    mesh = ax.pcolormesh(lon, lat, slice_.values, transform=ccrs.PlateCarree(),
                         shading="flat", cmap=cmap, vmin=lo, vmax=hi, **style)
    ax.set_global()
    if coastlines:
        ax.coastlines(linewidth=0.6)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.7,
                           label=slice_.label,
                           extend=colour_extend([slice_.values], lo, hi))
    return mesh
