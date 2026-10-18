"""Spectra and correlations drawn.

`spectrum` draws a `slices.Spectrum`'s power by degree (or its
amplitude, the square root) as a line on a logarithmic scale, from a
first degree, 0 by default and 1 to leave the mean out.  `correlation`
draws a `slices.Correlation` as an image, degree along and radius up,
each value filling its cell, on the diverging colour map from -1 to 1,
a degree without power left blank; `correlation_total` draws the
correlation over all degrees against radius, radius upward as in a
profile.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..slices import Correlation, Spectrum
from .style import DIVERGING

__all__ = ["spectrum", "correlation", "correlation_total"]


def spectrum(ax: Any, spectrum_: Spectrum, *, amplitude: bool = False,
             first: int = 0, label: str | None = None, **style: Any) -> Any:
    """Draw a spectrum on `ax`: the power by degree, or the amplitude with
    `amplitude=True`, from degree `first`, the vertical scale logarithmic;
    `label` for a legend; `style` to `plot`.  Returns the line drawn."""
    if not 0 <= int(first) <= int(spectrum_.degrees[-1]):
        raise ValueError(f"first must be a degree in 0..{int(spectrum_.degrees[-1])}, "
                         f"got {first}")
    values = spectrum_.amplitude if amplitude else spectrum_.power
    keep = spectrum_.degrees >= int(first)
    line, = ax.plot(spectrum_.degrees[keep], values[keep], label=label, **style)
    ax.set_yscale("log")
    return line


def _edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges midway between centres, the end cells as wide as their
    neighbours'."""
    c = np.asarray(centres, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def correlation(ax: Any, correlation_: Correlation, *, scale: float = 1.0,
                first: int = 1, cmap: str | None = None, colorbar: bool = False,
                **style: Any) -> Any:
    """Draw a correlation by degree and radius on `ax` as an image, from
    degree `first` (1 by default, the mean left out); radii multiplied by
    `scale`; `cmap` in place of the diverging one; `colorbar` a colour
    bar; `style` to `pcolormesh`.  Returns the mesh drawn."""
    degrees = correlation_.degrees
    if not 0 <= int(first) <= int(degrees[-1]):
        raise ValueError(f"first must be a degree in 0..{int(degrees[-1])}, got "
                         f"{first}")
    keep = degrees >= int(first)
    order = np.argsort(correlation_.radii)
    radii = correlation_.radii[order] * scale
    values = np.ma.masked_invalid(correlation_.by_degree[order][:, keep])
    mesh = ax.pcolormesh(_edges(degrees[keep]), _edges(radii), values, shading="flat",
                         cmap=DIVERGING if cmap is None else cmap, vmin=-1.0, vmax=1.0,
                         **style)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=correlation_.label)
    return mesh


def correlation_total(ax: Any, correlation_: Correlation, *, scale: float = 1.0,
                      label: str | None = None, **style: Any) -> Any:
    """Draw the correlation over all degrees against radius on `ax`, radius
    upward, radii multiplied by `scale`; `label` for a legend; `style` to
    `plot`.  Returns the line drawn."""
    order = np.argsort(correlation_.radii)
    line, = ax.plot(correlation_.total[order], correlation_.radii[order] * scale,
                    label=label, **style)
    ax.set_xlim(-1.0, 1.0)
    return line
