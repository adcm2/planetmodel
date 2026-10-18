"""The look every picture shares: boundary guides and colour scales.

A boundary of the skeleton is a faint guide beneath what is drawn
(`BOUNDARY_STYLE`).  A departure is drawn on a diverging colour map
(`DIVERGING`, negative red and positive blue) with limits symmetric
about zero, anything else on a sequential one (`SEQUENTIAL`), the limits
the finite range of what is drawn unless the caller gives them;
`colour_scale` decides them once for a picture of several parts, so that
every layer of a section shares one scale.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..slices import Shown

__all__ = ["BOUNDARY_STYLE", "DIVERGING", "SEQUENTIAL", "colour_scale"]

#: The style of the guide drawn at a layer boundary, beneath the picture.
BOUNDARY_STYLE = {"color": "0.85", "lw": 0.8, "zorder": 0.5}

#: The colour map of a departure: negative red, positive blue.
DIVERGING = "RdBu"

#: The colour map of anything else.
SEQUENTIAL = "viridis"


def colour_scale(values: Iterable[ArrayLike], shown: Shown, *, cmap: str | None = None,
                 vmin: float | None = None, vmax: float | None = None
                 ) -> tuple[str, float, float]:
    """The colour map and limits of a picture of `values`, the arrays of
    its parts: the caller's where given, else by the rule of the module
    docstring for what `shown` says the values are."""
    finite = [np.asarray(v, dtype=float) for v in values]
    finite = np.concatenate([v[np.isfinite(v)].ravel() for v in finite])
    if finite.size == 0:
        raise ValueError(f"{shown.label} has no finite value to draw")
    if shown.is_departure:
        bound = float(np.max(np.abs(finite)))
        bound = bound if bound > 0.0 else 1.0
        lo, hi = -bound, bound
        default = DIVERGING
    else:
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            lo, hi = lo - 0.5 * max(abs(lo), 1.0), hi + 0.5 * max(abs(hi), 1.0)
        default = SEQUENTIAL
    return (default if cmap is None else cmap, lo if vmin is None else float(vmin),
            hi if vmax is None else float(vmax))
