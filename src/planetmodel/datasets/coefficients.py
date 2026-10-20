"""Tables of a surface's spherical-harmonic coefficients.

A surface published as harmonic coefficients (a planet's shape, its
topography, the radius of an interface beneath it) comes as a table of
C_lm and S_lm in a normalisation and phase convention its source states.
`read_coefficients` reads one into a `HarmonicFunction`, the library's
real orthonormal coefficients (`harmonics.from_convention`), to `lmax`
where it is given, so that a shape model to degree 5759 is read only as
far as it is wanted.  Two formats, both read by SHTOOLS:

- `"shtools"`, text: lines `l m C_lm S_lm`, any further columns (the
  coefficients' uncertainties) ignored, any lines before the first that
  reads so (a header) skipped, `D` read as `E` in an exponent;
- `"bshc"`, Curtin University's binary format: little-endian float64,
  lmin and lmax, then every C_lm degree by degree, orders 0 ... l, from
  lmin to lmax, then every S_lm in the same order.

A file whose name ends in `.gz` is read through gzip.  The entries of a
degree below a table's lmin are zero.  Numbers are as the file holds
them: in metres, for the surfaces a registry entry reads.
"""
from __future__ import annotations

import gzip
import io
import os
from collections.abc import Callable
from typing import IO

import numpy as np

from ..expansions import HarmonicFunction
from ..harmonics import from_convention

__all__ = ["FORMATS", "read_coefficients"]

#: The formats of a coefficient table: SHTOOLS' text and binary.
FORMATS = ("shtools", "bshc")


def _opener(path: str | os.PathLike[str]) -> Callable[[], IO[bytes]]:
    """Open a file for reading bytes, through gzip where it is gzipped."""
    name = os.fspath(path)
    if name.endswith(".gz"):
        return lambda: gzip.open(name, "rb")
    return lambda: open(name, "rb")


def _count(lmin: int, lmax: int) -> int:
    """The number of (l, m) with lmin <= l <= lmax and 0 <= m <= l."""
    return (lmax + 1) * (lmax + 2) // 2 - lmin * (lmin + 1) // 2


def _read_bshc(path: str | os.PathLike[str], lmax: int | None
               ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """The C and S of a bshc table to `lmax`, its lmin and the lmax read."""
    with _opener(path)() as f:
        head = np.frombuffer(f.read(16), dtype="<f8")
        if head.size != 2 or head[0] != round(head[0]) or head[1] != round(head[1]):
            raise ValueError(f"{path}: not a bshc table (no lmin and lmax)")
        lmin, top = int(head[0]), int(head[1])
        if not 0 <= lmin <= top:
            raise ValueError(f"{path}: lmin {lmin} and lmax {top} are not a range")
        want = top if lmax is None else min(int(lmax), top)
        n, total = _count(lmin, want), _count(lmin, top)
        c = np.frombuffer(f.read(8 * n), dtype="<f8")
        f.seek(16 + 8 * total)
        s = np.frombuffer(f.read(8 * n), dtype="<f8")
    if c.size != n or s.size != n:
        raise ValueError(f"{path}: the table ends before its lmax {top}")
    return c, s, lmin, want


def _read_shtools(path: str | os.PathLike[str], lmax: int | None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The l, m, C and S of an SHTOOLS text table, to `lmax`."""
    with _opener(path)() as f:
        lines = f.read().decode("ascii").replace("D", "E").replace("d", "e")
    rows = lines.splitlines()
    start = 0
    for start, line in enumerate(rows):
        words = line.replace(",", " ").split()
        try:
            int(words[0]), int(words[1]), float(words[2]), float(words[3])
            break
        except (ValueError, IndexError):
            continue
    else:
        raise ValueError(f"{path}: no line of l, m, C_lm and S_lm")
    table = np.loadtxt(io.StringIO("\n".join(rows[start:]).replace(",", " ")),
                       usecols=(0, 1, 2, 3), ndmin=2)
    l, m = table[:, 0].astype(int), table[:, 1].astype(int)
    if np.any(m < 0) or np.any(m > l):
        raise ValueError(f"{path}: an order outside 0 ... l")
    keep = np.ones(l.size, dtype=bool) if lmax is None else l <= int(lmax)
    return l[keep], m[keep], table[keep, 2], table[keep, 3]


def read_coefficients(path: str | os.PathLike[str], *, format: str,
                      normalization: str = "4pi", csphase: int = 1,
                      lmax: int | None = None) -> HarmonicFunction:
    """A coefficient table as the function of the angles it expands, by
    the rules of the module docstring: `format` one of `FORMATS`;
    `normalization` and `csphase` the source's convention, as
    `harmonics.from_convention` takes them (SHTOOLS' own, 4 pi
    normalised without the Condon-Shortley phase, by default); `lmax` the
    highest degree to read, the table's own by default."""
    if format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
    if lmax is not None and int(lmax) < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    if format == "bshc":
        cs, ss, lmin, top = _read_bshc(path, lmax)
        l, m = np.tril_indices(top + 1)
        held = l >= lmin
        l, m = l[held], m[held]
    else:
        l, m, cs, ss = _read_shtools(path, lmax)
        if l.size == 0:
            raise ValueError(f"{path}: no coefficient to degree {lmax}")
        top = int(l.max())
    c = np.zeros((2, top + 1, top + 1))
    c[0, l, m] = cs
    c[1, l, m] = np.where(m > 0, ss, 0.0)
    return HarmonicFunction(from_convention(c, normalization=normalization,
                                            csphase=csphase))
