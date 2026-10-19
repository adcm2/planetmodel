"""A data set as the model it is: perturbations held over their reference.

A published model stated as perturbations of a reference, S40RTS as the
relative perturbation of PREM's shear velocity say, is read as a
`Model` holding exactly that.  Its geometry is the reference's, hollowed
at the inner radius of the perturbations' support and given a boundary
at the outer one, so that it runs to the reference's surface (a depth
means what the source's depths mean) and no field jumps inside a layer;
on each layer inside the support every perturbed name is a
`PerturbedField` over the reference's field of that name, restated on
the layer, and the layers beyond the support hold nothing.  Where the
data set states an isotropic velocity and the reference holds a
transversely isotropic pair, the reference is their Voigt average
(`materials.voigt_vs`, `voigt_vp`, a composed field).  Everything is
built in SI, the perturbations' radii being metres.  A source with
discontinuities of its own (a grid's near-repeated depths) gives each
name as the fields that tile its support between them, and each of its
discontinuities becomes a boundary too.  A data set of values rather
than perturbations is a model of its own (`standalone`): its support's
segments as layers, hollow below, and an empty layer above it up to the
source's surface.

The reference is the caller's when given, else the one the data set's
entry names: a model type of the catalogue in `REFERENCES`, built, or
another shipped data set, loaded.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from ..catalogue import PREM
from ..character import SCALAR
from ..fields import ComposedField, Field
from ..geometry import Geometry
from ..layerfunction import same_interval
from ..materials import voigt_vp, voigt_vs
from ..model import Layer, Model
from ..perturbation import PerturbedField
from ..skeleton import Skeleton
from ..units import Scales

if TYPE_CHECKING:
    from .registry import Dataset

__all__ = ["REFERENCES", "Segments", "reference_of", "reference_field",
           "over_reference", "standalone"]

#: The model types of the catalogue a data set's entry may name as its
#: reference, by that name.
REFERENCES: Mapping[str, Callable[[], Model]] = MappingProxyType({"PREM": PREM})

#: The Voigt averages a reference gives for an isotropic name it lacks.
_VOIGT = {"vs": (("vsv", "vsh"), voigt_vs), "vp": (("vpv", "vph"), voigt_vp)}


def reference_of(dataset: Dataset, *, reference: Model | None = None) -> Model:
    """The model a data set's perturbations are relative to: `reference`
    when given, else its entry's, built from `REFERENCES` or loaded from
    the shipped data set of that name."""
    if reference is not None:
        return reference
    name = dataset.reference
    if name is None:
        raise ValueError(f"{dataset.name} names no reference model; pass "
                         "reference= to read its perturbations against one")
    if name in REFERENCES:
        return REFERENCES[name]()
    from .entries import REGISTRY
    from .fetch import load

    if name not in REGISTRY:
        raise ValueError(f"{dataset.name} is relative to {name!r}, which is neither "
                         f"a model of the catalogue ({sorted(REFERENCES)}) nor a "
                         "shipped data set; pass reference=")
    model = load(name)
    if not isinstance(model, Model):
        raise TypeError(f"the reference {name!r} of {dataset.name} reads as "
                        f"{type(model).__name__}, not a Model")
    return model


def reference_field(layer: Layer, name: str) -> Field:
    """The field `name` of a reference layer, or the Voigt average of its
    transversely isotropic pair for `vs` and `vp`; refused by name."""
    if name in layer:
        return layer[name]
    if name in _VOIGT:
        pair, average = _VOIGT[name]
        if all(p in layer for p in pair):
            return ComposedField(average, (layer[pair[0]], layer[pair[1]]),
                                 character=SCALAR, name=name)
    raise ValueError(f"the reference layer {layer.index} ({layer.name}) holds no "
                     f"{name!r} to perturb (it holds {sorted(layer.names)})")


#: A perturbation, or the fields that tile its support from the centre out,
#: one on each segment between the source's own discontinuities.
type Segments = Field | Sequence[Field]


def _segments(fields: Mapping[str, Segments]) -> tuple[dict[str, tuple[Field, ...]],
                                                       list[float]]:
    """Every name's segments, checked to tile one support the same way for
    every name; with the support's breaks from the centre out."""
    if not fields:
        raise ValueError("a model of a data set needs at least one field")
    tiled = {k: (v,) if isinstance(v, Field) else tuple(v) for k, v in fields.items()}
    first = next(iter(tiled.values()))
    breaks = [float(first[0].interval[0])] + [float(f.interval[1]) for f in first]
    tol = 1e-9 * (breaks[-1] - breaks[0])
    for key, segs in tiled.items():
        ends = [float(segs[0].interval[0])] + [float(f.interval[1]) for f in segs]
        lows = [float(f.interval[0]) for f in segs[1:]]
        if (len(ends) != len(breaks)
                or not np.allclose(ends, breaks, rtol=0.0, atol=tol)
                or not np.allclose(lows, ends[1:-1], rtol=0.0, atol=tol)):
            raise ValueError(f"the fields of {key!r} do not tile the support "
                             f"{breaks[0]:g} ... {breaks[-1]:g} as the others do")
    return tiled, breaks


def _segment_at(segs: tuple[Field, ...], a: float, b: float) -> Field:
    """The segment holding [a, b], restated on it."""
    mid = 0.5 * (a + b)
    for f in segs:
        lo, hi = (float(x) for x in f.interval)
        if lo <= mid <= hi:
            return f if same_interval(f.interval, (a, b)) else f.on_interval(a, b)
    raise ValueError(f"no segment holds [{a:g}, {b:g}]")


def standalone(fields: Mapping[str, Segments], *, surface: float,
               name: str | None = None) -> Model:
    """The model of a data set of values (not perturbations): a skeleton
    from the inner end of its support to `surface` (or to the support's top
    where that is higher), a boundary at every segment's end, each name on
    the layers its segments cover and nothing above them; hollow unless the
    support reaches the centre."""
    tiled, breaks = _segments(fields)
    top = max(float(surface), breaks[-1])
    radii = sorted(set(breaks) | {top})
    geometry = Geometry(Skeleton(radii))
    layers = []
    for i in range(geometry.nlayers):
        a, b = geometry.skeleton.interval(i)
        if b > breaks[-1] * (1 + 1e-12):
            layers.append({})
        else:
            layers.append({key: _segment_at(segs, a, b) for key, segs in tiled.items()})
    return Model(geometry, layers, name=name)


def over_reference(reference: Model, perturbations: Mapping[str, Segments], *,
                   kind: str, name: str | None = None) -> Model:
    """The model of perturbations of `kind` (one of `perturbation.KINDS`)
    over the reference's fields of their names, as the module docstring
    states it: each a field on the support, or fields tiling it between
    the source's own discontinuities (each then a boundary too); in
    metres."""
    tiled, breaks = _segments(perturbations)
    ref = reference if reference.scales.is_si else reference.converted(Scales.SI)
    lo, hi = breaks[0], breaks[-1]
    sk = ref.skeleton
    tol = ref.geometry.rtol * sk.span
    if lo < sk.boundaries[0] - tol or hi > sk.boundaries[-1] + tol:
        raise ValueError(
            f"the perturbations on [{lo:g}, {hi:g}] reach beyond the reference "
            f"[{sk.boundaries[0]:g}, {sk.boundaries[-1]:g}]")
    geometry = ref.geometry
    if lo > sk.boundaries[0] + tol:
        geometry = geometry.hollowed(lo)
    inner = [r for r in breaks[1:] if geometry.skeleton.locate(r).boundary is None]
    if inner:
        geometry = geometry.refined(inner)
    layers = []
    for i in range(geometry.nlayers):
        a, b = geometry.skeleton.interval(i)
        if a < lo - tol or b > hi + tol:
            layers.append({})
            continue
        source = ref.layer(sk.locate(0.5 * (a + b)).layer)
        fields = {}
        for key, segs in tiled.items():
            base = reference_field(source, key)
            if not same_interval(base.interval, (a, b)):
                base = base.on_interval(a, b)
            fields[key] = PerturbedField(base, _segment_at(segs, a, b), kind=kind,
                                         name=key)
        layers.append(fields)
    return Model(geometry, layers, name=name)
