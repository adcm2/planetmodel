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
built in SI, the perturbations' radii being metres.

The reference is the caller's when given, else the one the data set's
entry names: a model type of the catalogue in `REFERENCES`, built, or
another shipped data set, loaded.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..catalogue import PREM
from ..character import SCALAR
from ..fields import ComposedField, Field
from ..layerfunction import same_interval
from ..materials import voigt_vp, voigt_vs
from ..model import Layer, Model
from ..perturbation import PerturbedField
from ..units import Scales

if TYPE_CHECKING:
    from .registry import Dataset

__all__ = ["REFERENCES", "reference_of", "reference_field", "over_reference"]

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


def over_reference(reference: Model, perturbations: Mapping[str, Field], *,
                   kind: str, name: str | None = None) -> Model:
    """The model of perturbations of `kind` (one of `perturbation.KINDS`)
    over the reference's fields of their names, as the module docstring
    states it; every perturbation lies on one interval, in metres."""
    if not perturbations:
        raise ValueError("a model of perturbations needs at least one")
    ref = reference if reference.scales.is_si else reference.converted(Scales.SI)
    intervals = {tuple(map(float, f.interval)) for f in perturbations.values()}
    first = next(iter(intervals))
    if any(not same_interval(first, iv) for iv in intervals):
        raise ValueError(f"the perturbations lie on different intervals: "
                         f"{sorted(intervals)}")
    lo, hi = first
    sk = ref.skeleton
    tol = ref.geometry.rtol * sk.span
    if lo < sk.boundaries[0] - tol or hi > sk.boundaries[-1] + tol:
        raise ValueError(
            f"the perturbations on [{lo:g}, {hi:g}] reach beyond the reference "
            f"[{sk.boundaries[0]:g}, {sk.boundaries[-1]:g}]")
    geometry = ref.geometry
    if lo > sk.boundaries[0] + tol:
        geometry = geometry.hollowed(lo)
    if geometry.skeleton.locate(hi).boundary is None:
        geometry = geometry.refined([hi])
    layers = []
    for i in range(geometry.nlayers):
        a, b = geometry.skeleton.interval(i)
        if a < lo - tol or b > hi + tol:
            layers.append({})
            continue
        source = ref.layer(sk.locate(0.5 * (a + b)).layer)
        fields = {}
        for key, p in perturbations.items():
            base = reference_field(source, key)
            if not same_interval(base.interval, (a, b)):
                base = base.on_interval(a, b)
            fields[key] = PerturbedField(base, p.on_interval(a, b), kind=kind,
                                         name=key)
        layers.append(fields)
    return Model(geometry, layers, name=name)
