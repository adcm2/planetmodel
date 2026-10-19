"""Models built from data sets: a data set's perturbations on a model of one's own.

`transplanted(model, source)` is the steps of `datasets.processing` in
their order, for every name the source carries:

1. re-referenced (`rereferenced`): the source's perturbation kept, or
   under `mode="absolute"` its values against the model's own field; a
   data set of values made a perturbation of its own reference first,
   `source_reference` or, for a data set given by its entry, the
   reference the entry names;
2. its spherical mean at each radius removed when `remove_mean=True`
   (degree 0; on any field, no extra needed but for a resampling);
3. filtered by degree when `lmax=` (or `lmin=`) asks, rolled off over
   `roll_off` degrees;
4. spread onto whole layers (`ends=`, `extend=`, `taper=`, `layers=`);
5. attached by `perturbed`, so that a model type's own rules run: on an
   `Elastic` model a perturbed velocity re-derives the Love moduli.

A source name the model holds lands on itself; an isotropic name (`vs`,
`vp`) lands on the transversely isotropic pair the model holds where the
source does not carry that pair itself (`vs` on `vsv` and `vsh`, as
SPECFEM applies S40RTS to PREM), and is otherwise left out; `names=`
maps each source name to the model's names explicitly.  Conversions,
velocity-to-density scalings among them, are not made here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..model import Model
from ..perturbation import perturbed
from .entries import DatasetLike, as_dataset
from .fetch import load
from .models import reference_of
from .processing import demeaned, filtered, rereferenced, spread

__all__ = ["transplanted", "PAIRS"]

#: The transversely isotropic pair an isotropic name lands on.
PAIRS = {"vs": ("vsv", "vsh"), "vp": ("vpv", "vph")}


def _held(model: Model) -> set[str]:
    """Every name some layer of the model holds."""
    return {n for layer in model.layers for n in layer.names}


def _landing(source: Model, model: Model,
             names: Mapping[str, str | Sequence[str]] | None
             ) -> dict[str, tuple[str, ...]]:
    """Each source name and the model names it lands on."""
    if names is not None:
        return {k: (v,) if isinstance(v, str) else tuple(v) for k, v in names.items()}
    carried, held = _held(source), _held(model)
    out: dict[str, tuple[str, ...]] = {}
    for name in sorted(carried):
        if name in held:
            out[name] = (name,)
        elif (name in PAIRS and all(p in held for p in PAIRS[name])
              and not any(p in carried for p in PAIRS[name])):
            out[name] = PAIRS[name]
    if not out:
        raise ValueError(f"the source carries {sorted(carried)}, none of which the "
                         f"model holds (it holds {sorted(held)}); names= maps them")
    return out


def transplanted(model: Model, source: Model | DatasetLike, *,
                 names: Mapping[str, str | Sequence[str]] | None = None,
                 mode: str = "relative", remove_mean: bool = False,
                 layers: Sequence[int | str] | None = None,
                 ends: tuple[int | str, int | str] | None = None, extend: bool = False,
                 taper: float | tuple[float, float] | None = None,
                 lmax: int | None = None, lmin: int = 0, roll_off: int | None = None,
                 source_reference: Model | None = None,
                 kind: str = "relative") -> Model:
    """The model with the source's perturbations on it, by the steps of the
    module docstring: `source` a data set's model or its entry (loaded),
    the rest each step's own arguments."""
    if not isinstance(source, Model):
        d = as_dataset(source)
        loaded = load(d)
        if not isinstance(loaded, Model):
            raise TypeError(f"{d.name} reads as {type(loaded).__name__}, not a model "
                            "to transplant (a regional grid is merged)")
        if source_reference is None and d.reference is not None and mode == "relative":
            source_reference = reference_of(d)
        source = loaded
    landing = _landing(source, model, names)
    perturbations = rereferenced(source, model, names=list(landing), mode=mode,
                                 source_reference=source_reference, kind=kind)
    out = model
    for name, p in perturbations.items():
        segments = p.segments
        if remove_mean:
            segments = tuple(demeaned(s) for s in segments)
        if lmax is not None:
            segments = tuple(filtered(s, lmax=lmax, lmin=lmin, roll_off=roll_off)
                             for s in segments)
        elif lmin:
            raise ValueError("lmin= filters by degree, and needs lmax= beside it")
        p = type(p)(p.name, p.kind, segments)
        on = spread(p, out, layers=layers, ends=ends, extend=extend, taper=taper)
        for target in landing[name]:
            for index, field in on.items():
                out = perturbed(out, target, field, kind=p.kind, layers=[index])
    return out
