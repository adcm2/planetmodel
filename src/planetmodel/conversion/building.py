"""Laws applied to models: anomalies read from a model, and the fields
they make.

An `Anomaly` is a departure from the radial reference on some layers of
a model, of one kind (`ANOMALIES`): a relative velocity anomaly dlnVs,
or a temperature anomaly dT in kelvin.  Its fields are the model's own,
composed with the laws, never sampled: functions of the reference point
whose values are in SI whatever the model's units, since the laws are
(the radius is converted to metres by the model's scales on the way in).
It carries the citations of the parameter sets that made it.

- `velocity_anomaly(model)` reads dlnVs from the model's shear speed: a
  perturbed field gives its own perturbation (relative as it stands, log
  and absolute converted), a fully 3D one is taken against its spherical
  mean (`means.spherical_mean`, on `grid` where it has none of its own),
  and a radial one has none, so its layer is left out.  The shear speed
  is `vs` where the layer holds it, else the Voigt average of the
  transversely isotropic pair, Vs^2 = (2 vsv^2 + vsh^2) / 3, taken
  against the same average of the references, else the one `name`
  names; a fluid layer, whose reference is zero, is left out.
- `temperature_anomaly(velocity, ...)` turns it into dT, by a
  `Sensitivity`, or through density by a `DensityScaling` and an
  `Expansivity` (dT = -R dlnVs / alpha), as sl3d's chain does.
- `with_viscosity(model, anomaly, law)` writes the viscosity of each
  layer of the anomaly as a log perturbation of the layer's own radial
  viscosity (or of the reference of a perturbed one): eta = eta_0
  exp(dln eta), the law the parameter set's, which states the anomaly
  it takes.  With `renormalise=grid` each radius's mean is put back to
  eta_0, dln eta less ln <exp(dln eta)>, the mean by the grid's
  quadrature (Latychev et al. 2005).
- `with_density(model, anomaly, law)` writes `rho` as a relative
  perturbation of its reference, through `with_field`, so that an
  `Elastic` model's moduli follow.
- `attached(model, anomaly, name, spec)` attaches the anomaly itself
  under a name and spec of the caller's, converted to the model's units,
  for a look at it.
- `with_power_law(model, *, exponent, transition_stress)` sets power-law
  creep's two parameters (`rheology.POWER_LAW_NAMES`) directly, numbers
  or profiles in SI, as radial fields on every layer holding a viscosity
  or Prony branches (or on `layers`).
- `sl3d_viscosity(model)` is sl3d's chain in one call: the velocity
  anomaly, through sl3d's density scaling and expansivity to a
  temperature, and its exponential law, with no renormalisation.  sl3d
  applies it to S20RTS without its degree 0 over its viscoelastic
  region, which is the caller's to set up (`datasets.transplanted` with
  `remove_mean=True`); it holds the Fortran's own output at its nodes to
  the Fortran's single precision.

Each model written cites the works of the anomaly and the law after its
own.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from ..character import SCALAR
from ..fields import ComposedField, Field, RadialField
from ..layerfunction import NumericLayer, polynomial_layer
from ..means import spherical_mean
from ..model import Model, cited
from ..perturbation import PerturbedField
from ..units import Scales
from ..vocabulary import FieldSpec
from .profiles import ProfileLike, as_profile
from .parameters import (SL3D_DENSITY_SCALING, SL3D_EXPANSIVITY, SL3D_VISCOSITY,
                         DensityLaw, DensityScaling, Expansivity, ParameterSet,
                         Sensitivity, ViscosityLaw)

if TYPE_CHECKING:
    from ..sampling import AngularGrid

__all__ = ["ANOMALIES", "Anomaly", "velocity_anomaly", "temperature_anomaly",
           "with_viscosity", "with_density", "attached", "sl3d_viscosity",
           "with_power_law"]

#: The kinds of anomaly: a relative velocity anomaly, a temperature anomaly.
ANOMALIES = ("velocity", "temperature")


@dataclass(frozen=True)
class Anomaly:
    """A departure from the radial reference, by the module docstring:
    `kind` one of `ANOMALIES`, `fields` by layer index, in SI, the
    model's `scales`, and the `citation` of the sets that made it."""

    kind: str
    fields: Mapping[int, Field]
    scales: Scales
    _: KW_ONLY
    citation: Iterable[str] = ()

    def __post_init__(self) -> None:
        if self.kind not in ANOMALIES:
            raise ValueError(f"kind must be one of {ANOMALIES}, got {self.kind!r}")
        if not self.fields:
            raise ValueError("an anomaly lies on at least one layer")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "citation", cited((), self.citation))

    @property
    def layers(self) -> tuple[int, ...]:
        """The indices of the layers it lies on."""
        return tuple(sorted(self.fields))


def _radius(f: Field) -> RadialField:
    """r itself on a field's interval, in the model's length."""
    return RadialField(f.interval, polynomial_layer(f.interval, [0.0, 1.0]))


def _composed(fn: Callable[..., np.ndarray], sources: list[Field], *,
              name: str | None = None) -> ComposedField:
    return ComposedField(fn, sources, character=SCALAR, name=name)


def _relative(f: Field, grid: AngularGrid | None) -> tuple[Field, Field] | None:
    """(the field's relative departure, its radial reference), None for a
    radial field."""
    pert = getattr(f, "perturbation", None)
    if pert is not None:
        ref, kind = f.reference, f.kind
        if kind == "relative":
            return pert, ref
        if kind == "log":
            return _composed(np.expm1, [pert]), ref
        return _composed(lambda p, r0: p / r0, [pert, ref]), ref
    if getattr(f, "is_radial", False):
        return None
    mean = spherical_mean(f, grid=grid)
    return _composed(lambda v, v0: v / v0 - 1.0, [f, mean]), mean


def _zero(ref: Field) -> bool:
    """Whether a radial reference vanishes on its interval (a fluid's vs)."""
    lo, hi = ref.interval
    v = np.asarray(ref(np.linspace(lo, hi, 5)), dtype=float)
    return bool(np.all(v == 0.0))


def velocity_anomaly(model: Model, *, name: str | None = None,
                     layers: Iterable[int | str] | None = None,
                     grid: AngularGrid | None = None) -> Anomaly:
    """dlnVs on every layer whose shear speed varies (or on `layers`), by
    the rule of the module docstring; `name` the speed to read, `grid` the
    quadrature of a fully 3D field's mean."""
    which = (range(model.nlayers) if layers is None
             else sorted({model.layer(k).index for k in layers}))
    out: dict[int, Field] = {}
    for i in which:
        layer = model.layer(i)
        if name is not None or "vs" in layer:
            key = "vs" if name is None else name
            if key not in layer:
                raise KeyError(f"layer {i} ({layer.name}) holds no {key!r}")
            got = _relative(layer[key], grid)
            if got is None or _zero(got[1]):
                continue
            out[i] = got[0]
        elif "vsv" in layer and "vsh" in layer:
            v, h = _relative(layer["vsv"], grid), _relative(layer["vsh"], grid)
            if v is None and h is None:
                continue
            v = v or (_composed(np.zeros_like, [layer["vsv"]]), layer["vsv"])
            h = h or (_composed(np.zeros_like, [layer["vsh"]]), layer["vsh"])
            if _zero(v[1]) and _zero(h[1]):
                continue

            def voigt(dv: np.ndarray, dh: np.ndarray, v0: np.ndarray,
                      h0: np.ndarray) -> np.ndarray:
                # sqrt(1 + x) - 1 as x / (1 + sqrt(1 + x)), no cancellation
                a, b = v0 * v0, h0 * h0
                x = (2.0 * a * dv * (2.0 + dv) + b * dh * (2.0 + dh)) / (2.0 * a + b)
                return x / (1.0 + np.sqrt(1.0 + x))

            out[i] = _composed(voigt, [v[0], h[0], v[1], h[1]])
        elif layers is not None:
            raise KeyError(f"layer {i} ({layer.name}) holds neither vs nor the pair "
                           "vsv and vsh; name= says which speed to read")
    if not out:
        raise ValueError("no layer's shear speed departs from its reference")
    return Anomaly("velocity", out, model.scales)


def _check(law: ParameterSet, anomaly: Anomaly, method: str, what: str) -> None:
    if not callable(getattr(law, method, None)):
        raise TypeError(f"{law.name}'s set has no law of {what} ({method})")
    if law.takes != anomaly.kind:
        raise ValueError(f"{law.name}'s law takes a {law.takes} anomaly; this one is "
                         f"a {anomaly.kind} anomaly")


def _applied(anomaly: Anomaly, apply: Callable[[np.ndarray, np.ndarray], np.ndarray],
             i: int, *, name: str | None = None) -> ComposedField:
    """apply(values, r in metres) on layer i's anomaly, composed."""
    f = anomaly.fields[i]
    length = anomaly.scales.length
    return _composed(lambda r, v: apply(v, r * length), [_radius(f), f], name=name)


def temperature_anomaly(velocity: Anomaly, *, sensitivity: Sensitivity | None = None,
                        scaling: DensityScaling | None = None,
                        expansivity: Expansivity | None = None) -> Anomaly:
    """dT from a velocity anomaly: by `sensitivity`, or by `scaling` and
    `expansivity` together (dT = -R dlnVs / alpha)."""
    if velocity.kind != "velocity":
        raise ValueError(f"a temperature anomaly is made from a velocity anomaly, "
                         f"not a {velocity.kind} one")
    if sensitivity is not None:
        if scaling is not None or expansivity is not None:
            raise ValueError("give sensitivity=, or scaling= and expansivity=, "
                             "not both")
        sets: tuple[ParameterSet, ...] = (sensitivity,)

        def apply(v: np.ndarray, r: np.ndarray) -> np.ndarray:
            return sensitivity.temperature(v, r)
    elif scaling is not None and expansivity is not None:
        sets = (scaling, expansivity)

        def apply(v: np.ndarray, r: np.ndarray) -> np.ndarray:
            return expansivity.temperature(scaling.relative_density(v, r), r)
    else:
        raise ValueError("a temperature anomaly needs sensitivity=, or scaling= and "
                         "expansivity= together")
    fields = {i: _applied(velocity, apply, i) for i in velocity.layers}
    return Anomaly("temperature", fields, velocity.scales,
                   citation=cited(velocity.citation,
                                  [w for s in sets for w in s.citation]))


def _reference(f: Field) -> Field:
    """A field's radial reference: itself if radial, a perturbed one's
    reference; anything else refused by the caller."""
    ref = getattr(f, "reference", None)
    if ref is not None:
        return ref
    if getattr(f, "is_radial", False):
        return f
    raise ValueError(f"{f!r} is neither radial nor perturbed, so it has no radial "
                     "reference to perturb")


def with_viscosity(model: Model, anomaly: Anomaly, law: ViscosityLaw, *,
                   renormalise: AngularGrid | None = None,
                   name: str = "viscosity") -> Model:
    """The model with its viscosity on the anomaly's layers a log
    perturbation of the layer's radial viscosity by `law` (a parameter set
    with `log_viscosity`), by the rule of the module docstring."""
    _check(law, anomaly, "log_viscosity", "viscosity")
    out = model
    for i in anomaly.layers:
        layer = model.layer(i)
        if name not in layer:
            raise KeyError(f"layer {i} ({layer.name}) holds no {name!r} to perturb; "
                           "give it a radial one first")
        ref = _reference(layer[name])
        dln = _applied(anomaly, law.log_viscosity, i, name=name)
        if renormalise is not None:
            mean = spherical_mean(_composed(np.exp, [dln]), grid=renormalise)
            dln = _composed(lambda d, m: d - np.log(m), [dln, mean], name=name)
        out = out.with_field(i, name, PerturbedField(ref, dln, kind="log", name=name),
                             replace=True)
    return out.with_citation(*anomaly.citation, *law.citation)


def with_density(model: Model, anomaly: Anomaly, law: DensityLaw, *,
                 name: str = "rho") -> Model:
    """The model with `rho` on the anomaly's layers a relative
    perturbation of its reference by `law` (a parameter set with
    `relative_density`)."""
    _check(law, anomaly, "relative_density", "density")
    out = model
    for i in anomaly.layers:
        layer = model.layer(i)
        if name not in layer:
            raise KeyError(f"layer {i} ({layer.name}) holds no {name!r}")
        ref = _reference(layer[name])
        drho = _applied(anomaly, law.relative_density, i, name=name)
        out = out.with_field(i, name, PerturbedField(ref, drho, kind="relative",
                                                     name=name), replace=True)
    return out.with_citation(*anomaly.citation, *law.citation)


def attached(model: Model, anomaly: Anomaly, name: str, spec: FieldSpec) -> Model:
    """The model with the anomaly itself under `name` on its layers, its
    values converted from SI to the model's units by `spec`'s dimensions,
    `spec` added to the model's specs."""
    k = model.scales.factor(spec.dimensions)
    layers = [dict(lay.fields) for lay in model.layers]
    for i in anomaly.layers:
        if name in layers[i]:
            raise ValueError(f"layer {i} already holds {name!r}")
        layers[i][name] = ComposedField(lambda v: v / k, [anomaly.fields[i]],
                                        character=spec.character, name=name)
    specs = {**{n: s for n, s in model.specs.items()}, name: spec}
    return model.replaced(layers=layers, specs=specs).with_citation(*anomaly.citation)


def sl3d_viscosity(model: Model, *, layers: Iterable[int | str] | None = None,
                   name: str = "viscosity") -> Model:
    """The model with the viscosity of sl3d's chain on the layers whose shear
    speed varies (or `layers`), by the rule of the module docstring."""
    dt = temperature_anomaly(velocity_anomaly(model, layers=layers),
                             scaling=SL3D_DENSITY_SCALING,
                             expansivity=SL3D_EXPANSIVITY)
    return with_viscosity(model, dt, SL3D_VISCOSITY, name=name)


def with_power_law(model: Model, *, exponent: ProfileLike,
                   transition_stress: ProfileLike,
                   layers: Iterable[int | str] | None = None) -> Model:
    """The model with `stress_exponent` and `transition_stress` (in Pa)
    on its creeping layers, by the rule of the module docstring."""
    from ..rheology import POWER_LAW_NAMES, branches
    which = ([lay.index for lay in model.layers
              if "viscosity" in lay or branches(lay)]
             if layers is None else sorted({model.layer(k).index for k in layers}))
    if not which:
        raise ValueError("no layer holds a viscosity or Prony branches to creep")
    profiles = dict(zip(POWER_LAW_NAMES, (as_profile(exponent),
                                          as_profile(transition_stress))))
    length = model.scales.length
    out = model
    for i in which:
        lo, hi = model.layer(i).interval
        for name, profile in profiles.items():
            k = model.scales.factor(model.spec(name).dimensions)
            fn = NumericLayer((lo, hi), lambda r, p=profile, k=k: p(r * length) / k)
            out = out.with_field(i, name, RadialField(
                (lo, hi), fn, character=model.spec(name).character, name=name),
                replace=True)
    return out
