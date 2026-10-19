"""Prony branches: generalised Maxwell bodies, built and put on a model.

A generalised Maxwell body in shear is a sum of branches, each a
fraction w_k of the unrelaxed shear modulus mu_U relaxing with time
tau_k, its complex modulus at angular frequency omega

    mu(omega) / mu_U = 1 - sum_k w_k / (1 + i omega tau_k),

the relaxed part (1 - sum w_k) mu_U (`rheology` reads the branches;
`vocabulary.FAMILIES` names them).  A `PronySeries` holds a body's
strengths w_k and its times in units of the Maxwell time eta / mu
(c_k = tau_k mu / eta), so that one series serves every point of a
model whose viscosity varies: `with_branches(model, series)` writes
`relaxation_strength_k` = w_k and `relaxation_time_k` = c_k eta / mu on
every solid layer holding a viscosity (or on `layers`), and removes the
viscosity, which the branches replace.  Built:

- `maxwell_series()`: one branch, w = 1, c = 1, the viscosity itself;
- `burgers_series(*, strength, viscosity_ratio)`: a Maxwell element (mu_M,
  eta_M) in series with a Kelvin-Voigt one (mu_K, eta_K), strength Delta
  = mu_M / mu_K and viscosity_ratio = eta_M / eta_K, exactly two
  branches with no relaxed part: the compliance J(s) = 1/mu_M + 1/(eta_M
  s) + 1/(mu_K + eta_K s) vanishes at the two negative roots s_k of
  eta_M eta_K s^2 + (eta_M mu_K + mu_M eta_K + mu_M eta_M) s + mu_M mu_K,
  so tau_k = -1 / s_k, and the residues of 1/J there give mu_k, which sum
  to mu_M (Ivins et al. 2022 call Delta the relaxation strength; Caron
  et al. 2017 take Delta and the viscosity ratio from 1 to 10);
- `fitted_series(modulus, *, band, ...)`: the strengths of branches at
  fixed times, `per_decade` of them spread a decade beyond the band each
  way, fitted by non-negative least squares to a complex modulus
  mu(omega) / mu_U given as a function of omega times the Maxwell time,
  at 20 frequencies a decade over `band`; for a body that flows
  (`fluid=True`) the strengths sum to one and its long-time viscosity is
  eta (sum w_k c_k = 1), both held as heavily weighted rows; `misfit` is
  the largest relative misfit of the modulus over the band;
- `andrade_series(alpha, *, andrade_time, band, ...)`: such a fit to the
  Andrade body J(omega) / J_U = 1 + Gamma(1 + alpha) (i omega tau_A)^-alpha
  + 1 / (i omega tau_M), tau_A in Maxwell times (`andrade_time`, 1 by
  default), which has no finite Prony form.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import KW_ONLY, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import nnls
from scipy.special import gamma

from ..character import SCALAR
from ..fields import ComposedField, constant_field
from ..materials import is_fluid, kappa_mu
from ..model import Model, cited

__all__ = ["PronySeries", "maxwell_series", "burgers_series", "fitted_series",
           "andrade_series", "with_branches"]


@dataclass(frozen=True)
class PronySeries:
    """A generalised Maxwell body by the module docstring: `strengths` w_k
    and `times` c_k in Maxwell times, `name`, the `misfit` of a fitted one
    and the works it rests on."""

    strengths: Sequence[float]
    times: Sequence[float]
    _: KW_ONLY
    name: str
    misfit: float = 0.0
    citation: Iterable[str] = ()

    def __post_init__(self) -> None:
        w = tuple(float(x) for x in self.strengths)
        c = tuple(float(x) for x in self.times)
        if len(w) != len(c) or not w:
            raise ValueError("a series has as many strengths as times, at least one")
        if any(not x >= 0.0 for x in w) or any(not x > 0.0 for x in c):
            raise ValueError("strengths are non-negative and times positive")
        if sum(w) > 1.0 + 1e-12:
            raise ValueError(f"the strengths sum to {sum(w):.15g}, more than the "
                             "whole unrelaxed modulus")
        object.__setattr__(self, "strengths", w)
        object.__setattr__(self, "times", c)
        object.__setattr__(self, "citation", cited((), self.citation))

    def modulus(self, omega: ArrayLike) -> np.ndarray:
        """mu(omega) / mu_U at `omega` in units of the inverse Maxwell time."""
        x = np.asarray(omega, dtype=float)
        out = np.ones(x.shape, dtype=complex)
        for w, c in zip(self.strengths, self.times):
            out = out - w / (1.0 + 1j * x * c)
        return out


def maxwell_series() -> PronySeries:
    """The Maxwell body, one branch: w = 1, tau = eta / mu."""
    return PronySeries((1.0,), (1.0,), name="Maxwell")


def burgers_series(*, strength: float, viscosity_ratio: float) -> PronySeries:
    """The Burgers body of the module docstring, Delta = mu_M / mu_K and
    eta_M / eta_K, as its two branches."""
    if not (strength > 0.0 and viscosity_ratio > 0.0):
        raise ValueError("strength and viscosity_ratio are positive")
    mu_m, eta_m = 1.0, 1.0
    mu_k, eta_k = 1.0 / strength, 1.0 / viscosity_ratio
    a, b, c = eta_m * eta_k, eta_m * mu_k + mu_m * eta_k + mu_m * eta_m, mu_m * mu_k
    root = np.sqrt(b * b - 4.0 * a * c)
    s = np.array([(-b - root) / (2.0 * a), 2.0 * c / (-b - root)])
    mu = mu_m * eta_m * (mu_k + eta_k * s) / (2.0 * a * s + b)
    return PronySeries(tuple(mu / mu_m), tuple(-1.0 / s),
                       name=f"Burgers (strength {strength:g}, viscosity ratio "
                            f"{viscosity_ratio:g})")


def fitted_series(modulus: Callable[[np.ndarray], ArrayLike], *,
                  band: tuple[float, float], per_decade: int = 2, fluid: bool = True,
                  name: str = "fitted") -> PronySeries:
    """The series fitted to `modulus`, mu(omega) / mu_U at omega in inverse
    Maxwell times, by the rule of the module docstring."""
    lo, hi = (float(x) for x in band)
    if not 0.0 < lo < hi:
        raise ValueError(f"band is (low, high) with 0 < low < high, got {band}")
    if per_decade < 1:
        raise ValueError("per_decade is at least one")
    omega = np.logspace(np.log10(lo), np.log10(hi), int(20 * np.log10(hi / lo)) + 1)
    n = int(round(per_decade * (np.log10(hi / lo) + 2.0))) + 1
    times = np.logspace(-np.log10(hi) - 1.0, -np.log10(lo) + 1.0, n)
    target = np.asarray(modulus(omega), dtype=complex)
    if not np.all(np.isfinite(target)):
        raise ValueError("the modulus is not finite over the band")
    basis = 1.0 / (1.0 + 1j * omega[:, None] * times[None, :])
    scale = 1.0 / np.abs(target)
    rows = [np.real(basis) * scale[:, None], np.imag(basis) * scale[:, None]]
    rhs = [np.real(1.0 - target) * scale, np.imag(1.0 - target) * scale]
    if fluid:
        heavy = 1e3 * np.sqrt(omega.size)
        rows += [heavy * np.ones((1, n)), heavy * times[None, :]]
        rhs += [np.array([heavy]), np.array([heavy])]
    w, _ = nnls(np.vstack(rows), np.concatenate(rhs), maxiter=50 * n)
    keep = w > 1e-14 * w.max()
    w, c = w[keep], times[keep]
    if w.sum() > 1.0:
        w = w / w.sum()
    series = PronySeries(tuple(w), tuple(c), name=name)
    misfit = float(np.max(np.abs(series.modulus(omega) - target) / np.abs(target)))
    return PronySeries(series.strengths, series.times, name=name, misfit=misfit)


def andrade_series(alpha: float, *, andrade_time: float = 1.0,
                   band: tuple[float, float] = (1e-3, 1e3), per_decade: int = 2
                   ) -> PronySeries:
    """The Andrade body of the module docstring fitted over `band` (in
    inverse Maxwell times), `alpha` in (0, 1)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha lies in (0, 1), got {alpha}")
    if not andrade_time > 0.0:
        raise ValueError("andrade_time is positive")
    g = gamma(1.0 + alpha)

    def modulus(omega: np.ndarray) -> np.ndarray:
        j = 1.0 + g * (1j * omega * andrade_time) ** (-alpha) + 1.0 / (1j * omega)
        return 1.0 / j

    return fitted_series(modulus, band=band, per_decade=per_decade, fluid=True,
                         name=f"Andrade (alpha {alpha:g}, Andrade time "
                              f"{andrade_time:g} Maxwell times)")


def with_branches(model: Model, series: PronySeries, *,
                  layers: Iterable[int | str] | None = None,
                  name: str = "viscosity") -> Model:
    """The model with `series` on every solid layer holding `name` (or on
    `layers`), by the rule of the module docstring: the strengths constant,
    the times c_k eta / mu, the viscosity removed."""
    which = ([lay.index for lay in model.layers if name in lay
              and not is_fluid(lay)] if layers is None
             else sorted({model.layer(k).index for k in layers}))
    if not which:
        raise ValueError(f"no solid layer holds {name!r} to build branches from")
    fields = [dict(lay.fields) for lay in model.layers]
    for i in which:
        lay = model.layer(i)
        if name not in lay:
            raise KeyError(f"layer {i} ({lay.name}) holds no {name!r}")
        if is_fluid(lay):
            raise ValueError(f"layer {i} ({lay.name}) is fluid: it has no shear to "
                             "relax")
        eta = lay[name]
        _, mu = kappa_mu(lay)
        for k, (w, c) in enumerate(zip(series.strengths, series.times), start=1):
            fields[i][f"relaxation_strength_{k}"] = constant_field(
                lay.interval, w, name=f"relaxation_strength_{k}")
            fields[i][f"relaxation_time_{k}"] = ComposedField(
                lambda e, m, c=c: c * np.asarray(e) / np.asarray(m), [eta, mu],
                character=SCALAR, name=f"relaxation_time_{k}")
        del fields[i][name]
    return model.replaced(layers=fields).with_citation(*series.citation)

