"""The pointwise laws, on numbers in SI.

Each law is a function of numpy arrays, its parameters keyword-only and
already evaluated at the points (a profile is evaluated by whoever
applies the law, `planetmodel.conversion.profiles`); every relative
quantity is a fraction, every logarithm natural. With a seismic anomaly
taken as purely thermal:

- the temperature anomaly from a relative velocity anomaly and the
  sensitivity s = d ln Vs / dT (1/K, negative), dT = dlnVs / s
  (`temperature_from_velocity`), and the sensitivity's anelastic part
  after Karato (1993), as Steinberger & Calderwood (2006) eq. (21) write
  it, d ln Vs / dT = (d ln Vs / dT)_anharmonic - F(a) Q^-1 H / (pi R T^2)
  with F(a) = (pi a / 2) cot(pi a / 2), Q the shear quality factor, a its
  frequency exponent (Q ~ omega^a) and H the activation enthalpy
  (`anelastic_sensitivity`);
- density from velocity, dln rho = R dlnVs with R = d ln rho / d ln Vs
  (`density_from_velocity`), and from temperature, dln rho = -alpha dT,
  alpha the volumetric expansivity (`density_from_temperature`), with
  its inverse dT = -dln rho / alpha (`temperature_from_density`);
- viscosity, as the natural logarithm of its ratio to the radial
  profile: the exponential law dln eta = -eps dT
  (`exponential_viscosity`), and the scaling of Wu, Wang & Steffen
  (2013), log10 (eta / eta_0) = -0.4343 beta (E* + p V*) / (R T_0^2 s)
  dlnVs, here dln eta = -beta a dlnVs / (T_0^2 s) with a = (E* + p V*) /
  R in kelvin, T_0 the background temperature and beta the fraction of
  the anomaly that is thermal (`beta_viscosity`).

A fast anomaly is cold, dense and stiff under every law here.
`GAS_CONSTANT` is the molar gas constant, exact in the SI since 2019.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["GAS_CONSTANT", "temperature_from_velocity", "anelastic_sensitivity",
           "density_from_velocity", "density_from_temperature",
           "temperature_from_density", "exponential_viscosity", "beta_viscosity"]

#: The molar gas constant, J / (mol K): the Avogadro constant times the
#: Boltzmann constant, both exact.
GAS_CONSTANT = 6.02214076e23 * 1.380649e-23


def _nonzero(x: ArrayLike, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if np.any(a == 0.0) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite and nowhere zero")
    return a


def temperature_from_velocity(dlnvs: ArrayLike, *, sensitivity: ArrayLike
                              ) -> np.ndarray:
    """dT = dlnVs / s, s = d ln Vs / dT in 1/K; in K."""
    return np.asarray(dlnvs, dtype=float) / _nonzero(sensitivity, "the sensitivity")


def anelastic_sensitivity(temperature: ArrayLike, *, q: ArrayLike,
                          enthalpy: ArrayLike, exponent: ArrayLike) -> np.ndarray:
    """The anelastic part of d ln Vs / dT, -F(a) Q^-1 H / (pi R T^2) with
    F(a) = (pi a / 2) cot(pi a / 2), in 1/K: `temperature` T in K, `q` the
    shear quality factor, `enthalpy` H in J/mol, `exponent` a in (0, 1)."""
    t = _nonzero(temperature, "the temperature")
    a = np.asarray(exponent, dtype=float)
    if np.any((a <= 0.0) | (a >= 1.0)):
        raise ValueError("the frequency exponent of Q lies in (0, 1)")
    f = 0.5 * np.pi * a / np.tan(0.5 * np.pi * a)
    return (-f * np.asarray(enthalpy, dtype=float)
            / (np.pi * _nonzero(q, "Q") * GAS_CONSTANT * t * t))


def density_from_velocity(dlnvs: ArrayLike, *, scaling: ArrayLike) -> np.ndarray:
    """dln rho = R dlnVs, R = d ln rho / d ln Vs."""
    return np.asarray(scaling, dtype=float) * np.asarray(dlnvs, dtype=float)


def density_from_temperature(dt: ArrayLike, *, expansivity: ArrayLike) -> np.ndarray:
    """dln rho = -alpha dT, alpha in 1/K, dT in K."""
    return -np.asarray(expansivity, dtype=float) * np.asarray(dt, dtype=float)


def temperature_from_density(dlnrho: ArrayLike, *, expansivity: ArrayLike
                             ) -> np.ndarray:
    """dT = -dln rho / alpha, in K."""
    return -np.asarray(dlnrho, dtype=float) / _nonzero(expansivity, "the expansivity")


def exponential_viscosity(dt: ArrayLike, *, epsilon: ArrayLike) -> np.ndarray:
    """dln eta = -eps dT, eps in 1/K."""
    return -np.asarray(epsilon, dtype=float) * np.asarray(dt, dtype=float)


def beta_viscosity(dlnvs: ArrayLike, *, beta: ArrayLike, activation: ArrayLike,
                   background: ArrayLike, sensitivity: ArrayLike) -> np.ndarray:
    """dln eta = -beta a dlnVs / (T_0^2 s): `activation` a = (E* + p V*) / R
    in K, `background` T_0 in K, `sensitivity` s = d ln Vs / dT in 1/K,
    `beta` the thermal fraction of the anomaly."""
    t0 = _nonzero(background, "the background temperature")
    return (-np.asarray(beta, dtype=float) * np.asarray(activation, dtype=float)
            * np.asarray(dlnvs, dtype=float)
            / (t0 * t0 * _nonzero(sensitivity, "the sensitivity")))
