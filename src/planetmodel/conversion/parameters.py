"""Parameter sets: a law's values with their sources.

A parameter set is a frozen dataclass of profiles (`profiles`) in SI,
with a `name` and the works it rests on (`citation`, gathered as a
model's are, `model.cited`), so that a model built with it cites them.
One class per law:

- `Sensitivity`: d ln Vs / dT (1/K), for `laws.temperature_from_velocity`;
- `DensityScaling`: d ln rho / d ln Vs, for `laws.density_from_velocity`;
- `Expansivity`: alpha (1/K), for `laws.density_from_temperature` and
  its inverse;
- `ExponentialViscosity`: eps (1/K), for `laws.exponential_viscosity`;
- `BetaViscosity`: beta, a = (E* + p V*) / R (K), T_0 (K) and the
  sensitivity, for `laws.beta_viscosity`.

The shipped sets are those read in their sources:

- sl3d's chain, as its Fortran states it and Crawford's thesis (2019,
  sec. 4.6.4) describes it after Latychev et al. (2005):
  `SL3D_DENSITY_SCALING`, the natural spline through 0.1, 0.25, 0.337,
  0.337, 0.25, 0.1 at radii 3870, 4524, 5043, 5537, 5870, 6203 km, which
  the thesis attributes to Forte & Woodward (1997);
  `SL3D_EXPANSIVITY`, 2.5e-5 (a0 + a1 t + a2 t^2 + a3 t^3) with t = r /
  2891 km, 1.0e-5 at the CMB and 2.5e-5 at 670 km, attributed to
  Chopelas & Boehler (1992); `SL3D_VISCOSITY`, eps = 0.01 per K;
- `MILNE_2018_VISCOSITY`, the same law with eps = 0.04 per K (Milne et
  al. 2018 eq. 3; Yousefi et al. 2021 use it too);
- `PAULSON_2005_DENSITY_SCALING`, 0.3 after Karato (1993), which Paulson
  et al. (2005) apply below 400 km, and `PAULSON_2005_EXPANSIVITY`,
  alpha = [r' + 3 (1 - r')] 1e-5 with r' the depth over the CMB's (taken
  2891 km), 3e-5 at the surface and 1e-5 at the CMB;
- `MITROVICA_2011_DENSITY_SCALING`, 0.4 after Bolton (1996);
- `GPLSPEC_DENSITY_SCALING`, 0.5, gplspec's, whose basis is not stated
  and which exceeds every published value (0.1 to 0.4);
- Ivins & Sammis's (1995) Table 1, at depths of 350 to 2835 km taken
  below a radius of 6371 km, linear between them and held beyond:
  `IVINS_SAMMIS_1995_EXPANSIVITY`, and `ivins_sammis_beta(beta,
  sensitivity, *, convection)`, the beta law of Wu, Wang & Steffen (2013)
  with the table's a = (E* + P V*) / R and its background temperature,
  of single-layer convection (`"whole"`, after Leitch & Yuen 1989, the
  one Wu et al. use) or of intermittently layered convection
  (`"layered"`, after Tackley et al. 1993); beta and d ln Vs / dT are
  the caller's, the table giving neither (`CONVECTIONS`).

The regime of each, purely thermal anomalies, is its docstring's, not a
check.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import KW_ONLY, dataclass, fields

import numpy as np

from ..model import cited
from .profiles import (Profile, ProfileLike, as_profile, constant, polynomial, spline,
                       tabulated)

__all__ = ["ParameterSet", "Sensitivity", "DensityScaling", "Expansivity",
           "ExponentialViscosity", "BetaViscosity", "SL3D_DENSITY_SCALING",
           "SL3D_EXPANSIVITY", "SL3D_VISCOSITY", "MILNE_2018_VISCOSITY",
           "PAULSON_2005_DENSITY_SCALING", "PAULSON_2005_EXPANSIVITY",
           "MITROVICA_2011_DENSITY_SCALING", "GPLSPEC_DENSITY_SCALING",
           "IVINS_SAMMIS_1995_EXPANSIVITY", "ivins_sammis_beta", "CONVECTIONS"]


@dataclass(frozen=True)
class ParameterSet:
    """What every parameter set has: its `name` and its `citation`; the
    profiles are its subclass's fields, numbers made constant profiles."""

    _: KW_ONLY
    name: str
    citation: Iterable[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "citation", cited((), self.citation))
        for f in fields(self):
            if f.name not in ("name", "citation", "_"):
                object.__setattr__(self, f.name, as_profile(getattr(self, f.name)))

    def cite(self) -> str:
        """The works the set rests on, one a line."""
        return "\n".join(self.citation)


@dataclass(frozen=True)
class Sensitivity(ParameterSet):
    """d ln Vs / dT in 1/K, negative."""

    sensitivity: ProfileLike


@dataclass(frozen=True)
class DensityScaling(ParameterSet):
    """d ln rho / d ln Vs."""

    scaling: ProfileLike


@dataclass(frozen=True)
class Expansivity(ParameterSet):
    """The volumetric thermal expansivity alpha in 1/K."""

    expansivity: ProfileLike


@dataclass(frozen=True)
class ExponentialViscosity(ParameterSet):
    """eps in dln eta = -eps dT, in 1/K."""

    epsilon: ProfileLike


@dataclass(frozen=True)
class BetaViscosity(ParameterSet):
    """The scaling of Wu, Wang & Steffen (2013): `beta` the thermal fraction,
    `activation` a = (E* + p V*) / R in K, `background` T_0 in K and
    `sensitivity` d ln Vs / dT in 1/K."""

    beta: ProfileLike
    activation: ProfileLike
    background: ProfileLike
    sensitivity: ProfileLike


_CRAWFORD = ("Crawford, O. (2019).  On the viscoelastic deformation of the Earth.  "
             "PhD thesis, University of Cambridge, doi:10.17863/CAM.34835")
_LATYCHEV = ("Latychev, K., J. X. Mitrovica, J. Tromp, M. E. Tamisiea, D. Komatitsch "
             "and C. C. Christara (2005).  Glacial isostatic adjustment on 3-D Earth "
             "models: a finite-volume formulation.  Geophys. J. Int. 161, 421-444, "
             "doi:10.1111/j.1365-246X.2005.02536.x")
_FORTE = ("Forte, A. M. and R. L. Woodward (1997).  Seismic-geodynamic constraints on "
          "three-dimensional structure, vertical flow, and heat transfer in the "
          "mantle.  J. Geophys. Res. 102, 17981-17994, doi:10.1029/97JB01276")
_CHOPELAS = ("Chopelas, A. and R. Boehler (1992).  Thermal expansivity in the lower "
             "mantle.  Geophys. Res. Lett. 19, 1983-1986, doi:10.1029/92GL02144")
_MILNE = ("Milne, G. A., K. Latychev, A. Schaeffer, J. W. Crowley, B. S. Lecavalier "
          "and A. Audette (2018).  The influence of lateral Earth structure on "
          "glacial isostatic adjustment in Greenland.  Geophys. J. Int. 214, "
          "1252-1266, doi:10.1093/gji/ggy189")
_PAULSON = ("Paulson, A., S. Zhong and J. Wahr (2005).  Modelling post-glacial rebound "
            "with lateral viscosity variations.  Geophys. J. Int. 163, 357-371, "
            "doi:10.1111/j.1365-246X.2005.02645.x")
_KARATO = ("Karato, S. (1993).  Importance of anelasticity in the interpretation of "
           "seismic tomography.  Geophys. Res. Lett. 20, 1623-1626, "
           "doi:10.1029/93GL01767")
_MITROVICA = ("Mitrovica, J. X., N. Gomez, E. Morrow, C. Hay, K. Latychev and M. E. "
              "Tamisiea (2011).  On the robustness of predictions of sea level "
              "fingerprints.  Geophys. J. Int. 187, 729-742, "
              "doi:10.1111/j.1365-246X.2011.05090.x")

#: sl3d's d ln rho / d ln Vs: a natural spline through six knots.
SL3D_DENSITY_SCALING = DensityScaling(
    spline([3870e3, 4524e3, 5043e3, 5537e3, 5870e3, 6203e3],
           [0.1, 0.25, 0.337, 0.337, 0.25, 0.1]),
    name="sl3d", citation=(_CRAWFORD, _LATYCHEV, _FORTE))

#: sl3d's expansivity: a cubic in r / 2891 km times 2.5e-5 per K.
SL3D_EXPANSIVITY = Expansivity(
    polynomial([2.5e-5 * c for c in (-1.04953003, 2.86975098, -2.09738922,
                                     0.59281731)], scale=2891e3),
    name="sl3d", citation=(_CRAWFORD, _CHOPELAS))

#: sl3d's exponential law, eps = 0.01 per K.
SL3D_VISCOSITY = ExponentialViscosity(0.01, name="sl3d", citation=(_CRAWFORD,
                                                                   _LATYCHEV))

#: The exponential law with eps = 0.04 per K.
MILNE_2018_VISCOSITY = ExponentialViscosity(0.04, name="Milne et al. 2018",
                                            citation=(_MILNE,))

#: d ln rho / d ln Vs = 0.3, after Karato (1993).
PAULSON_2005_DENSITY_SCALING = DensityScaling(0.3, name="Paulson et al. 2005",
                                              citation=(_PAULSON, _KARATO))

#: alpha linear in depth, 3e-5 per K at the surface to 1e-5 at the CMB.
PAULSON_2005_EXPANSIVITY = Expansivity(
    polynomial([1e-5 * (3.0 - 2.0 * 6371e3 / 2891e3), 2e-5], scale=2891e3),
    name="Paulson et al. 2005", citation=(_PAULSON,))

#: d ln rho / d ln Vs = 0.4.
MITROVICA_2011_DENSITY_SCALING = DensityScaling(
    0.4, name="Mitrovica et al. 2011", citation=(_MITROVICA,))

#: gplspec's d ln rho / d ln Vs = 0.5, its basis not stated.
GPLSPEC_DENSITY_SCALING = DensityScaling(constant(0.5), name="gplspec")

_IVINS = ("Ivins, E. R. and C. G. Sammis (1995).  On lateral viscosity contrast in the "
          "mantle and the rheology of low-frequency geodynamics.  Geophys. J. Int. "
          "123, 305-322, doi:10.1111/j.1365-246X.1995.tb06856.x")
_WU = ("Wu, P., H. Wang and H. Steffen (2013).  The role of thermal effect on mantle "
       "seismic anomalies under Laurentia and Fennoscandia from observations of "
       "Glacial Isostatic Adjustment.  Geophys. J. Int. 192, 7-17, "
       "doi:10.1093/gji/ggs009")

#: Ivins & Sammis (1995) Table 1: depth (km), a (1e4 K), T0 of single-layer
#: and of layered convection (1e3 K), alpha (1e-5 / K).
_TABLE1 = {
    "depth": (350.0, 450.0, 550.0, 700.0, 900.0, 1200.0, 1500.0, 1800.0, 2200.0,
              2600.0, 2835.0),
    "activation": (8.2, 8.3, 8.5, 8.3, 8.7, 9.35, 9.97, 10.6, 11.0, 11.4, 11.7),
    "whole": (1.99, 2.1, 2.2, 2.23, 2.275, 2.296, 2.47, 2.53, 2.562, 2.53, 2.675),
    "layered": (2.088, 2.069, 2.1, 2.3, 2.5, 2.6, 2.65, 2.7, 2.675, 2.75, 3.25),
    "expansivity": (1.725, 1.65, 1.5, 1.463, 1.425, 1.3, 1.177, 1.05, 0.9, 0.75,
                    0.65),
}

#: The background temperatures of Ivins & Sammis's Table 1, by the
#: convection they come from.
CONVECTIONS = ("whole", "layered")


def _by_depth(values: Sequence[float], unit: float) -> Profile:
    """A column of Table 1 as a profile of radius, its depths below 6371 km."""
    radii = 6371e3 - 1e3 * np.asarray(_TABLE1["depth"])[::-1]
    return tabulated(radii, unit * np.asarray(values)[::-1])


#: alpha of Ivins & Sammis's Table 1, 1.725e-5 per K at 350 km to 0.65e-5 at
#: 2835 km.
IVINS_SAMMIS_1995_EXPANSIVITY = Expansivity(_by_depth(_TABLE1["expansivity"], 1e-5),
                                            name="Ivins & Sammis 1995",
                                            citation=(_IVINS,))


def ivins_sammis_beta(beta: ProfileLike, sensitivity: ProfileLike, *,
                      convection: str = "whole") -> BetaViscosity:
    """The beta law with Ivins & Sammis's a and background temperature of
    `convection` (one of `CONVECTIONS`), `beta` and `sensitivity` (d ln Vs
    / dT, 1/K) the caller's, as the module docstring says."""
    if convection not in CONVECTIONS:
        raise ValueError(f"convection must be one of {CONVECTIONS}, got "
                         f"{convection!r}")
    return BetaViscosity(beta, _by_depth(_TABLE1["activation"], 1e4), _by_depth(
        _TABLE1[convection], 1e3), sensitivity,
        name=f"Wu et al. 2013 with Ivins & Sammis 1995 ({convection})",
        citation=(_WU, _IVINS))

