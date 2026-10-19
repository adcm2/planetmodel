"""Conversions: seismic fields into thermal and rheological ones.

The laws that turn a model's seismic structure into what a solver of
slow deformation reads (a temperature anomaly from a velocity anomaly, a
density from either, a viscosity from them) are written on numbers in
SI (`laws`), each a short function a reader can hold against its
source; the parameters that vary with radius are profiles (`profiles`);
and a law's values travel with their sources as a parameter set
(`parameters`), so that whatever is built with one cites it.  Nothing
here is particular to the Earth.

Reached through its modules, off the top-level surface, with numpy and
scipy alone:

    from planetmodel.conversion import laws, parameters
    dt = laws.temperature_from_density(
        laws.density_from_velocity(dlnvs, scaling=parameters.SL3D_DENSITY_SCALING
                                   .scaling(r)),
        expansivity=parameters.SL3D_EXPANSIVITY.expansivity(r))
"""
from . import laws, parameters, profiles
from .laws import (GAS_CONSTANT, anelastic_sensitivity, beta_viscosity,
                   density_from_temperature, density_from_velocity,
                   exponential_viscosity, temperature_from_density,
                   temperature_from_velocity)
from .parameters import (BetaViscosity, DensityScaling, Expansivity,
                         ExponentialViscosity, ParameterSet, Sensitivity)
from .profiles import (Profile, ProfileLike, as_profile, constant, polynomial, spline,
                       tabulated)

__all__ = ["laws", "parameters", "profiles", "GAS_CONSTANT", "anelastic_sensitivity",
           "beta_viscosity", "density_from_temperature", "density_from_velocity",
           "exponential_viscosity", "temperature_from_density",
           "temperature_from_velocity", "BetaViscosity", "DensityScaling",
           "Expansivity", "ExponentialViscosity", "ParameterSet", "Sensitivity",
           "Profile", "ProfileLike", "as_profile", "constant", "polynomial", "spline",
           "tabulated"]
