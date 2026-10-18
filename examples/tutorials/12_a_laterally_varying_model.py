# %% [markdown]
# # 12. A model that varies with direction
#
# The models of the earlier tutorials are spherically symmetric: every
# field depends on the radius alone. A *laterally varying* model has
# fields that also depend on the colatitude and longitude. A common way
# to state one is as a one-dimensional *reference*, a radial profile,
# together with a *perturbation*, a field of all three coordinates that
# says how far the model departs from the reference at each point.
# Seismic tomography is published this way, as a relative perturbation
# of a reference velocity, and so is a viscosity for glacial isostatic
# adjustment, as the logarithm of its ratio to a radial profile.
#
# This tutorial builds such a model from PREM, writes it to a *model
# file*, reads it back, compares the two, and draws a map of the model
# at one depth. A model file is one netCDF file holding a model's
# skeleton, its fields at the nodes of a radial layout and an angular
# grid, and its mapping. Python users read it with the library, and the
# C++ codes that solve on a model read the same file. Its format is
# described on the "model file" page of the documentation.
#
# The tutorial needs the `netcdf` extra (netCDF4) for the file and the
# `plot` extra (matplotlib) for the figure, which is written to
# `examples/figures/`.

# %%
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from planetmodel import (PREM, HarmonicField, PerturbedField, constant_field,
                         gauss_legendre, perturbed, spherical_mean)
from planetmodel.harmonics import packing
from planetmodel.modelfile import read_model, read_model_file, write_model

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## A reference with a viscosity
#
# PREM has no viscosity, so one is added: a radial profile of 1e22 Pa s
# in the lower mantle and 1e21 Pa s in the upper mantle, on the solid
# layers between the core and the lithosphere. A field of the radius
# alone is a reference like any other.

# %%
prem = PREM()
lower = ("lowermost_mantle", "lower_mantle", "upper_lower_mantle")
upper = ("transition_zone_lower", "transition_zone_middle",
         "transition_zone_upper", "low_velocity_zone")
model = prem
for name in lower + upper:
    layer = prem.layer(name)
    eta = 1e22 if name in lower else 1e21
    model = model.with_field(layer.index, "viscosity",
                             constant_field(layer.interval, eta))
print(model)

# %% [markdown]
# ## Two perturbations over the lower mantle
#
# Both perturbations are given as spherical-harmonic expansions whose
# coefficients are functions of radius, a `HarmonicField`. The
# expansion uses the real orthonormal harmonics of
# `planetmodel.harmonics`, up to degree 8 here, and each coefficient is
# a cubic spline in radius that vanishes at the core-mantle boundary
# and at 670 km depth, so that the perturbation fades out at the ends of
# the lower mantle. The pattern is a sum of low-degree harmonics, the
# kind of large-scale structure the lower mantle has.
#
# The velocity perturbation is *relative*: the perturbed velocity is
# `vsv * (1 + delta)`, with `delta` within about a percent either way.
# The viscosity perturbation is *logarithmic*: the perturbed viscosity
# is `viscosity * exp(delta)`, with `delta` within about 0.45 either
# way, so that across the lower mantle the viscosity varies by a factor
# of about two and a half. A log perturbation keeps the viscosity
# positive whatever the pattern.
#
# `perturbed` wraps the named field on every layer the perturbation's
# interval covers. On PREM, which is an `Elastic` model, perturbing
# `vsv` also re-derives the Love modulus L from the perturbed velocity.

# %%
lo, hi = 3480e3, 5701e3
L = 8
s, l, m = packing(L)
rng = np.random.default_rng(12)
knots = np.linspace(lo, hi, 5)
taper = np.array([0.0, 0.7, 1.0, 0.7, 0.0])


def expansion(amplitude, decay):
    """Coefficients at the knots: a random large-scale pattern, tapered."""
    pattern = np.zeros((2, L + 1, L + 1))
    pattern[s, l, m] = rng.standard_normal(s.size) * amplitude / (1.0 + l) ** decay
    pattern[0, 0, 0] = 0.0                     # no change to the spherical mean
    return CubicSpline(knots, taper[:, None, None, None] * pattern, axis=0)


dlnvs = HarmonicField((lo, hi), expansion(0.01, 1.0))
dlneta = HarmonicField((lo, hi), expansion(0.8, 1.5))
model = perturbed(model, "vsv", dlnvs, kind="relative")
model = perturbed(model, "viscosity", dlneta, kind="log")
vsv = model.layer("lower_mantle")["vsv"]
modulus = model.layer("lower_mantle")["L"]
print(type(vsv).__name__, vsv.kind, "| L is now", type(modulus).__name__)

r, t, p = 4.5e6, 1.2, 2.0
print(f"vsv at r = 4500 km: reference {vsv.reference(r):.1f} m/s, "
      f"perturbed {vsv(r, t, p):.1f} m/s")

# %% [markdown]
# The *spherical mean* of a field is its average over the sphere at each
# radius. For a relative perturbation whose own mean is zero the mean of
# the field is its reference. For a log perturbation it is not: the
# average of `exp(delta)` is more than one, so the mean viscosity is
# above the reference profile even though the mean of `delta` is zero.
# The library keeps the two apart; here the mean is taken by quadrature
# on a Gauss-Legendre grid.

# %%
eta = model.layer("lower_mantle")["viscosity"]
mean = spherical_mean(eta, grid=gauss_legendre(24))
print(f"viscosity at r = 4500 km: reference {eta.reference(r):.3g} Pa s, "
      f"spherical mean {mean(r):.3g} Pa s")

# %% [markdown]
# ## Writing the model file
#
# `write_model` takes the angular grid the fields are stored on. A grid
# of band `lmax` resolves spherical harmonics up to that degree; this
# one has band 32, four times the perturbations' band of 8. The extra
# resolution is what a local interpolation rule needs: between the nodes
# a file states how each field is to be read, and here the bicubic rule
# is asked for, whose error on a grid four times the band is a few parts
# in ten thousand of the field. A reader with spherical transforms could
# use the harmonic rule on the same file instead.
#
# The radial layout is exact: the writer puts element boundaries at the
# breakpoints of PREM's polynomials and of the coefficient splines, so
# the reference profiles come back as the same polynomials. Only the
# base fields are written; a model type re-derives the rest.

# %%
workdir = Path(tempfile.mkdtemp(prefix="planetmodel_"))
names = ["rho", "vpv", "vsv", "vph", "vsh", "eta", "viscosity"]
path = write_model(workdir / "prem_3d.nc", model, grid=gauss_legendre(32),
                   fields=names,
                   interpolation={"vsv": ("polynomial", "cubic"),
                                  "viscosity": ("polynomial", "cubic")},
                   meta={"made_by": "tutorial 12"})
print(path.name, f"{path.stat().st_size / 1e6:.1f} MB")

data = read_model_file(path)
part = data.parts[0]
print("parts:", len(data.parts), "| grid:", part.grid, "| radial layout:", part.mesh)
print("variables:", sorted(part.variables))

# %% [markdown]
# ## Reading it back
#
# `read_model` gives a plain `Model`. A field that was a reference with
# a perturbation comes back as one, its reference an exact polynomial
# and its perturbation a `GriddedField`, the values on the grid read by
# the stated rule. At the file's own nodes the values are the original
# model's exactly; between the nodes they differ by the interpolation's
# error.

# %%
back = read_model(path)
f = back.layer("lower_mantle")["vsv"]
print(type(f).__name__, f.kind, "| perturbation:", type(f.perturbation).__name__,
      f.perturbation.lateral)

rr = rng.uniform(3.7e6, 5.6e6, 2000)
tt, pp = rng.uniform(0, np.pi, 2000), rng.uniform(0, 2 * np.pi, 2000)
for name in ("vsv", "viscosity"):
    a = model.layer("lower_mantle")[name]
    b = back.layer("lower_mantle")[name]
    rel = np.abs(b(rr, tt, pp) / a(rr, tt, pp) - 1.0).max()
    print(f"{name}: largest relative difference at 2000 random points {rel:.1e}")
ref = back.layer("lower_mantle")["vsv"].reference
print("reference vsv exact:",
      np.allclose(ref(rr), vsv.reference(rr), rtol=1e-12, atol=0))
assert isinstance(f, PerturbedField)

# %% [markdown]
# ## A map at one depth
#
# The map shows the model read back from the file, 2000 km below the
# surface, on a grid of latitudes and longitudes in degrees: the shear
# velocity as a percentage departure from the reference, and the
# viscosity on a logarithmic scale. The map is drawn with matplotlib
# alone, latitude against longitude, with no projection or coastlines.

# %%
import matplotlib.pyplot as plt  # noqa: E402

depth = 2000e3
radius = 6371e3 - depth
lat = np.linspace(-89.5, 89.5, 180)
lon = np.linspace(0.5, 359.5, 360)
theta = np.deg2rad(90.0 - lat)[:, None]
phi = np.deg2rad(lon)[None, :]
layer = back.layer("lower_mantle")
dvs = 100.0 * (layer["vsv"](radius, theta, phi) / layer["vsv"].reference(radius) - 1.0)
log_eta = np.log10(layer["viscosity"](radius, theta, phi))

fig, axes = plt.subplots(2, 1, figsize=(8, 7.5))
for ax, values, label, cmap in ((axes[0], dvs, "vsv departure [%]", "RdBu"),
                                (axes[1], log_eta, "log10 viscosity [Pa s]",
                                 "viridis")):
    vmax = np.abs(values).max() if cmap == "RdBu" else None
    im = ax.pcolormesh(lon, lat, values, shading="auto", cmap=cmap,
                       vmin=-vmax if vmax else None, vmax=vmax)
    ax.set_xlabel("longitude [degrees]")
    ax.set_ylabel("latitude [degrees]")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
axes[0].set_title(f"the model read back, {depth / 1e3:.0f} km deep")
fig.tight_layout()
out = FIGURES / "tutorial_12_a_laterally_varying_model.png"
fig.savefig(out, dpi=100)
print("figure written to", out)
