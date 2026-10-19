# %% [markdown]
# # 14. Published models
#
# Most models of the Earth's interior are published by their authors as
# files: tables of coefficients, grids of values, tables of a radial
# model. `planetmodel.datasets` knows where a set of them are kept, fetches
# them when they are first asked for, checks that each file is the one its
# entry records, and reads it into the library's own objects, so that a
# published model can be looked at, compared with another, and put onto a
# model of your own.
#
# This tutorial goes through those steps with real models: the
# shear-velocity model S40RTS, the model S362ANI+M given as velocities on
# a grid, a regional model of the North Atlantic, and a model of the Moon.
# It needs the network the first time it runs (the files are a few
# megabytes, and are kept in a cache afterwards), the `netcdf` extra
# (netCDF4) for the gridded models, the `maps` extra (cartopy) for the
# maps, and the `harmonics` extra (ducc0) for the spectra. The figures are
# written to `examples/figures/`.

# %%
import tempfile
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from planetmodel import PREM, gauss_legendre, equiangular, from_geographic, slices
from planetmodel import datasets
from planetmodel.modelfile import read_model, write_model
from planetmodel.plotting import correlation, profile, section, spectrum, sphere_map

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## The registry
#
# A *data set* is a published model the library knows about. Its *entry*
# says where each of its files is served, the file's checksum (a
# fingerprint that changes if a single byte of the file does), how the
# file is read, which model its values are relative to, and how its
# authors ask to be cited, with its licence in their own words. The
# library holds the entries, never the data: the files come from their
# authors' own hosts. `REGISTRY` holds the shipped entries by name.

# %%
print(sorted(datasets.REGISTRY))
print()
print(datasets.describe("S40RTS"))

# %% [markdown]
# `fetch` brings a data set's files into a cache, a directory on your
# machine (`PLANETMODEL_DATA` if you set it, else your platform's usual
# place for caches), and gives their paths. It downloads a file only when
# it is not already there, and checks its checksum every time; a file
# that does not match is refused. `cache_directory()` says where the cache
# is.

# %%
print(datasets.fetch("S40RTS"))
print("the cache:", datasets.cache_directory())

# %% [markdown]
# ## A data set as the model it is
#
# S40RTS (Ritsema and others, 2011) is a model of the shear velocity of
# the mantle, published as coefficients of spherical harmonics up to
# degree 40 on 21 radial splines between the core-mantle boundary and the
# Moho. What it describes is a *relative perturbation*: how much faster
# or slower than PREM the shear velocity is at each point, as a fraction
# of PREM's velocity there.
#
# `load` reads it as a model holding exactly that: a model on PREM's
# layers from the core-mantle boundary to the surface, whose field `vs` on
# each mantle layer is PREM's shear velocity times one plus the
# perturbation, and whose crust holds nothing. (PREM's shear velocity in
# its upper mantle differs between the two polarisations; `vs` is their
# Voigt average.) The perturbation is held exactly, as the harmonic
# expansion the file is. Everything the library does with a model works
# on it as it stands: here a map of the perturbation 1000 km deep, as
# its authors draw it.

# %%
s40 = datasets.load("S40RTS")
print(s40)
deep = slices.sphere(s40, "vs", depth=1000e3, relative_to="reference")
print(deep.label)

fig = plt.figure(figsize=(12, 4.5))
ax = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
sphere_map(ax, deep, colorbar=True, coastlines=True)
ax.set_title("S40RTS, vs 1000 km deep")
across = slices.section(s40, "vs", from_geographic(0.0, -60.0), azimuth=0.5 * np.pi,
                        length=np.pi, radii=(3480e3, 6346.6e3), relative_to="reference")
ax = fig.add_subplot(1, 2, 2, projection="polar")
section(ax, across, scale=1e-3, colorbar=True)
ax.set_rticks([4000, 5000, 6000])
ax.set_title("the mantle along the equator, from 60 degrees west")
fig.tight_layout()
out = FIGURES / "tutorial_14_s40rts.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## A published model on a model of your own
#
# `transplanted` puts a data set's perturbation onto a model of your own.
# The perturbation is kept as a percentage (`mode="relative"`, the usual
# practice), so that your model's own radial structure stays as it is.
# Because PREM holds two shear velocities, `vsv` and `vsh`, and S40RTS
# perturbs one isotropic `vs`, the perturbation goes onto both, as
# SPECFEM applies S40RTS to PREM. PREM's model type then recomputes its
# elastic moduli from the perturbed velocities.
#
# `lmax=` keeps the perturbation's degrees up to a band; `roll_off=`
# brings the highest of them down smoothly rather than cutting them off.
# The spectra show what that does: the power of the perturbation at each
# degree, at one depth, before and after. Above degree 12 the filtered
# perturbation has no power at all, so its curve leaves the bottom of the
# picture.

# %%
prem = PREM()
on_prem = datasets.transplanted(prem, "S40RTS")
smooth = datasets.transplanted(prem, "S40RTS", lmax=12, roll_off=4)
lay = on_prem.layer("lower_mantle")
r, theta, phi = 5.0e6, 1.0, 2.0
L = lay["rho"](r) * lay["vsv"](r, theta, phi) ** 2
again = np.allclose(lay["L"](r, theta, phi), L)
print("vsv on the lower mantle:", type(lay["vsv"]).__name__,
      "| L there is rho vsv^2 again:", again)

fig, ax = plt.subplots(figsize=(7, 4))
for model, label in ((on_prem, "degree 40"), (smooth, "to degree 12, rolled off")):
    sp = slices.spectrum(model, "vsv", lmax=40, depth=1000e3, relative_to="reference")
    spectrum(ax, sp, label=label)
ax.set_ylim(1e-7, 1e-1)                 # above degree 12 the filtered power is zero
ax.set_xlabel("degree")
ax.set_ylabel("power")
ax.legend()
ax.set_title("the perturbation of vsv 1000 km deep, by degree")
fig.tight_layout()
out = FIGURES / "tutorial_14_spectra.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## A model given as velocities
#
# S362ANI+M (Moulik and Ekström, 2014) is offered by the Earth Model
# Collaboration as velocities in km/s on a grid of latitude, longitude
# and depth. Read as it is, it is a model of values, not of
# perturbations. Its authors' reference is the radial model STW105, which
# the registry also holds; `transplanted` makes the velocities relative to
# STW105 first, and carries that percentage onto PREM.
#
# The grid runs from 25 to 2890 km deep, so it reaches neither PREM's
# Moho (24.4 km) nor its core-mantle boundary (2891 km), and covers two of
# PREM's layers only in part. Rather than guess, the library refuses, and
# says how to choose: `ends=("cmb", "moho")` stretches the model's depth
# range onto the span between two of PREM's interfaces, `extend=True`
# carries its end values out to the layers' ends, and `taper=` takes the
# perturbation to zero inside its ends.

# %%
try:
    datasets.transplanted(prem, "S362ANI+M")
except ValueError as exc:
    print("refused:", str(exc)[:240], "...")
s362 = datasets.transplanted(prem, "S362ANI+M", ends=("cmb", "moho"))

# %% [markdown]
# ## Comparing two models
#
# Two models of the same mantle agree more at some scales and depths than
# at others. The correlation of their perturbations by degree and depth
# shows where: near one where they agree, near zero where they are
# unrelated. The two models were made from different data by different
# methods. They agree best at the largest scales, and least at the smaller
# scales between about 5500 and 6100 km radius, in and around the
# transition zone.

# %%
compared = slices.correlation(on_prem, "vsv", s362, lmax=18,
                              depths=np.linspace(150e3, 2800e3, 25),
                              relative_to="reference")
fig, ax = plt.subplots(figsize=(7, 5))
correlation(ax, compared, scale=1e-3, colorbar=True)
ax.set_xlabel("degree")
ax.set_ylabel("radius [km]")
ax.set_title("S40RTS against S362ANI+M on PREM, vsv")
fig.tight_layout()
out = FIGURES / "tutorial_14_correlation.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## A regional model inside a global one
#
# A regional model covers part of the Earth in more detail. The Earth
# Model Collaboration's North Atlantic part of the Collaborative Seismic
# Earth Model covers latitudes 20 to 75 degrees north and longitudes 55
# degrees west to 10 east, down to 500 km. `load` reads it as a regional
# grid, which can be asked for its values inside its box. `merged` blends
# it into a global model: inside the box, the regional values; outside,
# the global ones; between them, over a band of the width asked for, a
# smooth change from one to the other, and again over 50 km at the base
# of the regional model. The result is held on the angular grid given.

# %%
atlantic = datasets.load("CSEM-North-Atlantic")
print("a regional grid of", atlantic.names)
blended = on_prem
for name in ("vsv", "vsh"):
    blended = datasets.merged(blended, atlantic, name=name, width=np.radians(5.0),
                              grid=equiangular(180, 360), radial_width=(50e3, 0.0))

fig = plt.figure(figsize=(12, 5))
view = ccrs.Orthographic(-18.0, 55.0)
for k, (model, title) in enumerate(((on_prem, "S40RTS on PREM"),
                                    (blended, "with the North Atlantic merged"))):
    ax = fig.add_subplot(1, 2, k + 1, projection=view)
    at = slices.sphere(model, "vsv", depth=150e3, relative_to="mean")
    sphere_map(ax, at, colorbar=True, coastlines=True, vmin=-6.0, vmax=6.0)
    ax.set_title(f"{title}: vsv 150 km deep")
fig.tight_layout()
out = FIGURES / "tutorial_14_merged.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## Writing the result
#
# A model built this way is written to a model file like any other, and
# read back; this is the file a code such as AdGIA or sl3dpp reads.
# Writing samples the fields on the angular grid given; a grid about four
# times finer than the highest degree held keeps the fields accurate
# between its nodes.

# %%
with tempfile.TemporaryDirectory() as tmp:
    path = write_model(Path(tmp) / "prem_s40rts.nc", smooth, grid=gauss_legendre(48),
                       fields=["rho", "vsv", "vsh"])
    back = read_model(path)
    written = float(smooth.layer("lower_mantle")["vsv"](r, theta, phi))
    read = float(back.layer("lower_mantle")["vsv"](r, theta, phi))
    print(f"vsv at one point: written {written:.3f}, read back {read:.3f} m/s")

# %% [markdown]
# ## Another body
#
# Nothing here is particular to the Earth. The registry holds models of
# the Moon from the compilation of Garcia and others (2019), in the
# format of the TauP toolkit, which lists depths with the velocities and
# density below them and names the boundaries. VPREMOON is one: a model
# 1737.1 km in radius, with a crust, a mantle and a liquid outer core.

# %%
moon = datasets.load("VPREMOON")
print(moon)
fig, ax = plt.subplots(figsize=(5, 6))
profile(ax, [lay["vp"] for lay in moon.layers], scale=1e-3, value_scale=1e-3,
        label="vp")
profile(ax, [lay["vs"] for lay in moon.layers], scale=1e-3, value_scale=1e-3,
        label="vs")
ax.set_xlabel("velocity [km/s]")
ax.set_ylabel("radius [km]")
ax.legend()
ax.set_title("VPREMOON")
fig.tight_layout()
out = FIGURES / "tutorial_14_moon.png"
fig.savefig(out, dpi=100)
print("figure written to", out)
