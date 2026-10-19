# %% [markdown]
# # 15. Sensitivity kernels
#
# A *sensitivity kernel* says how a number computed from a model changes
# when the model changes. The number is called the *objective*, J: a
# misfit between predictions and observations, say, or a single
# predicted travel time. For one parameter of the model, here the shear
# velocity, the kernel K is a field over the planet such that a small
# change dm of the parameter changes the objective by
#
#     dJ = the integral of K dm over the planet's volume.
#
# Codes such as AdGIA and sl3dpp compute kernels on meshes of their own.
# To look at one, the code samples it at points that planetmodel's C++
# library gives it and writes a *model file* (tutorial 12); the Python
# reads the file as a model whose fields are the kernels, so every
# picture of tutorial 13 applies to it as it stands. This tutorial shows
# the pictures that suit a kernel.
#
# Kernels look unlike models of the Earth in two ways. They change sign,
# so a colour scale should be centred on zero. And a kernel computed
# numerically is often very large close to where an observation was
# made, so that a picture scaled to its largest value shows the large
# value and little else. (In an inversion the derivative is usually
# turned into a model update by a smoothing, or *regularising*,
# operator, which tames such peaks; the pictures here are of the raw
# kernel, as a code writes it.)
#
# The file used here is the one `cpp/examples/write_kernel.cpp` writes:
# a synthetic kernel of the shear velocity in the mantle, with a broad
# pattern of both signs and a peak 200 km wide beside a receiver at the
# surface, at colatitude 60 degrees and longitude 30; and a kernel of a
# shift of the 670 km discontinuity, a field on that interface. Run as it
# stands, the tutorial writes the same file with the Python, so that it
# needs no C++ build. Given the path of the file the C++ example wrote,
# it reads that one instead and checks it against the Python's:
#
#     python 15_kernels.py ../../build/cpp/tests/written/kernel.nc
#
# It needs the `netcdf` extra (netCDF4) for the model file, the `maps`
# extra (cartopy) for the maps, and the `harmonics` extra (ducc0) for the
# spectra. The figures are written to `examples/figures/`, and the
# ParaView files to `examples/figures/tutorial_15_vtk/`.

# %%
import sys
import tempfile
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from planetmodel import (DENSITY, SCALAR, AnalyticField, Dimensions, FieldSpec,
                         Geometry, Model, RadialField, RadialMesh, Skeleton,
                         gauss_legendre, slices)
from planetmodel.modelfile import (read_model, read_model_file, read_surfaces,
                                   write_model)
from planetmodel.plotting import profile, section, spectrum, sphere_map, write_vtk

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## The file a code writes
#
# The code ran on a model of three layers: a core, and a mantle split at
# the 670 km discontinuity. The kernel file is on that model's
# *skeleton*, its boundary radii, and holds a density for reference
# beside the kernels. The functions below are the synthetic kernels, the
# same as the C++ example's.
#
# What the code chooses, the file records. The kernel is stored at the
# code's own radial nodes, every 50 km in the mantle, in elements of two
# nodes, so that it is read back as straight lines between them; and on
# a Gauss-Legendre grid of 65 by 129 points, about 2.8 degrees apart,
# read between its nodes by the bilinear rule. The bilinear rule is
# chosen because the peak is narrow next to that spacing: the smoother
# bicubic rule would overshoot around it, leaving a ring of values of
# the wrong sign that the kernel does not have. Its name, `K_vs`, and its
# units are the code's: a
# kernel per unit volume of a dimensionless objective with respect to a
# velocity has units of s m^-4. The library has no list of kernels.

# %%
R, CMB, D670 = 6371e3, 3480e3, 5701e3
RECEIVER = (np.pi / 3, np.pi / 6)            # colatitude, longitude, in radians


def from_receiver(t, p):
    """The angle between a direction and the receiver's."""
    t0, p0 = RECEIVER
    c = np.cos(t) * np.cos(t0) + np.sin(t) * np.sin(t0) * np.cos(p - p0)
    return np.arccos(np.clip(c, -1.0, 1.0))


def shear_kernel(r, t, p):
    """A pattern of both signs, zero at the core-mantle boundary and the
    surface, and a peak 200 km wide at the receiver."""
    s = (r - CMB) / (R - CMB)
    delta = from_receiver(t, p)
    broad = np.sin(np.pi * s) * np.cos(2.0 * delta)
    d2 = r ** 2 + R ** 2 - 2.0 * r * R * np.cos(delta)
    return 1e-15 * (broad + 20.0 * np.exp(-d2 / (2.0 * 200e3 ** 2)))


def shape_kernel(t, p):
    """A pattern of both signs and a peak beneath the receiver."""
    delta = from_receiver(t, p)
    c = np.cos(delta)
    return 1e-9 * (0.5 * (3.0 * c ** 2 - 1.0) + 5.0 * np.exp(-delta ** 2 / 0.02))


def write_standin(path):
    """The file of the C++ example, written by the Python."""
    skeleton = Skeleton([0.0, CMB, D670, R])
    geometry = Geometry(skeleton, layer_names=["core", "lower mantle", "upper mantle"],
                        interface_names=["cmb", "d670", "surface"])
    core = RadialField((0.0, CMB), lambda r: 12000.0 - 2000.0 * (r / CMB) ** 2,
                       character=DENSITY)
    rho = [core] + [RadialField(skeleton.interval(i),
                                lambda r: 5500.0 - 2200.0 * (r - CMB) / (R - CMB),
                                character=DENSITY)
                    for i in (1, 2)]
    kernel = [AnalyticField(skeleton.interval(i), shear_kernel, name="K_vs")
              for i in (1, 2)]
    specs = {
        "K_vs": FieldSpec(SCALAR, Dimensions(length=-4, time=1),
                          meaning="derivative of the objective with respect to vs, "
                                  "per unit volume"),
        "K_d670": FieldSpec(SCALAR, Dimensions(length=-3),
                            meaning="derivative of the objective with respect to a "
                                    "shift of d670, per unit area")}
    model = Model(geometry, [{"rho": rho[0]}, {"rho": rho[1], "K_vs": kernel[0]},
                             {"rho": rho[2], "K_vs": kernel[1]}],
                  specs=specs, name="synthetic kernel")
    # the code's nodes: every 500 km in the core, every 50 km in the mantle
    b = skeleton.boundaries
    edges = np.unique(np.concatenate([
        np.linspace(b[i], b[i + 1], int(np.ceil((b[i + 1] - b[i]) / step)) + 1)
        for i, step in enumerate((500e3, 50e3, 50e3))]))
    fine = gauss_legendre(64)
    return write_model(path, model, grid={0: gauss_legendre(4), 1: fine, 2: fine},
                       radial=RadialMesh(skeleton, ngll=2, edges=edges),
                       interpolation={"K_vs": ("polynomial", "linear"),
                                      "K_d670": ("polynomial", "linear")},
                       surfaces={"K_d670": {"d670": shape_kernel}},
                       surface_grid=fine, meta={"code": "planetmodel tutorial 15"})


tmp = tempfile.TemporaryDirectory()
standin = write_standin(Path(tmp.name) / "kernel.nc")
path = Path(sys.argv[1]) if len(sys.argv) > 1 else standin
print("reading", path)

# %% [markdown]
# ## Reading it
#
# `read_model` gives the model, its fields the kernels; `read_surfaces`
# gives the fields on interfaces beside it, by name and then by
# interface. The model's *specs* (the meaning and units of each name)
# come from the file, so the kernels' units are the code's.

# %%
model = read_model(path)
on_faces = read_surfaces(path)
print(model)
print("K_vs is held on layers", model.layers_with("K_vs"), "in",
      slices.sphere(model, "K_vs", depth=100e3).label)
print("on interfaces:", {name: list(held) for name, held in on_faces.items()})

if path != standin:
    # the C++ file against the Python's, at the nodes both hold
    mine, theirs = read_model_file(standin), read_model_file(path)
    groups = [(f"part {a.layers}", a.variables, b.variables)
              for a, b in zip(mine.parts, theirs.parts)]
    groups.append(("on interfaces", mine.surfaces.variables, theirs.surfaces.variables))
    for where, ours, its in groups:
        assert set(ours) == set(its), f"{where}: {sorted(ours)} against {sorted(its)}"
        for name, var in ours.items():
            a, b = var.values, its[name].values
            worst = np.nanmax(np.abs(a - b)) / np.nanmax(np.abs(a))
            print(f"{name}, {where}: the largest difference is {worst:.1e} of the "
                  "largest value")
            assert worst < 1e-12, f"{name}, {where}: the two files differ"

# %% [markdown]
# ## Maps at depths
#
# `diverging=True` draws a field on a colour scale centred on zero, red
# for negative and blue for positive, whatever the field is. At 100 km
# depth the peak beside the receiver is a hundred times larger than the
# rest of the kernel, so the map scaled to it shows one dot (top left).
# `clip=99` sets the colour limits where 99 percent of the values lie
# within them, and the arrows on the colour bar mark that some values
# run beyond (top right). At 1500 km the peak has gone and the broad
# pattern of both signs is plain.
#
# The bottom right map is the kernel integrated along each radius
# through the mantle, the integral of K r^2 dr: how sensitive the
# objective is to the whole column beneath each point.
# `normalised=True` divides it by its largest absolute value, so its
# label has no unit; it is clipped too, since the peak is in the
# columns beside the receiver. (`coastlines=True` would add the
# coastlines, which cartopy downloads the first time; they are left out
# so that the tutorial runs offline.)

# %%
fig = plt.figure(figsize=(12, 7))
shallow = slices.sphere(model, "K_vs", depth=100e3)
deep = slices.sphere(model, "K_vs", depth=1500e3)
column = slices.integrated(model, "K_vs", radii=(CMB, R))
for k, (s, title, kw) in enumerate((
        (shallow, "100 km deep", {}), (shallow, "100 km deep, clipped", {"clip": 99}),
        (deep, "1500 km deep", {}), (column, "integrated through the mantle",
                                     {"normalised": True, "clip": 99}))):
    ax = fig.add_subplot(2, 2, k + 1, projection=ccrs.Robinson())
    sphere_map(ax, s, diverging=True, colorbar=True, **kw)
    ax.set_title(title)
fig.tight_layout()
out = FIGURES / "tutorial_15_maps.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## A section
#
# A cross-section along the great circle from the receiver eastward, a
# third of the way round the planet, drawn as a wedge of the mantle with
# the receiver at its left end: the peak is the dark spot at the top
# left, and the broad pattern changes sign along the arc. The same
# `clip=` keeps the peak from washing out the rest.

# %%
across = slices.section(model, "K_vs", RECEIVER, azimuth=0.5 * np.pi,
                        length=2.0 * np.pi / 3.0, radii=(CMB, R), m=241)
fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": "polar"})
section(ax, across, scale=1e-3, diverging=True, clip=99, colorbar=True)
ax.set_rticks([4000, 5000, 6000])
ax.set_title("K_vs eastward from the receiver", pad=20)
fig.tight_layout()
out = FIGURES / "tutorial_15_section.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## Profiles and the one-dimensional kernel
#
# On the left, the kernel along two radii, on a scale that is
# logarithmic away from zero so that both show: beneath the receiver,
# where it climbs to the peak near the surface, and 90 degrees south of
# it, where the broad pattern has the opposite sign and there is no
# peak.
#
# On the right, the *one-dimensional kernel*: r^2 times the integral of
# K over the sphere at each radius (`mean_profile(..., total=True)`). For
# a change dm that depends on the radius alone, dJ is the integral of
# this kernel times dm along the radius, so it is the kernel a
# spherically symmetric model would have. It can be compared directly
# with a kernel computed for a one-dimensional model.

# %%
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
under = slices.profile(model, "K_vs", *RECEIVER, radii=(CMB, R), n=200)
away = slices.profile(model, "K_vs", RECEIVER[0] + np.pi / 2, RECEIVER[1],
                      radii=(CMB, R), n=200)
profile(left, under, scale=1e-3, label="beneath the receiver")
profile(left, away, scale=1e-3, label="90 degrees south")
left.set_xscale("symlog", linthresh=1e-15)
left.set_xlabel(under.label)
left.set_ylabel("radius [km]")
left.legend()
one_d = slices.mean_profile(model, "K_vs", total=True)
profile(right, one_d, scale=1e-3)
right.set_xlabel(one_d.label)
right.set_ylim(CMB * 1e-3, R * 1e-3)
fig.tight_layout()
out = FIGURES / "tutorial_15_profiles.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## Spectra
#
# The power of the kernel by spherical-harmonic degree (tutorial 13). The
# three maps are normalised first, each divided by its largest value, so
# that their spectra are of numbers without units and can share one
# axis. At 1500 km depth the kernel's pattern is degrees 0 and 2; the
# other even degrees, a hundred million times weaker, come from reading
# the kernel between the file's nodes along straight lines, not from the
# kernel itself. At 100 km the peak, narrow in angle,
# spreads its power over every degree up to the highest: this is the
# part a regularising operator removes. The column integral, which
# takes in the peak, spreads its power in the same way.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
grid = gauss_legendre(96)
maps = ((slices.sphere(model, "K_vs", depth=100e3, grid=grid), "100 km deep"),
        (slices.sphere(model, "K_vs", depth=1500e3, grid=grid), "1500 km deep"),
        (slices.integrated(model, "K_vs", radii=(CMB, R), grid=grid),
         "integrated through the mantle"))
for s, label in maps:
    spectrum(ax, slices.spectrum_of(slices.normalised(s), lmax=48), label=label)
ax.set_ylim(1e-12, 1.0)
ax.set_xlabel("degree")
ax.set_ylabel("power, of each map divided by its largest value")
ax.legend()
ax.set_title("K_vs by degree")
fig.tight_layout()
out = FIGURES / "tutorial_15_spectra.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## A kernel on an interface
#
# The second kernel is of a shift of the 670 km discontinuity: how J
# changes when the interface moves up or down by a small amount at each
# point. It is a function on the interface, not a field of a layer, and
# the file holds it in a group of its own. `slices.surface` makes it a
# map on that interface, and `slices.spectrum_of` gives its spectrum, as
# for any map.

# %%
shape = slices.surface(model, on_faces["K_d670"]["d670"], "d670", name="K_d670")
fig = plt.figure(figsize=(12, 4.5))
ax = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
sphere_map(ax, shape, diverging=True, colorbar=True)
ax.set_title("sensitivity to a shift of the 670 km discontinuity")
ax = fig.add_subplot(1, 2, 2)
spectrum(ax, slices.spectrum_of(shape, lmax=32))
ax.set_xlabel("degree")
ax.set_ylabel("power")
ax.set_title("K_d670 by degree")
fig.tight_layout()
out = FIGURES / "tutorial_15_shape.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## In ParaView
#
# `write_vtk` writes the kernel for ParaView as tutorial 13 does; the
# core, which holds no kernel, is left blank. In ParaView, colour by
# `K_vs`, choose a diverging colour map (*Cool to Warm*), and set the
# range symmetric about zero by hand, a few times smaller than the
# largest value, so that the peak does not hide the rest.

# %%
vtk = write_vtk(FIGURES / "tutorial_15_vtk" / "tutorial_15", model,
                grid=gauss_legendre(32), n=12, fields=["K_vs"])
print("ParaView files written to", vtk)
tmp.cleanup()
