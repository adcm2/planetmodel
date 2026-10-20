# %% [markdown]
# # A published viscosity model
#
# Lloyd et al. (2024) inferred a 3D viscosity of the mantle from the
# tomographic model GLAD-M25; the registry fetches it from the British
# Geological Survey and reads it as a model of the viscosity itself. Put
# on PREM's mantle, it becomes a log perturbation of PREM's own radial
# viscosity, which a model file stores and a GIA code reads.
#
# Needs: netcdf, maps
# Network: first run (GLAD-M25-viscosity, 142 MB)
# Run: poetry run python examples/conversion/07_published_viscosity.py

# %%
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import planetmodel as pm
from planetmodel import datasets
from planetmodel.plotting import sphere_map

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

published = datasets.load("GLAD-M25-viscosity")
print(datasets.describe("GLAD-M25-viscosity").splitlines()[0])

# %% [markdown]
# PREM given 5e20 Pa s above 670 km and 5e21 below, as the paper's
# reference does, and the published model put on it.

# %%
prem = pm.PREM(ocean=False)
for lay in prem.layers:
    lo, hi = lay.interval
    if 3480e3 <= lo and hi <= 6291e3:
        prem = prem.with_field(lay.index, "viscosity", pm.constant_field(
            (lo, hi), 5e21 if hi <= 5701e3 else 5e20, name="viscosity"))
mantle = [lay.index for lay in prem.layers if "viscosity" in lay]
model = datasets.transplanted(prem, published, names={"viscosity": "viscosity"},
                              mode="absolute", kind="log", layers=mantle, extend=True)

# %%
fig = plt.figure(figsize=(12, 4))
for k, depth in enumerate((250e3, 1000e3, 2500e3)):
    ax = fig.add_subplot(1, 3, k + 1, projection=ccrs.Robinson())
    at = pm.slices.sphere(model, "viscosity", depth=depth, relative_to="reference",
                          departure="log10")
    sphere_map(ax, at, colorbar=True, diverging=True, coastlines=True)
    ax.set_title(f"{depth / 1e3:.0f} km deep")
fig.tight_layout()
fig.savefig(FIGURES / "conversion_07.png", dpi=100)
print("figure written to", FIGURES / "conversion_07.png")
print(model.cite())
