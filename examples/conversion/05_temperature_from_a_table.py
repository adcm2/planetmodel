# %% [markdown]
# # A temperature from a table
#
# An offline tool (BurnMan, Perple_X, VBRc) gives the shear speed as a
# table against temperature and pressure. A `PropertyTable` holds it, is
# written to and read from a small netCDF file, and is inverted for the
# temperature that gives a speed at a pressure. On a model, the anomaly
# is the temperature of the point's speed less that of the reference
# speed, both at the hydrostatic pressure. Here a made-up table stands in
# for a generator's.
#
# Needs: plot, netcdf
# Network: none
# Run: poetry run python examples/conversion/05_temperature_from_a_table.py

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import planetmodel as pm
from planetmodel.conversion import building
from planetmodel.conversion.tables import PropertyTable

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# A shear speed falling with temperature, faster as it nears a made-up
# solidus, and rising with pressure: monotone in temperature, as an
# inversion needs.

# %%
t = np.linspace(500.0, 4000.0, 71)
p = np.linspace(0.0, 140e9, 57)
tt, pp = np.meshgrid(t, p, indexing="ij")
vs = 5000.0 + 18e-9 * pp - 0.25 * (tt - 1600.0) - 120.0 * np.exp((tt - 2600.0) / 400.0)
table = PropertyTable(t, p, {"vs": vs}, name="a made-up table", units={"vs": "m s-1"})
with tempfile.TemporaryDirectory() as tmp:
    table = PropertyTable.read(table.write(Path(tmp) / "table.nc"))
print(f"at 25 GPa, 5.2 km/s is {table.temperature_of('vs', 5200.0, 25e9):.0f} K")

# %% [markdown]
# On PREM with a 1 % anomaly in the upper mantle.

# %%
f = pm.AnalyticField((5701e3, 6151e3), lambda r, t, p: 0.01 * np.cos(t) + 0 * r * p)
model = pm.PREM()
for name in ("vsv", "vsh"):
    model = pm.perturbed(model, name, f, kind="relative")
dt = building.table_anomaly(model, table)
i = model.layer("transition_zone_middle").index
r = np.array([5900e3])
at_pole = dt.fields[i](r, np.zeros(1), np.zeros(1))[0]
print(f"a 1 % fast anomaly at 470 km is {at_pole:.0f} K by the table")

# %%
fig, ax = plt.subplots(figsize=(6, 4))
for pressure in (0.0, 25e9, 75e9, 135e9):
    ax.plot(t, table("vs", t, np.full_like(t, pressure)) / 1e3,
            label=f"{pressure / 1e9:.0f} GPa")
ax.set_xlabel("temperature (K)")
ax.set_ylabel("Vs (km/s)")
ax.set_title(table.name)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "conversion_05.png", dpi=100)
print("figure written to", FIGURES / "conversion_05.png")
