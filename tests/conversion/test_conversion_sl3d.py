"""sl3d's chain held to the Fortran's own output: 400 of its nodes
(`data/sl3d_viscosity.txt`, written by sl3d's program visc_from_vs), the
chain from sl3d's dlnVs in the fast suite, and from S20RTS itself, fetched,
under the network marker."""
from pathlib import Path

import numpy as np
import pytest

import planetmodel as pm
from planetmodel.conversion import building, parameters as P

FIXTURE = Path(__file__).parent / "data" / "sl3d_viscosity.txt"
R, THETA, PHI, DLNVS, RATIO = np.loadtxt(FIXTURE).T


def test_the_chain_from_sl3d_s_anomaly_is_sl3d_s_viscosity():
    dt = P.SL3D_EXPANSIVITY.temperature(
        P.SL3D_DENSITY_SCALING.relative_density(DLNVS, R), R)
    ratio = np.exp(P.SL3D_VISCOSITY.log_viscosity(dt, R))
    # sl3d's ratio carries about 1e-8 of its own, even where dT is zero
    assert np.allclose(ratio, RATIO, rtol=2e-7, atol=0)
    assert RATIO.min() < 0.1 and RATIO.max() > 5.0


def test_sl3d_viscosity_is_the_three_steps():
    prem = pm.PREM()
    lo, hi = 3480e3, 5701e3
    for lay in prem.layers:
        a, b = lay.interval
        if lo <= a and b <= hi:
            prem = prem.with_field(lay.index, "viscosity",
                                   pm.constant_field((a, b), 1e21, name="viscosity"))
    f = pm.AnalyticField((lo, hi), lambda r, t, p: 0.01 * np.cos(t) + 0 * r * p)
    for n in ("vsv", "vsh"):
        prem = pm.perturbed(prem, n, f, kind="relative")
    m = building.sl3d_viscosity(prem)
    r = np.linspace(lo + 1e3, hi - 1e3, 5)
    t, p = np.full(5, 0.7), np.full(5, 2.0)
    i = prem.layer("lower_mantle").index
    sel = (r >= prem.layer(i).interval[0]) & (r <= prem.layer(i).interval[1])
    dt = -P.SL3D_DENSITY_SCALING.scaling(r) * 0.01 * np.cos(0.7) / \
        P.SL3D_EXPANSIVITY.expansivity(r)
    want = 1e21 * np.exp(-0.01 * dt)
    assert np.allclose(m.layer(i)["viscosity"](r[sel], t[sel], p[sel]), want[sel],
                       rtol=1e-12)
    assert m.cite().startswith(prem.cite())


@pytest.mark.network
def test_from_s20rts_itself():
    from planetmodel.datasets import (fetch, over_reference, read_sph, sph_field,
                                      transplanted)
    (path,) = fetch("S20RTS")
    vs = sph_field(read_sph(path), name="vs")
    s20 = over_reference(pm.PREM(), {"vs": vs}, kind="relative", name="S20RTS")
    prem = pm.PREM()
    for lay in prem.layers:
        a, b = lay.interval
        if a >= 3480e3 and b <= 6346.6e3:
            prem = prem.with_field(lay.index, "viscosity",
                                   pm.constant_field((a, b), 1e21, name="viscosity"))
    m = building.sl3d_viscosity(transplanted(prem, s20, remove_mean=True))
    a = building.velocity_anomaly(m)
    got, anomaly = np.empty_like(R), np.empty_like(R)
    for lay in m.layers:
        lo, hi = lay.interval
        sel = (R >= lo) & (R < hi)
        if sel.any() and lay.index in a.layers:
            got[sel] = lay["viscosity"](R[sel], THETA[sel], PHI[sel]) / 1e21
            anomaly[sel] = a.fields[lay.index](R[sel], THETA[sel], PHI[sel])
    # sl3d holds S20RTS in single precision
    assert np.allclose(anomaly, DLNVS, rtol=0, atol=2e-7)
    assert np.allclose(got, RATIO, rtol=1e-5, atol=0)
