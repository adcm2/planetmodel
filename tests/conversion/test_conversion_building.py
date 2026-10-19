"""Laws applied to models: the velocity anomaly read from a model's shear
speed (perturbed, Voigt-averaged, fully 3D), the temperature anomaly by
either route, the viscosity and density they write, renormalised means,
units at the boundary, citations, refusals, and the model file."""
import numpy as np
import pytest

import planetmodel as pm
from planetmodel.conversion import building, parameters as P
from planetmodel.units import TEMPERATURE
from planetmodel.vocabulary import FieldSpec

LO, HI = 3480e3, 5701e3


def dlnvs(r, t, p):
    return 0.01 * np.sin(t) ** 2 * np.cos(2 * p) * (0.5 + (r - LO) / (HI - LO))


def viscous_prem():
    """PREM with a radial viscosity in its mantle (2e21 below 670 km, 5e20
    above) and the anomaly above on vsv and vsh between the CMB and 670 km."""
    prem = pm.PREM()
    for lay in prem.layers:
        lo, hi = lay.interval
        if LO <= lo and hi <= 6291e3:
            eta = 2e21 if hi <= HI else 5e20
            prem = prem.with_field(lay.index, "viscosity",
                                   pm.constant_field((lo, hi), eta, name="viscosity"))
    f = pm.AnalyticField((LO, HI), dlnvs)
    for name in ("vsv", "vsh"):
        prem = pm.perturbed(prem, name, f, kind="relative")
    return prem


@pytest.fixture(scope="module")
def model():
    return viscous_prem()


def points(lay, n=50, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = lay.interval
    return (rng.uniform(lo, hi, n), np.arccos(rng.uniform(-1, 1, n)),
            rng.uniform(0, 2 * np.pi, n))


def test_the_velocity_anomaly_is_the_perturbation_of_the_voigt_speed(model):
    a = building.velocity_anomaly(model)
    mantle = [lay.index for lay in model.layers
              if lay.interval[0] >= LO and lay.interval[1] <= HI]
    assert a.kind == "velocity" and a.layers == tuple(mantle)
    for i in a.layers:
        r, t, p = points(model.layer(i))
        assert np.allclose(a.fields[i](r, t, p), dlnvs(r, t, p), rtol=1e-12,
                           atol=1e-16)
    # the pair perturbed apart: the Voigt speed's own departure
    prem = pm.PREM()
    g = pm.AnalyticField((LO, HI), lambda r, t, p: 0.02 + 0 * r * t * p)
    only = pm.perturbed(prem, "vsh", g, kind="relative")
    i = only.layer("lower_mantle").index
    r = np.array([5000e3])
    v0, h0 = prem.layer(i)["vsv"](r), prem.layer(i)["vsh"](r)
    want = np.sqrt((2 * v0 ** 2 + (1.02 * h0) ** 2) / (2 * v0 ** 2 + h0 ** 2)) - 1
    got = building.velocity_anomaly(only).fields[i](r, np.array([1.0]),
                                                    np.array([2.0]))
    assert np.allclose(got, want, rtol=1e-13)
    with pytest.raises(ValueError, match="departs"):
        building.velocity_anomaly(prem)


def test_a_fully_3d_speed_is_taken_against_its_spherical_mean():
    two = pm.LayeredIsotropicElastic([0.0, LO, 6371e3], rho=[1e4, 4e3],
                                     vp=[9e3, 11e3], vs=[0.0, 6e3])
    i = 1
    lay = two.layer(i)
    grid = pm.gauss_legendre(12)
    vs = pm.ComposedField(lambda v, d: v * (1.0 + d), [lay["vs"], pm.AnalyticField(
        lay.interval, lambda r, t, p: 0.01 * np.cos(t) + 0.005 * np.sin(t) ** 2)],
        character=pm.SCALAR)
    three = two.with_field(i, "vs", pm.GriddedField.sampled(vs, grid, ngll=4,
                                                            lateral="harmonic"),
                           replace=True)
    a = building.velocity_anomaly(three, layers=[i], grid=grid)
    r = np.full(grid.ntheta * grid.nphi, 5000e3)
    t = np.repeat(grid.colatitudes, grid.nphi)
    p = np.tile(grid.longitudes, grid.ntheta)
    values = a.fields[i](r, t, p).reshape(grid.ntheta, grid.nphi)
    assert abs(float(grid.mean(values))) < 1e-12


def test_sl3d_s_route_to_temperature_and_viscosity(model):
    v = building.velocity_anomaly(model)
    dt = building.temperature_anomaly(v, scaling=P.SL3D_DENSITY_SCALING,
                                      expansivity=P.SL3D_EXPANSIVITY)
    assert dt.kind == "temperature" and dt.citation[0].startswith("Crawford")
    m = building.with_viscosity(model, dt, P.SL3D_VISCOSITY)
    for i in dt.layers:
        r, t, p = points(model.layer(i), seed=i)
        want_dt = (-P.SL3D_DENSITY_SCALING.scaling(r) * dlnvs(r, t, p)
                   / P.SL3D_EXPANSIVITY.expansivity(r))
        assert np.allclose(dt.fields[i](r, t, p), want_dt, rtol=1e-12)
        eta = m.layer(i)["viscosity"]
        assert isinstance(eta, pm.PerturbedField) and eta.kind == "log"
        eta0 = model.layer(i)["viscosity"](r)
        assert np.allclose(eta(r, t, p), eta0 * np.exp(-0.01 * want_dt), rtol=1e-12)
    above = model.layer("transition_zone_lower")
    assert m.layer(above.index)["viscosity"] is above["viscosity"]
    assert m.citation == model.citation + tuple(
        w for w in P.SL3D_DENSITY_SCALING.citation + P.SL3D_EXPANSIVITY.citation[1:])
    pm.testing.check_model(m)


def test_the_beta_law_is_the_exponential_law_of_its_own_temperature(model):
    v = building.velocity_anomaly(model)
    beta = P.ivins_sammis_beta(1.0, -1.2e-4)
    by_beta = building.with_viscosity(model, v, beta)
    dt = building.temperature_anomaly(v, sensitivity=P.Sensitivity(-1.2e-4, name="s"))
    eps = P.ExponentialViscosity(pm.conversion.profiles.Profile(
        lambda r: beta.activation(r) / beta.background(r) ** 2, description="a/T0^2"),
        name="eps")
    by_eps = building.with_viscosity(model, dt, eps)
    i = v.layers[1]
    r, t, p = points(model.layer(i))
    assert np.allclose(by_beta.layer(i)["viscosity"](r, t, p),
                       by_eps.layer(i)["viscosity"](r, t, p), rtol=1e-12)
    assert "Wu, P." in by_beta.cite()


def test_renormalised_each_radius_keeps_its_mean_viscosity(model):
    grid = pm.gauss_legendre(16)
    dt = building.temperature_anomaly(building.velocity_anomaly(model),
                                      sensitivity=P.Sensitivity(-1e-4, name="s"))
    m = building.with_viscosity(model, dt, P.MILNE_2018_VISCOSITY, renormalise=grid)
    i = dt.layers[0]
    lo, hi = model.layer(i).interval
    for r in (lo + 0.2 * (hi - lo), 0.5 * (lo + hi)):
        th = np.repeat(grid.colatitudes, grid.nphi)
        ph = np.tile(grid.longitudes, grid.ntheta)
        eta = m.layer(i)["viscosity"](np.full(th.shape, r), th, ph)
        mean = float(grid.mean(eta.reshape(grid.ntheta, grid.nphi)))
        assert mean == pytest.approx(2e21, rel=1e-12)
        assert np.ptp(eta) > 0.1 * 2e21


def test_density_follows_and_so_do_the_moduli(model):
    v = building.velocity_anomaly(model)
    m = building.with_density(model, v, P.PAULSON_2005_DENSITY_SCALING)
    i = v.layers[0]
    r, t, p = points(model.layer(i))
    rho0 = model.layer(i)["rho"](r)
    lay = m.layer(i)
    assert np.allclose(lay["rho"](r, t, p), rho0 * (1 + 0.3 * dlnvs(r, t, p)),
                       rtol=1e-12)
    rho, vsv = lay["rho"](r, t, p), lay["vsv"](r, t, p)
    assert np.allclose(lay["L"](r, t, p), rho * vsv ** 2, rtol=1e-12)
    dt = building.temperature_anomaly(v, sensitivity=P.Sensitivity(-1e-4, name="s"))
    by_t = building.with_density(model, dt, P.SL3D_EXPANSIVITY)
    assert np.allclose(by_t.layer(i)["rho"](r, t, p),
                       rho0 * (1 - P.SL3D_EXPANSIVITY.expansivity(r) * dlnvs(r, t, p)
                               / -1e-4), rtol=1e-12)


def test_units_are_converted_at_the_boundary(model):
    nd = model.nondimensionalised()
    dt = building.temperature_anomaly(building.velocity_anomaly(nd),
                                      scaling=P.SL3D_DENSITY_SCALING,
                                      expansivity=P.SL3D_EXPANSIVITY)
    m = building.with_viscosity(nd, dt, P.SL3D_VISCOSITY)
    si = building.with_viscosity(model, building.temperature_anomaly(
        building.velocity_anomaly(model), scaling=P.SL3D_DENSITY_SCALING,
        expansivity=P.SL3D_EXPANSIVITY), P.SL3D_VISCOSITY)
    i = dt.layers[1]
    r, t, p = points(model.layer(i))
    k = nd.scales.factor(nd.spec("viscosity").dimensions)
    assert np.allclose(k * m.layer(i)["viscosity"](r / nd.scales.length, t, p),
                       si.layer(i)["viscosity"](r, t, p), rtol=1e-12)
    # an intermediate attached in the model's units
    spec = FieldSpec(pm.SCALAR, TEMPERATURE, meaning="temperature anomaly")
    shown = building.attached(nd, dt, "dT", spec)
    kt = nd.scales.factor(TEMPERATURE)
    assert np.allclose(kt * shown.layer(i)["dT"](r / nd.scales.length, t, p),
                       dt.fields[i](r / nd.scales.length, t, p), rtol=1e-12)
    assert shown.spec("dT") == spec


def test_refusals(model):
    v = building.velocity_anomaly(model)
    with pytest.raises(ValueError, match="takes a temperature anomaly"):
        building.with_viscosity(model, v, P.SL3D_VISCOSITY)
    with pytest.raises(TypeError, match="no law of viscosity"):
        building.with_viscosity(model, v, P.SL3D_DENSITY_SCALING)
    with pytest.raises(KeyError, match="holds no 'viscosity'"):
        building.with_viscosity(pm.perturbed(pm.PREM(), "vsv", pm.AnalyticField(
            (LO, HI), dlnvs), kind="relative"), v, P.ivins_sammis_beta(0.5, -1e-4))
    with pytest.raises(ValueError, match="sensitivity=, or scaling="):
        building.temperature_anomaly(v, scaling=P.SL3D_DENSITY_SCALING)
    with pytest.raises(ValueError, match="made from a velocity anomaly"):
        building.temperature_anomaly(building.temperature_anomaly(
            v, sensitivity=P.Sensitivity(-1e-4, name="s")),
            sensitivity=P.Sensitivity(-1e-4, name="s"))


@pytest.mark.netcdf
def test_a_converted_model_writes_and_reads_back(model, tmp_path):
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.modelfile import read_model, write_model
    dt = building.temperature_anomaly(building.velocity_anomaly(model),
                                      scaling=P.SL3D_DENSITY_SCALING,
                                      expansivity=P.SL3D_EXPANSIVITY)
    m = building.with_viscosity(model, dt, P.SL3D_VISCOSITY)
    path = write_model(tmp_path / "eta.nc", m, grid=pm.gauss_legendre(12),
                       fields=["rho", "vsv", "vsh", "viscosity"])
    pm.testing.check_model_file(path, m, rtol=1e-10)
    back = read_model(path)
    assert back.citation == m.citation
    i = dt.layers[0]
    f = back.layer(i)["viscosity"]
    assert isinstance(f, pm.PerturbedField) and f.kind == "log"
