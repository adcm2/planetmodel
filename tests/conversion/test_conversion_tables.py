"""Tables of properties against temperature and pressure: read
bilinearly, inverted exactly for temperature (the inverse of the
bilinear reading, at any point), refused outside, through a file in
other units, and a temperature anomaly from one on a model."""
import numpy as np
import pytest

import planetmodel as pm
from planetmodel.conversion import building, parameters
from planetmodel.conversion.tables import PropertyTable

T = np.linspace(300.0, 4000.0, 38)
P = np.linspace(0.0, 140e9, 29)


def linear(t, p):
    return 7000.0 - 0.5 * (t - 1500.0) + 2e-9 * (p - 50e9)


def curved(t, p):
    return 7000.0 - 0.5 * (t - 1500.0) - 1e-4 * (t - 1500.0) ** 2 + 2e-9 * p


def table(fn, **kw):
    tt, pp = np.meshgrid(T, P, indexing="ij")
    return PropertyTable(T, P, {"vs": fn(tt, pp), "rho": 3300.0 + 0 * tt},
                         name="a test table", units={"vs": "m s-1"}, **kw)


def test_read_bilinearly_and_inverted_exactly():
    rng = np.random.default_rng(0)
    t, p = rng.uniform(300.0, 4000.0, 500), rng.uniform(0.0, 140e9, 500)
    lin = table(linear)
    assert np.allclose(lin("vs", t, p), linear(t, p), rtol=1e-13)
    assert np.allclose(lin.temperature_of("vs", linear(t, p), p), t, rtol=1e-12)
    # any table: the inversion undoes its own bilinear reading
    cur = table(curved)
    assert np.allclose(cur.temperature_of("vs", cur("vs", t, p), p), t, rtol=1e-12)
    at_nodes = cur.temperature_of("vs", curved(T[5], P[3]), P[3])
    assert at_nodes == pytest.approx(T[5], rel=1e-13)


def test_refusals():
    lin = table(linear)
    with pytest.raises(ValueError, match="temperature lies outside"):
        lin("vs", 5000.0, 1e9)
    with pytest.raises(ValueError, match="pressure lies outside"):
        lin("vs", 1000.0, 200e9)
    with pytest.raises(ValueError, match="lies outside what"):
        lin.temperature_of("vs", 9000.0, 1e9)
    with pytest.raises(ValueError, match="not strictly monotone"):
        lin.temperature_of("rho", 3300.0, 1e9)
    with pytest.raises(KeyError, match="holds no 'vp'"):
        lin("vp", 1000.0, 1e9)
    with pytest.raises(ValueError, match="shape"):
        PropertyTable(T, P, {"vs": np.zeros((3, 3))}, name="x")
    with pytest.raises(ValueError, match="increases"):
        PropertyTable(T[::-1], P, {"vs": np.zeros((T.size, P.size))}, name="x")


@pytest.mark.netcdf
def test_a_file_in_other_units(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4",
                                  reason="needs the planetmodel[netcdf] extra")
    lin = table(linear, citation=["A table maker (2026)."])
    back = PropertyTable.read(lin.write(tmp_path / "t.nc"))
    assert np.array_equal(back.properties["vs"], lin.properties["vs"])
    assert back.citation == lin.citation and back.name == "a test table"
    path = tmp_path / "km.nc"
    with netCDF4.Dataset(path, "w") as ds:
        for axis, values, unit in (("temperature", T, "K"),
                                   ("pressure", P / 1e9, "GPa")):
            ds.createDimension(axis, values.size)
            var = ds.createVariable(axis, "f8", (axis,))
            var.units, var[:] = unit, values
        var = ds.createVariable("vs", "f8", ("temperature", "pressure"))
        var.units = "km/s"
        var[:] = lin.properties["vs"] / 1e3
    km = PropertyTable.read(path)
    assert np.allclose(km.pressure, P) and np.allclose(km.properties["vs"],
                                                       lin.properties["vs"])
    assert km.units["vs"] == "m s-1"


def test_a_temperature_anomaly_from_a_table():
    lo, hi = 3480e3, 5701e3
    f = pm.AnalyticField((lo, hi), lambda r, t, p: 0.01 * np.cos(t) + 0 * r * p)
    model = pm.PREM()
    for name in ("vsv", "vsh"):
        model = pm.perturbed(model, name, f, kind="relative")
    lin = table(linear, citation=["A table maker (2026)."])
    dt = building.table_anomaly(model, lin)
    assert dt.kind == "temperature" and dt.citation == ("A table maker (2026).",)
    i = model.layer("lower_mantle").index
    r, t, p = np.array([4000e3, 5000e3]), np.array([0.3, 2.0]), np.array([1.0, 4.0])
    v0 = np.sqrt((2 * model.layer(i)["vsv"].reference(r) ** 2
                  + model.layer(i)["vsh"].reference(r) ** 2) / 3.0)
    # linear in temperature: dT = -(Vs - Vs_0) / 0.5, the pressure cancelling
    want = -v0 * 0.01 * np.cos(t) / 0.5
    assert np.allclose(dt.fields[i](r, t, p), want, rtol=1e-11)
    # and on to a viscosity, citing the table
    model = model.with_field(i, "viscosity", pm.constant_field(
        model.layer(i).interval, 2e21, name="viscosity"))
    one = building.Anomaly("temperature", {i: dt.fields[i]}, dt.scales,
                           citation=dt.citation)
    eta = building.with_viscosity(model, one, parameters.SL3D_VISCOSITY)
    assert np.allclose(eta.layer(i)["viscosity"](r, t, p),
                       2e21 * np.exp(-0.01 * want), rtol=1e-11)
    assert "A table maker (2026)." in eta.citation
