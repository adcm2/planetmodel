"""netCDF grids into the library's form, written here in every variant the
survey of EMC's files found, from one analytic field, so that each variant
must read to the same model."""
import hashlib

import numpy as np
import pytest

import planetmodel as pm
from planetmodel import datasets
from planetmodel.datasets import (Dataset, GridReader, RegionalGrid, Source, grid_model,
                                  read_grid)
from planetmodel.testing import check_dataset

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

R = 6371e3
DEPTHS = np.array([25.0, 100.0, 200.0, 410.0, 410.01, 600.0, 1000.0, 2000.0, 2890.0])


def field(depth_km, lat, lon):
    """vs in km/s at depths, latitudes and longitudes (degrees), broadcast,
    with a jump of 0.2 km/s across 410 km."""
    d, la, lo = np.broadcast_arrays(depth_km, np.radians(lat), np.radians(lon))
    jump = np.where(d > 410.005, 0.2, 0.0)
    return (4.4 + 2.5e-4 * d + jump + 0.05 * np.sin(la)
            + 0.03 * np.cos(la) * np.cos(lo) + 0.02 * np.cos(la) * np.sin(lo))


def write(path, *, depth=DEPTHS, lat=None, lon=None, order=("depth", "latitude",
          "longitude"), vertical="depth", vertical_units="km", units="km.s-1",
          values=None, fill=("_FillValue", 99999.0), valid_range=None, extra=None,
          name="vs", lat_units="degrees_north"):
    """A netCDF file of one variable on (depth, latitude, longitude) in the
    given dimension order, from `field` unless `values` are given."""
    lat = np.linspace(-90.0, 90.0, 46) if lat is None else np.asarray(lat, float)
    lon = np.linspace(-180.0, 180.0, 91) if lon is None else np.asarray(lon, float)
    v = (field(depth[:, None, None], lat[None, :, None], lon[None, None, :])
         if values is None else np.asarray(values, float))
    with netCDF4.Dataset(path, "w") as ds:
        axes = {"depth": depth, "latitude": lat, "longitude": lon}
        if vertical == "radius":
            axes = {"radius": R / 1e3 - depth, "latitude": lat, "longitude": lon}
            order = tuple("radius" if a == "depth" else a for a in order)
        for a, c in axes.items():
            ds.createDimension(a, c.size)
            var = ds.createVariable(a, "f8", (a,))
            var[:] = c
        ds["latitude"].units = lat_units
        ds["longitude"].units = "degrees_east"
        vname = "radius" if vertical == "radius" else "depth"
        ds[vname].units = vertical_units
        ds[vname].positive = "up" if vertical == "radius" else "down"
        kw = {}
        if fill and fill[0] == "_FillValue":
            kw["fill_value"] = np.float32(fill[1])
        var = ds.createVariable(name, "f4", order, **kw)
        var.set_auto_mask(False)
        if fill and fill[0] == "missing_value":
            var.missing_value = np.float32(fill[1])
        if valid_range is not None:
            var.valid_range = np.array(valid_range, dtype=np.float32)
        var.units = units
        canon = ("radius" if vertical == "radius" else "depth", "latitude", "longitude")
        var[:] = np.transpose(v, [canon.index(a) for a in order])
        for k, val in (extra or {}).items():
            ds.setncattr(k, val)
    return path


def vs_model(path, **kw):
    g = read_grid(path, **kw)
    return g, grid_model(g, {"vs": ("vs", None)}, name="t")


def at_random(model, rng, n=200):
    lo, hi = model.skeleton.boundaries[0], 6346e3
    r = rng.uniform(lo, hi, n)
    theta = np.arccos(rng.uniform(-1, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    out = np.full(n, np.nan)
    for i in range(n):
        loc = model.skeleton.locate(r[i])
        if loc.boundary is None and model.layer(loc.layer).names:
            out[i] = model.layer(loc.layer)["vs"](r[i], theta[i], phi[i])
    return out


def test_a_grid_reads_to_its_nodes_exactly(tmp_path):
    g, m = vs_model(write(tmp_path / "a.nc"))
    assert g.is_global and g.grid.kind == "equiangular" and g.grid.pole_nodes == (True,
                                                                                    True)
    assert g.grid.nphi == 90 and g.grid.longitudes[0] == 0.0     # the seam dropped
    pm.testing.check_model(m)
    lay = m.layer(m.skeleton.locate(R - 150e3).layer)
    theta, phi = g.grid.colatitudes[None, :, None], g.grid.longitudes[None, None, :]
    lat, lon = 90.0 - np.degrees(theta), np.degrees(phi)
    for d in (100.0, 200.0):
        got = lay["vs"](R - d * 1e3, theta, phi)
        want = 1e3 * field(d, lat, lon).astype(np.float32).astype(float)
        assert np.allclose(got, want, rtol=1e-12, atol=0.0)


def test_a_near_repeated_depth_is_a_boundary(tmp_path):
    g, m = vs_model(write(tmp_path / "a.nc"))
    jump = R - 410.005e3
    assert np.any(np.isclose(m.skeleton.boundaries, jump, rtol=0.0, atol=1e-6))
    assert int(np.sum(np.diff(g.radius) == 0.0)) == 1
    above = m.layer(m.skeleton.locate(jump + 1.0).layer)["vs"](jump + 1.0, 1.0, 1.0)
    below = m.layer(m.skeleton.locate(jump - 1.0).layer)["vs"](jump - 1.0, 1.0, 1.0)
    assert above - below == pytest.approx(-200.0, abs=1.0)


VARIANTS = {
    "lat, lon, depth": dict(order=("latitude", "longitude", "depth")),
    "depth, lon, lat": dict(order=("depth", "longitude", "latitude")),
    "latitude descending": dict(lat=np.linspace(90.0, -90.0, 46)),
    "longitudes from 0, no seam": dict(lon=np.arange(0.0, 360.0, 4.0)),
    "spelled-out units": dict(units="kilometers/second", vertical_units="kilometer "),
    "a radius axis": dict(vertical="radius", vertical_units="kilometer"),
    "missing_value, not _FillValue": dict(fill=("missing_value", 9999.0)),
    "no fill at all": dict(fill=None),
    "a valid_range narrower than the data": dict(valid_range=(4.5, 4.6)),
    "a misleading latitude unit": dict(lat_units="degrees"),
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_reads_to_the_same_model(tmp_path, variant):
    _, canonical = vs_model(write(tmp_path / "a.nc"))
    _, other = vs_model(write(tmp_path / "b.nc", **VARIANTS[variant]))
    a = at_random(canonical, np.random.default_rng(1))
    b = at_random(other, np.random.default_rng(1))
    assert np.allclose(a, b, rtol=1e-12, atol=0.0, equal_nan=True)


def test_the_shipped_kinds_are_recognised_and_others_are_custom(tmp_path):
    cells = read_grid(write(tmp_path / "c.nc", lat=np.arange(-89.0, 90, 2.0),
                            lon=np.arange(-179.0, 180, 2.0)))
    assert cells.grid.kind == "equiangular" and cells.grid.pole_nodes == (False, False)
    assert np.degrees(cells.grid.longitudes[0]) == pytest.approx(1.0)
    fours = read_grid(write(tmp_path / "f.nc", lat=np.linspace(-88.0, 88.0, 45)))
    assert fours.grid.kind == "equiangular"                         # 4-degree cells
    odd = read_grid(write(tmp_path / "o.nc", lat=np.linspace(-88.0, 88.0, 89)))
    assert odd.grid.kind == "custom"


def test_a_grid_without_poles_takes_its_nearest_rings_mean(tmp_path):
    lat = np.linspace(-88.0, 88.0, 45)
    g, m = vs_model(write(tmp_path / "o.nc", lat=lat))
    lay = m.layer(m.skeleton.locate(R - 150e3).layer)
    r = R - 100e3
    ring = np.asarray(g.values["vs"])[np.argmin(np.abs(g.radius - r)), 0] * 1e3
    assert lay["vs"](r, 0.0, 0.3) == pytest.approx(np.mean(ring), rel=1e-12)


def test_pole_rows_and_seams_hold_one_value_or_are_refused(tmp_path):
    lat = np.linspace(-90.0, 90.0, 46)
    lon = np.linspace(-180.0, 180.0, 91)
    v = field(DEPTHS[:, None, None], lat[None, :, None], lon[None, None, :])
    noisy = v.copy()
    noisy[:, -1, :] += 0.01 * np.cos(np.radians(lon))              # the north row
    path = write(tmp_path / "p.nc", values=noisy)
    with pytest.raises(ValueError, match="along the pole row.*pole_rows='mean'"):
        read_grid(path)
    g = read_grid(path, pole_rows="mean")
    assert np.ptp(g.values["vs"][:, 0, :], axis=-1).max() == 0.0
    seam = v.copy()
    seam[:, 1:-1, -1] += 0.01                                        # off the poles
    path = write(tmp_path / "s.nc", values=seam)
    with pytest.raises(ValueError, match="seam repeated; seam='mean'"):
        read_grid(path)
    g = read_grid(path, seam="mean")
    assert g.grid.nphi == 90


def test_a_glitched_coordinate_is_rebuilt_when_the_entry_says_so(tmp_path):
    lat = np.arange(-89.5, 90.0, 1.0)
    glitched = lat.copy()
    glitched[[76, 103]] = [-13.6, 13.6]                             # LITHO1.0's kind
    lon = np.arange(-179.5, 180.0, 4.0)
    v = field(DEPTHS[:, None, None], lat[None, :, None], lon[None, None, :])
    path = write(tmp_path / "g.nc", lat=glitched, lon=lon, values=v)
    assert read_grid(path).grid.kind == "custom"
    fixed = read_grid(path, regular=True)
    assert fixed.grid.kind == "equiangular"
    far = lat.copy()
    far[76] = -12.0
    with pytest.raises(ValueError, match="half a step"):
        read_grid(write(tmp_path / "f.nc", lat=far, lon=lon, values=v), regular=True)


def test_missing_slices_and_values(tmp_path):
    lat, lon = np.linspace(-90.0, 90.0, 46), np.linspace(-180.0, 180.0, 91)
    v = field(DEPTHS[:, None, None], lat[None, :, None], lon[None, None, :])
    top = v.copy()
    top[0] = 99999.0                                                 # the 25 km slice
    g, m = vs_model(write(tmp_path / "t.nc", values=top))
    assert m.skeleton.boundaries[0] == pytest.approx(R - 2890e3)
    assert m.layers[-2].interval[1] == pytest.approx(R - 100e3)      # 25 km dropped
    inside = v.copy()
    inside[5] = np.nan
    with pytest.raises(ValueError, match="whole slice inside its support"):
        vs_model(write(tmp_path / "i.nc", values=inside))
    holes = v.copy()
    holes[4, 10, 10] = 99999.0
    with pytest.raises(ValueError, match="missing at 1 nodes.*fill="):
        vs_model(write(tmp_path / "h.nc", values=holes))
    g = read_grid(write(tmp_path / "h2.nc", values=holes))
    model = grid_model(g, {"vs": ("vs", None)}, fill=4500.0)
    pm.testing.check_model(model)


def test_units_are_checked_against_what_a_variable_is(tmp_path):
    g = read_grid(write(tmp_path / "u.nc"))
    with pytest.raises(ValueError, match="must be of dimensionless"):
        grid_model(g, {"vs": ("vs", "relative")}, reference=pm.PREM())
    pct = read_grid(write(tmp_path / "p.nc", units="percent", name="dvs",
                          values=np.full((DEPTHS.size, 46, 91), 1.5)))
    m = grid_model(pct, {"dvs": ("vs", "relative")}, reference=pm.PREM())
    lay = m.layer("lower_mantle")
    assert lay["vs"].perturbation(5e6, 1.0, 1.0) == pytest.approx(0.015)
    wrong = read_grid(write(tmp_path / "w.nc", units="g.cm-1"))
    with pytest.raises(ValueError, match="units= corrects a mislabel"):
        grid_model(wrong, {"vs": ("vs", None)})
    fixed = grid_model(wrong, {"vs": ("vs", None)}, units={"vs": "km/s"})
    pm.testing.check_model(fixed)
    with pytest.raises(ValueError, match="states no unit"):
        read_grid(write(tmp_path / "e.nc", vertical_units=""))
    assert read_grid(write(tmp_path / "e2.nc", vertical_units=""),
                     vertical_units="km").radius is not None


def test_a_regional_grid_evaluates_inside_its_box(tmp_path):
    lat, lon = np.arange(50.0, 70.01, 1.0), np.arange(-20.0, 20.01, 1.0)
    path = write(tmp_path / "r.nc", lat=lat, lon=lon, depth=DEPTHS[:6])
    g = read_grid(path)
    assert not g.is_global and g.grid is None
    rg = RegionalGrid(g, {"vs": ("vs", None)})
    (r0, r1), (t0, t1), (p0, p1) = rg.extent
    assert np.degrees([p0, p1]) == pytest.approx([340.0, 380.0])       # contiguous
    theta, phi = pm.from_geographic(lat[3], lon[30])                 # at a node
    got = rg("vs", R - 100e3, theta, phi)
    assert got == pytest.approx(1e3 * float(np.float32(field(100.0, lat[3], lon[30]))))
    assert rg.contains(*pm.from_geographic(60.0, 0.0))
    assert not rg.contains(*pm.from_geographic(60.0, 30.0))
    assert np.isnan(rg("vs", R - 100e3, *pm.from_geographic(60.0, 30.0)))
    assert np.isnan(rg("vs", R - 1000e3, theta, phi))                # below its support
    with pytest.raises(ValueError, match="global grid"):
        grid_model(g, {"vs": ("vs", None)})


def test_a_reader_gives_a_model_over_its_reference_or_a_regional_grid(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setenv("PLANETMODEL_DATA", str(tmp_path / "cache"))
    pct = write(tmp_path / "p.nc", units="%", name="dvs",
                values=np.full((DEPTHS.size, 46, 91), -2.0))
    d = Dataset("pct", (Source(pct.as_uri(), hashlib.sha256(pct.read_bytes())
                               .hexdigest()),),
                GridReader({"dvs": ("vs", "relative")}), title="t", citation=("c",),
                licence="l", reference="PREM")
    m = datasets.load(d)
    lay = m.layer("lower_mantle")
    assert isinstance(lay["vs"], pm.PerturbedField)
    want = 0.98 * float(lay["vs"].reference(5e6))
    assert lay["vs"](5e6, 1.0, 1.0) == pytest.approx(want)
    check_dataset(d, paths=datasets.fetch(d))
    box = write(tmp_path / "r.nc", lat=np.arange(50.0, 70.01, 1.0),
                lon=np.arange(-20.0, 20.01, 1.0), depth=DEPTHS[:6])
    r = Dataset("box", (Source(box.as_uri(), hashlib.sha256(box.read_bytes())
                               .hexdigest()),),
                GridReader({"vs": ("vs", None)}), title="t", citation=("c",),
                licence="l")
    assert isinstance(datasets.load(r), RegionalGrid)
    check_dataset(r, paths=datasets.fetch(r))
    coarse = datasets.load(d, stride=3)
    assert coarse.layer("lower_mantle")["vs"].perturbation.grid.ntheta == 16


def test_refusals_of_what_is_not_a_grid_of_latitude_and_longitude(tmp_path):
    path = tmp_path / "xy.nc"
    with netCDF4.Dataset(path, "w") as ds:
        for a in ("x", "y"):
            ds.createDimension(a, 3)
            ds.createVariable(a, "f8", (a,))[:] = np.arange(3.0)
        ds.createVariable("vs", "f4", ("y", "x"))[:] = np.ones((3, 3))
    with pytest.raises(ValueError, match="projected"):
        read_grid(path)
    with pytest.raises(ValueError, match="pole_rows must be"):
        read_grid(write(tmp_path / "a.nc"), pole_rows="drop")


EMC_ENTRIES = ("S362ANI+M", "SEMUCB-WM1", "SGLOBE-rani", "GyPSuM", "GLAD-M25",
               "CSEM-North-Atlantic")


@pytest.mark.parametrize("name", EMC_ENTRIES)
def test_the_shipped_emc_entries_hold_to_their_contract(name):
    d = datasets.REGISTRY[name]
    check_dataset(d)
    assert isinstance(d.reader, GridReader)
    assert d.sources[0].url.startswith(
        "https://data.earthscope.org/archive/seismology/products/emc/netcdf/")
    assert "CC BY 4.0" in d.licence and any("10.17611/" in c for c in d.citation)
    assert any("Trabant" in c for c in d.citation)
