"""Strata and the readers of crustal and lithospheric models, on files and
grids the tests write: CRUST1.0's cell order and surfaces, absent strata
of zero thickness and the rule for their properties, the mantle beneath
the Moho; LITHO1.0's absent strata, overlapping strata and lid; tables
of longitude, latitude and values; a surface from a grid's variable.
Nothing here touches the network."""
import numpy as np
import pytest

from planetmodel import equiangular, from_geographic
from planetmodel.datasets import (CRUST1_STRATA, LITHO1_STRATA, Dataset, GridData,
                                  GridSurfaceReader, Source, Strata, Stratum, Surface,
                                  TableStrataReader, litho1_strata, read_crust1,
                                  read_table)
from planetmodel.testing import check_dataset, check_strata

R = 6371e3


def entry(reader, name="test"):
    return Dataset(name, (Source("https://example.org/file.dat", "0" * 64),), reader,
                   title="A test", citation=("A, 2026",), licence="CC0")


# -- strata ------------------------------------------------------------------------

def test_strata_are_their_surfaces_and_properties():
    top = Surface(lambda t, p: 1000.0 + 0 * t, "elevation", datum=R)
    mid = Surface(lambda t, p: -10e3 + 2e3 * np.cos(t), "elevation", datum=R)
    base = Surface(lambda t, p: -30e3 + 0 * t, "elevation", datum=R)
    upper_vp = lambda t, p: 6000.0 + 0 * t
    s = Strata.from_surfaces((top, mid, base), ("upper", "lower"), {
        "upper": {"vp": (upper_vp, upper_vp)},
        "lower": {"vp": (lambda t, p: 6500.0 + 0 * t, lambda t, p: 7000.0 + 0 * t)}},
        below={"vp": lambda t, p: 8000.0 + 0 * t}, name="two")
    assert s.names == ("upper", "lower") and len(s) == 2
    assert s.surfaces == (top, mid, base)
    assert s["lower"] is s[1] and s["upper"].bottom is s["lower"].top
    t, p = np.array([0.3, 2.0]), np.array([1.0, 4.0])
    assert np.allclose(s["lower"].thickness(t, p), 20e3 + 2e3 * np.cos(t))
    a, b = s["lower"].top.radius(t, p), s["lower"].bottom.radius(t, p)
    assert np.allclose(s["lower"].value("vp", a, t, p), 6500.0)
    assert np.allclose(s["lower"].value("vp", b, t, p), 7000.0)
    assert np.allclose(s["lower"].value("vp", 0.5 * (a + b), t, p), 6750.0)
    assert np.allclose(s["upper"].value("vp", R, t, p), 6000.0)
    with pytest.raises(KeyError, match="no property"):
        s["upper"].value("vs", R, t, p)
    with pytest.raises(KeyError, match="no stratum"):
        s["crust"]
    check_strata(s)
    assert "below: ['vp']" in repr(s)
    with pytest.raises(ValueError, match="is not"):
        Strata((Stratum("a", top, mid), Stratum("b", base, base)))
    with pytest.raises(ValueError, match="repeat"):
        Strata((Stratum("a", top, mid), Stratum("a", mid, base)))
    with pytest.raises(ValueError, match="surfaces"):
        Strata.from_surfaces((top, base), ("a", "b"), {})


# -- CRUST1.0 ----------------------------------------------------------------------

def crust1_files(tmp_path):
    """CRUST1.0's four files, every cell alike (no water, 500 m of ice, no
    middle sediments) but one, at (10.5 N, 70.5 W), with 4 km of water, no
    ice, no middle sediments and a deeper Moho."""
    default = [0.5, 0.5, 0.0, -1.0, -1.0, -2.0, -12.0, -22.0, -35.0]
    bnds = np.tile(default, (180, 360, 1))
    lat = 89.5 - np.arange(180.0)
    lon = -179.5 + np.arange(360.0)
    i, j = int(np.flatnonzero(lat == 10.5)[0]), int(np.flatnonzero(lon == -70.5)[0])
    bnds[i, j] = [0.0, -4.0, -4.0, -4.5, -4.5, -5.0, -9.0, -15.0, -50.0]
    vp = np.tile([1.5, 3.81, 2.0, 0.0, 3.0, 6.0, 6.5, 7.0, 8.1], (180, 360, 1))
    vs = np.tile([0.0, 1.94, 1.0, 0.0, 1.6, 3.5, 3.7, 3.9, 4.5], (180, 360, 1))
    rho = np.tile([1.02, 0.92, 2.0, 0.0, 2.4, 2.7, 2.85, 3.0, 3.3], (180, 360, 1))
    vp[i, j, 8] = 7.9
    paths = []
    for suffix, table in (("bnds", bnds), ("vp", vp), ("vs", vs), ("rho", rho)):
        path = tmp_path / f"crust1.{suffix}"
        np.savetxt(path, table.reshape(64800, 9), fmt="%7.2f")
        paths.append(path)
    return paths


def test_crust1_is_read_cell_by_cell_from_the_top_down(tmp_path):
    c = read_crust1(crust1_files(tmp_path))
    assert c.names == CRUST1_STRATA and len(c.surfaces) == 9
    t, p = from_geographic(10.5, -70.5)
    t0, p0 = from_geographic(-40.5, 100.5)
    # the odd cell: its Moho 50 km down, no ice, no middle sediments
    moho = c["lower_crust"].bottom
    assert moho.measures == "elevation" and moho.datum == R
    assert np.isclose(moho(t, p), -50e3) and np.isclose(moho(t0, p0), -35e3)
    assert np.isclose(c["water"].thickness(t, p), 4e3)
    assert c["ice"].thickness(t, p) == 0.0
    assert c["middle_sediments"].thickness(t, p) == 0.0
    # an absent stratum takes the next present one's values below it
    assert np.isclose(c["ice"].value("vp", R - 4e3, t, p), 2000.0)
    assert np.isclose(c["middle_sediments"].value("vp", R - 4.5e3, t, p), 3000.0)
    assert np.isclose(c["middle_sediments"].value("vp", R - 1e3, t0, p0), 3000.0)
    assert np.isclose(c["water"].value("vs", R, t0, p0), 1940.0)    # the ice's
    # present strata keep their own, in SI; the mantle beneath the Moho
    assert np.isclose(c["upper_crust"].value("rho", R - 5e3, t0, p0), 2700.0)
    assert np.isclose(c["water"].value("vp", R, t, p), 1500.0)
    assert np.isclose(c.below["vp"](t, p), 7900.0) and np.isclose(c.below["vp"](t0, p0),
                                                                  8100.0)
    check_strata(c)


def test_crust1_refuses_what_is_not_its_files(tmp_path):
    paths = crust1_files(tmp_path)
    with pytest.raises(ValueError, match=".rho file"):
        read_crust1(paths[:3])
    rows = paths[0].read_text().splitlines()
    rows[0] = "   0.00   1.00  -3.69  -4.99  -4.99  -4.99  -5.67  -7.15 -11.75"
    paths[0].write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="above the one before"):
        read_crust1(paths)
    paths[0].write_text("\n".join(rows[:100]) + "\n")
    with pytest.raises(ValueError, match="64800"):
        read_crust1(paths)


# -- LITHO1.0 ----------------------------------------------------------------------

def litho1_grid():
    """LITHO1.0's variables on a coarse global grid: no water anywhere, no
    ice but in the south, sediments missing in the north, an overlap of
    the upper sediments into the upper crust in one row, as the EMC grid
    has."""
    grid = equiangular(18, 36)
    shape = (grid.ntheta, grid.nphi)
    north = (np.degrees(np.pi / 2 - grid.colatitudes) > 30.0)[:, None] * np.ones(shape)
    south = (np.degrees(np.pi / 2 - grid.colatitudes) < -60.0)[:, None] * np.ones(shape)
    tops = {"water": np.nan, "ice": -2.0, "upper_sediments": 0.0,
            "middle_sediments": 1.0, "lower_sediments": 2.0, "upper_crust": 3.0,
            "middle_crust": 12.0, "lower_crust": 22.0, "lid": 35.0}
    bottoms = {"water": np.nan, "ice": 0.0, "upper_sediments": 1.0,
               "middle_sediments": 2.0, "lower_sediments": 3.0, "upper_crust": 12.0,
               "middle_crust": 22.0, "lower_crust": 35.0, "lid": 120.0}
    values, units = {}, {}
    for n in LITHO1_STRATA:
        top = np.full(shape, tops[n], dtype=float)
        bottom = np.full(shape, bottoms[n], dtype=float)
        if n == "ice":
            top[south == 0], bottom[south == 0] = np.nan, np.nan
        if n.endswith("sediments"):
            top[north == 1], bottom[north == 1] = np.nan, np.nan
        if n == "upper_sediments":
            bottom[9] = 3.5                                  # into the crust
        values[f"{n}_top_depth"], values[f"{n}_bottom_depth"] = top, bottom
        units[f"{n}_top_depth"] = units[f"{n}_bottom_depth"] = "km"
        k = LITHO1_STRATA.index(n)
        for end, base in (("top", top), ("bottom", bottom)):
            present = np.where(np.isnan(base), np.nan, 1.0)
            for prop, value, unit in (("density", 2000.0 + 100 * k, "kg.m-3"),
                                      ("vp", 3.0 + 0.5 * k, "km.s-1"),
                                      ("vs", 1.5 + 0.3 * k, "km.s-1"),
                                      ("vp2", 3.1 + 0.5 * k, "km.s-1"),
                                      ("vs2", 1.6 + 0.3 * k, "km.s-1"),
                                      ("qkappa", 0.0, ""), ("qmu", 600.0, ""),
                                      ("eta", 1.0, "")):
                bump = 0.1 if end == "bottom" and prop == "vs" else 0.0
                values[f"{n}_{end}_{prop}"] = present * (value + bump)
                units[f"{n}_{end}_{prop}"] = unit
    for prop, value, unit in (("depth", 120.0, "km"), ("density", 3300.0, "kg.m-3"),
                              ("vp", 8.1, "km.s-1"), ("vs", 4.4, "km.s-1"),
                              ("vp2", 8.2, "km.s-1"), ("vs2", 4.5, "km.s-1"),
                              ("qkappa", 0.0, ""), ("qmu", 70.0, ""), ("eta", 1.0, "")):
        values[f"asthenospheric_mantle_top_{prop}"] = np.full(shape, value)
        units[f"asthenospheric_mantle_top_{prop}"] = unit
    return GridData(grid.colatitudes, grid.longitudes, None, values, units, grid=grid,
                    source="litho")


def test_litho1_s_lid_bottom_is_a_surface_of_depth():
    s = litho1_strata(litho1_grid())
    assert s.names == LITHO1_STRATA
    lab = s["lid"].bottom
    assert lab.measures == "depth" and lab.datum == R
    t, p = from_geographic(45.0, 10.0)                 # north: no ice, no sediments
    ts, ps = from_geographic(-75.0, 10.0)              # south: ice
    assert np.isclose(lab(t, p), 120e3) and np.isclose(lab.radius(t, p), R - 120e3)
    assert s["water"].thickness(t, p) == 0.0 and s["ice"].thickness(t, p) == 0.0
    assert np.isclose(s["ice"].thickness(ts, ps), 2e3)
    assert s["upper_sediments"].thickness(t, p) == 0.0
    assert np.isclose(s["upper_crust"].top(t, p), 3e3)
    # absent strata take the next present one's top values; in SI
    assert np.isclose(s["water"].value("vpv", R, t, p), 1000.0 * (3.0 + 0.5 * 5))
    assert np.isclose(s["water"].value("vpv", R, ts, ps), 1000.0 * (3.0 + 0.5 * 1))
    # a property linear between its top and bottom values
    crust = s["upper_crust"]
    a, b = crust.top.radius(t, p), crust.bottom.radius(t, p)
    assert np.isclose(crust.value("vsv", a, t, p), 1000.0 * (1.5 + 0.3 * 5))
    assert np.isclose(crust.value("vsv", b, t, p), 1000.0 * (1.6 + 0.3 * 5))
    assert np.isclose(crust.value("rho", 0.5 * (a + b), t, p), 2500.0)
    assert np.isclose(s.below["vsh"](t, p), 4500.0)
    check_strata(s)


def test_litho1_takes_the_next_top_where_a_stratum_overlaps_it():
    data = litho1_grid()
    s = litho1_strata(data)
    t = data.grid.colatitudes[9]                     # the row whose sediments overlap
    p = data.grid.longitudes[3]
    lat = 90.0 - np.degrees(t)
    assert lat <= 30.0                               # sediments present there
    assert np.isclose(s["upper_sediments"].bottom(t, p), 1e3)   # the next top, 1 km
    assert np.isclose(s["upper_crust"].top(t, p), 3e3)


# -- tables ------------------------------------------------------------------------

def write_table(path, *, header=True, comma=False, drop=None, twice=False):
    lon = -179.0 + 2.0 * np.arange(180)
    lat = -89.0 + 2.0 * np.arange(90)
    rows = []
    for lo in lon:                                          # longitude outer
        for la in lat:
            surface = 500.0 + 10.0 * np.cos(np.radians(la))
            moho = -30e3 - 100.0 * np.sin(np.radians(lo))
            lab = -100e3 - 1e3 * np.cos(np.radians(la))
            rows.append([lo, la, surface, moho, lab, 2800.0, 3300.0, 3400.0, 1.0])
    rows = np.array(rows)
    if drop is not None:
        rows = np.delete(rows, drop, axis=0)
    if twice:
        rows = np.vstack([rows, rows[:1]])
    sep = ", " if comma else "  "
    lines = (["LithoRef-like: a table", "", "LONG LAT ELEVATION MOHO LAB RHO_C RHO_L "
              "RHO_SL GEOID"] if header else [])
    lines += [sep.join(f"{v:.2f}" for v in r) + "\r" for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_table_is_read_as_a_grid_whatever_its_order(tmp_path):
    for comma in (False, True):
        d = read_table(write_table(tmp_path / "t.xyz", comma=comma),
                       columns=("ELEVATION", "LAB"), units={"LAB": "m"})
        assert d.grid.kind == "equiangular" and d.grid.ntheta == 90
        assert d.values["LAB"].shape == (90, 180) and d.units["LAB"] == "m"
        i = int(np.argmin(np.abs(d.theta - np.radians(90.0 - 11.0))))
        j = int(np.argmin(np.abs(d.phi - np.radians(31.0))))
        want = 500.0 + 10.0 * np.cos(np.radians(11.0))
        assert np.isclose(d.values["ELEVATION"][i, j], want, atol=0.01)
    named = read_table(write_table(tmp_path / "h.xyz", header=False),
                       names=("LONG", "LAT", "E", "M", "L", "A", "B", "C", "G"))
    assert set(named.values) == {"E", "M", "L", "A", "B", "C", "G"}
    with pytest.raises(ValueError, match="column names"):
        read_table(write_table(tmp_path / "h.xyz", header=False))
    with pytest.raises(ValueError, match="each once"):
        read_table(write_table(tmp_path / "d.xyz", drop=[7]))
    with pytest.raises(ValueError, match="each once"):
        read_table(write_table(tmp_path / "w.xyz", twice=True))
    with pytest.raises(KeyError, match="MOHOO"):
        read_table(write_table(tmp_path / "t.xyz"), columns=("MOHOO",))


def test_a_table_s_columns_are_strata(tmp_path):
    reader = TableStrataReader(surfaces=("ELEVATION", "MOHO", "LAB"),
                               names=("crust", "lithospheric_mantle"),
                               measures="elevation",
                               properties={"crust": {"rho": "RHO_C"},
                                           "lithospheric_mantle": {"rho": "RHO_L"}},
                               below={"rho": "RHO_SL"},
                               units={"ELEVATION": "m", "MOHO": "m", "LAB": "km",
                                      "RHO_C": "g.cm-3", "RHO_L": "kg m-3",
                                      "RHO_SL": "kg m-3"})
    path = write_table(tmp_path / "t.xyz")
    e = entry(reader, "LithoRef-like")
    s = reader.read((path,), e)
    assert s.names == ("crust", "lithospheric_mantle") and s.name == "LithoRef-like"
    t, p = from_geographic(31.0, 51.0)
    lab = s["lithospheric_mantle"].bottom
    assert np.isclose(lab(t, p), 1e3 * (-100e3 - 1e3 * np.cos(np.radians(31.0))),
                      rtol=1e-3)                      # the LAB column read as km
    assert np.isclose(s["crust"].value("rho", R - 10e3, t, p), 2.8e6)   # g/cm^3
    assert np.isclose(s.below["rho"](t, p), 3400.0)
    check_dataset(e)
    with pytest.raises(ValueError, match="above the one before"):
        TableStrataReader(surfaces=("LAB", "MOHO", "ELEVATION"), names=("a", "b"),
                          measures="elevation", properties={}).read((path,), e)
    with pytest.raises(ValueError, match="measures"):
        TableStrataReader(surfaces=("A", "B"), names=("a",), measures="radius",
                          properties={})


# -- a surface from a grid ---------------------------------------------------------

@pytest.mark.netcdf
def test_a_grid_s_variable_is_a_surface(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4",
                                  reason="needs the planetmodel[netcdf] extra")
    lat = -89.0 + 2.0 * np.arange(90)
    lon = -179.0 + 2.0 * np.arange(180)
    thick = 100.0 + 50.0 * np.cos(np.radians(lat))[:, None] * np.ones(lon.size)
    path = tmp_path / "cam.nc"
    with netCDF4.Dataset(path, "w") as ds:
        for name, c in (("latitude", lat), ("longitude", lon)):
            ds.createDimension(name, c.size)
            v = ds.createVariable(name, "f8", (name,))
            v[:] = c
        ds["latitude"].units, ds["longitude"].units = "degrees_north", "degrees_east"
        v = ds.createVariable("thickness", "f4", ("latitude", "longitude"),
                              fill_value=np.float32(99999.0))
        v.units = "km"
        v[:] = thick
    reader = GridSurfaceReader("thickness", measures="thickness", datum=R)
    s = reader.read((path,), entry(reader, "CAM-like"))
    t, p = from_geographic(41.0, 33.0)
    assert s.measures == "thickness" and s.name == "CAM-like"
    want = 1e3 * (100.0 + 50.0 * np.cos(np.radians(41.0)))
    assert np.isclose(s(t, p), want, rtol=1e-6)
    assert np.isclose(s.radius(t, p), R - s(t, p))
    with netCDF4.Dataset(path, "a") as ds:
        ds["thickness"][3, 4] = 99999.0
    with pytest.raises(ValueError, match="missing at 1 nodes"):
        reader.read((path,), entry(reader))
    with pytest.raises(ValueError, match="not a length"):
        GridSurfaceReader("thickness", measures="depth", units="km.s-1").read(
            (path,), entry(reader))
