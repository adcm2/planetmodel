"""Strata put on a model: the reliefs of the interfaces moved to their
surfaces, in the model's units and on the surfaces' grid; the geometry
route, whose mapping is built from exactly those reliefs and whose layer
between two moved interfaces holds the thickness-weighted strata; the
fields route, which samples the strata through the chosen layers of a
spherical skeleton; and what each refuses, by name."""
import numpy as np
import pytest

import planetmodel as pm
from planetmodel.datasets import Strata, Surface, relief, reliefs, with_strata
from planetmodel.displacement import layer_linear
from planetmodel.frames import cartesian_points
from planetmodel.gridded import GriddedFunction
from planetmodel.testing import check_geometry, check_model

R = 6371e3
GRID = pm.equiangular(18, 36)                   # 10-degree cells, no pole nodes


def on_grid(fn, *, lateral="linear"):
    return GriddedFunction.sampled(fn, GRID, lateral=lateral)


def depth(d0, amplitude):
    """A smooth depth below sea level (m), single-valued at the poles."""
    return on_grid(lambda t, p: d0 + amplitude * (np.cos(t) ** 2
                                                   + 0.5 * np.sin(t) * np.cos(p)))


def constant(value):
    return on_grid(lambda t, p: np.full(np.broadcast(t, p).shape, float(value)))


@pytest.fixture(scope="module")
def strata():
    """Three strata from 1 to 3 km below sea level down to a Moho 18 to 36
    km deep, with vp, vs and rho each (rho linear in the lower crust), and
    the mantle below."""
    levels = [depth(2e3, -1e3), depth(6e3, 2e3), depth(14e3, 4e3), depth(24e3, 8e3)]
    surfaces = [Surface(f, "depth", datum=R, name=n) for f, n in
                zip(levels, ["top", "upper", "lower", "moho"])]
    props = {"sediments": {"vp": (constant(2500.0),) * 2, "vs": (constant(1200.0),) * 2,
                           "rho": (constant(2100.0),) * 2},
             "upper_crust": {"vp": (constant(6000.0),) * 2,
                             "vs": (constant(3500.0),) * 2,
                             "rho": (constant(2700.0),) * 2},
             "lower_crust": {"vp": (constant(6800.0),) * 2,
                             "vs": (constant(3900.0),) * 2,
                             "rho": (constant(2850.0), constant(3050.0))}}
    return Strata.from_surfaces(surfaces, list(props), props, name="three",
                                below={"vp": constant(8100.0), "vs": constant(4500.0),
                                       "rho": constant(3300.0)}).with_citation("S.")


@pytest.fixture(scope="module")
def prem():
    """PREM without its ocean, its crust one layer between the Moho and
    the surface."""
    return pm.PREM(ocean=False).coarsened(drop=["d15"])[0]


def nodes():
    return GRID.colatitudes[:, None], GRID.longitudes[None, :]


def test_a_relief_is_the_surface_s_radius_less_the_interface_s(strata, prem):
    moho = strata.surfaces[-1]
    h = relief(moho, prem, "moho")
    assert h.grid == GRID and h.lateral == "cubic"
    t, p = nodes()
    b = prem.geometry.interface("moho").radius
    assert np.allclose(h.values, moho.radius(t, p) - b, rtol=0, atol=1e-6)
    assert np.allclose(h.poles, moho.radius(np.array([0.0, np.pi]), 0.0) - b, atol=1e-6)
    # in the model's units
    nd = prem.nondimensionalised()
    h_nd = relief(moho, nd, "moho", lateral="linear")
    assert np.allclose(h_nd.values * nd.scales.length, h.values, rtol=1e-12, atol=1e-6)
    assert h_nd.lateral == "linear"
    # a surface on no grid of its own needs one
    smooth = Surface(lambda t, p: 30e3 + 0 * t * p, "depth", datum=R)
    with pytest.raises(ValueError, match="grid="):
        relief(smooth, prem, "moho")
    assert relief(smooth, prem, "moho", grid=GRID).values.shape == (18, 36)


def test_reliefs_are_keyed_as_asked_and_refuse_by_name(strata, prem):
    out = reliefs(strata, prem, interfaces={"surface": 0, "moho": strata.surfaces[-1]})
    assert list(out) == ["moho", "surface"]                 # centre outward
    assert np.array_equal(out["moho"].values,
                          relief(strata.surfaces[-1], prem, "moho").values)
    assert np.array_equal(reliefs(strata, prem, interfaces={9: -1})[9].values,
                          out["moho"].values)
    assert np.array_equal(reliefs(strata, prem, interfaces={"surface": "sediments"})
                          ["surface"].values, out["surface"].values)
    with pytest.raises(ValueError, match="is not above"):
        reliefs(strata, prem, interfaces={"surface": -1, "moho": 0})
    with pytest.raises(ValueError, match="are one"):
        reliefs(strata, prem, interfaces={"moho": -1, 9: 0})
    with pytest.raises(ValueError, match="is not a surface of"):
        reliefs(strata, prem, interfaces={"moho": Surface(constant(3e4), "depth",
                                                          datum=R)})
    with pytest.raises(KeyError, match="no interface named"):
        reliefs(strata, prem, interfaces={"nonesuch": 0})
    with pytest.raises(ValueError, match="at least one"):
        reliefs(strata, prem, interfaces={})
    # the surface moved below the Moho the model keeps: its crust inverted
    with pytest.raises(ValueError, match=r"layer 10 would be .* a node"):
        reliefs(strata, prem, interfaces={"surface": -1})


def test_the_geometry_route_moves_the_interfaces_and_fills_the_layer(strata, prem):
    faces = {"moho": -1, "surface": 0}
    m = with_strata(prem, strata, interfaces=faces)
    assert type(m) is type(prem) and m.citation == prem.citation + ("S.",)
    check_geometry(m.geometry)
    check_model(m)
    # the mapping is layer_linear over exactly the reliefs `reliefs` gives
    shifts = reliefs(strata, prem, interfaces=faces)
    stack = m.geometry.mapping.h.reliefs
    b = prem.skeleton.boundaries
    for name, h in shifts.items():
        j = int(np.argmin(np.abs(b - prem.geometry.interface(name).radius)))
        assert np.array_equal(stack[j].values, h.values)
    assert sum(h is not None for h in stack) == 2
    again = prem.stretched(layer_linear(prem.skeleton, stack))
    rng = np.random.default_rng(0)
    X = cartesian_points(rng.uniform(5e6, R, 200), np.arccos(rng.uniform(-1, 1, 200)),
                         rng.uniform(0, 2 * np.pi, 200))
    assert np.array_equal(again.geometry.mapping(X), m.geometry.mapping(X))
    # the moved interfaces lie on the surfaces at the nodes
    t, p = np.meshgrid(GRID.colatitudes, GRID.longitudes, indexing="ij")
    for name, k in (("moho", -1), ("surface", 0)):
        r = prem.geometry.interface(name).radius
        moved = np.linalg.norm(m.geometry.mapping(cartesian_points(np.full(t.shape, r),
                                                                   t, p)), axis=-1)
        assert np.allclose(moved, strata.surfaces[k].radius(t, p), rtol=0, atol=1e-6)
    # the crust: the thickness-weighted strata, constant in radius
    crust = m.layer(prem.geometry.interface("moho").between[1])
    a, c = crust.interval
    radii = [s.radius(t, p) for s in strata.surfaces]
    w = [radii[k] - radii[k + 1] for k in range(3)]
    rho = (w[0] * 2100.0 + w[1] * 2700.0 + w[2] * 0.5 * (2850.0 + 3050.0)) / sum(w)
    vs = (w[0] * 1200.0 + w[1] * 3500.0 + w[2] * 3900.0) / sum(w)
    for r in (a, 0.3 * a + 0.7 * c, c):
        assert np.allclose(crust["rho"](r, t, p), rho, rtol=1e-12)
        assert np.allclose(crust["vsv"](r, t, p), vs, rtol=1e-12)
        assert np.allclose(crust["vsh"](r, t, p), vs, rtol=1e-12)
        assert np.allclose(crust["L"](r, t, p), rho * vs ** 2, rtol=1e-12)
    assert crust["qmu"] is prem.layer(crust.index)["qmu"]       # not the strata's
    below = prem.layer(crust.index - 1)
    assert m.layer(below.index)["vsv"] is below["vsv"]
    # one stratum's own values
    one = with_strata(prem, strata, interfaces=faces, take={crust.index: "upper_crust"})
    assert np.allclose(one.layer(crust.index)["vpv"](a, t, p), 6000.0, rtol=1e-14)


def test_the_geometry_route_refuses_by_name(strata, prem):
    whole = pm.PREM(ocean=False)
    with pytest.raises(ValueError, match=r"has 2 layers.*\['d15'\].*coarsened"):
        with_strata(whole, strata, interfaces={"moho": -1, "surface": 0})
    with pytest.raises(ValueError, match="take= names layer 3"):
        with_strata(prem, strata, interfaces={"moho": -1, "surface": 0}, take={3: "x"})
    with pytest.raises(ValueError, match="outside its strata"):
        with_strata(prem, strata, interfaces={"moho": -1, "surface": "upper_crust"},
                    take={10: "sediments"})
    with pytest.raises(ValueError, match="needs interfaces="):
        with_strata(prem, strata)
    with pytest.raises(ValueError, match="spacing= belongs to the fields route"):
        with_strata(prem, strata, interfaces={"moho": -1}, spacing=1e3)
    with pytest.raises(ValueError, match="interfaces= belongs to the geometry route"):
        with_strata(prem, strata, route="fields", interfaces={"moho": -1}, spacing=1e3)
    with pytest.raises(ValueError, match="route must be one of"):
        with_strata(prem, strata, route="mesh")
    with pytest.raises(ValueError, match="do not all hold 'eta'"):
        with_strata(prem, strata, interfaces={"moho": -1, "surface": 0},
                    properties={"eta": "eta"})


def test_the_fields_route_samples_the_strata_on_a_spherical_skeleton(strata):
    prem = pm.PREM(ocean=False)
    m = with_strata(prem, strata, route="fields", spacing=1e3)
    assert m.geometry.is_identity and m.citation[-1] == "S."
    check_model(m)
    filled = [lay.index for lay in m.layers
              if isinstance(lay["vsv"], pm.GriddedField)]
    assert filled == [prem.layer(n).index
                      for n in ("lid", "lower_crust", "upper_crust")]
    lay = m.layer("upper_crust")
    a, c = lay.interval
    t, p = nodes()
    radii = [s.radius(t, p) for s in strata.surfaces]
    for r in lay["vsv"].breaks:
        got = lay["vsv"](r, t, p)
        want = np.where(r >= radii[0], prem.layer(lay.index)["vsv"](r),
                        np.where(r >= radii[1], 1200.0,
                                 np.where(r >= radii[2], 3500.0,
                                          np.where(r >= radii[3], 3900.0, 4500.0))))
        assert np.allclose(got, want, rtol=1e-12), r
    # rho linear in radius through the lower crust, at a node of its own
    deep = m.layer("lid")
    r = deep["rho"].breaks[-3]
    inside = (r >= radii[3]) & (r < radii[2])
    s = (radii[2] - r) / (radii[2] - radii[3])
    want = np.where(inside, 2850.0 + s * 200.0, 3300.0)
    assert np.allclose(deep["rho"](r, t, p), want, rtol=1e-12)
    # in the model's units, and on the layers asked for
    nd = prem.nondimensionalised()
    only = with_strata(nd, strata, route="fields", spacing=1e3 / nd.scales.length,
                       layers=["upper_crust"])
    assert not isinstance(only.layer("lid")["vsv"], pm.GriddedField)
    k = nd.scales.factor(nd.spec("vsv").dimensions)
    r = only.layer("upper_crust")["vsv"].breaks[2]
    assert np.allclose(k * only.layer("upper_crust")["vsv"](r, t, p),
                       m.layer("upper_crust")["vsv"](r * nd.scales.length, t, p),
                       rtol=1e-12)
    with pytest.raises(ValueError, match="spacing="):
        with_strata(prem, strata, route="fields")


@pytest.mark.netcdf
def test_a_model_with_strata_as_geometry_writes_and_reads_back(strata, prem, tmp_path):
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.modelfile import read_model, write_model
    m = with_strata(prem, strata, interfaces={"moho": -1, "surface": 0})
    path = write_model(tmp_path / "strata.nc", m, grid=GRID)
    pm.testing.check_model_file(path, m)
    back = read_model(path)
    assert back.citation == m.citation
    rng = np.random.default_rng(5)
    X = cartesian_points(rng.uniform(6.2e6, 6.36e6, 100),
                         np.arccos(rng.uniform(-1, 1, 100)), rng.uniform(0, 6.3, 100))
    assert np.allclose(back.geometry.mapping(X), m.geometry.mapping(X), rtol=0,
                       atol=1e-6)
