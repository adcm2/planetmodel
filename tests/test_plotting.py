"""The radial profile helpers: radius upward, a joiner at every jump."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib",
                                 reason="needs the planetmodel[plot] extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from planetmodel import PREM, LayeredIsotropicElastic, constant_field  # noqa: E402
from planetmodel.plotting import profile, radial_profile  # noqa: E402


def test_radius_is_vertical_and_every_jump_gets_a_joiner():
    m = LayeredIsotropicElastic([0.0, 0.5, 0.8, 1.0], rho=[3.0, 2.0, 2.0],
                                vp=[2.0, 2.0, 2.0], vs=[1.0, 0.0, 1.0])
    fig, ax = plt.subplots()
    lines = radial_profile(ax, m, "rho", scale=2.0, value_scale=10.0, label="d")
    # three segments and two joiners (the second jump is zero-height, still drawn)
    joiners = [ln for ln in lines if ln.get_label() == "_joiner"]
    segments = [ln for ln in lines if ln.get_label() != "_joiner"]
    assert len(segments) == 3 and len(joiners) == 2
    x, y = segments[0].get_data()
    assert np.allclose(x, 30.0) and y[0] == 0.0 and np.isclose(y[-1], 1.0)
    jx, jy = joiners[0].get_data()
    assert np.allclose(jy, 1.0) and list(jx) == [30.0, 20.0]
    assert segments[0].get_label() == "d" and segments[1].get_label().startswith("_")
    assert len({ln.get_color() for ln in lines}) == 1
    assert all(ln.get_linestyle() == "-" for ln in lines)
    # the guides at the four boundaries sit beneath every profile line
    guides = [ln for ln in ax.get_lines() if ln not in lines]
    assert sorted(ln.get_ydata()[0] for ln in guides) == [0.0, 1.0, 1.6, 2.0]
    assert all(g.get_zorder() < ln.get_zorder() for g in guides for ln in lines)
    plt.close(fig)
    fig, ax = plt.subplots()
    radial_profile(ax, m, "rho", boundaries=False)
    assert len(ax.get_lines()) == 5
    plt.close(fig)


def test_a_missing_layer_is_a_gap_without_a_joiner():
    m = PREM(ocean=False)
    fig, ax = plt.subplots()
    lines = radial_profile(ax, m, "qmu", joiners=True)
    held = m.layers_with("qmu")
    joiners = [ln for ln in lines if ln.get_label() == "_joiner"]
    segments = [ln for ln in lines if ln.get_label() != "_joiner"]
    assert len(segments) == len(held)
    # no joiner across the outer core, which lacks qmu
    cmb = m.geometry.interface("cmb").radius
    icb = m.geometry.interface("icb").radius
    assert all(ln.get_data()[1][0] not in (icb, cmb) for ln in joiners)
    plt.close(fig)


def test_refusals():
    m = LayeredIsotropicElastic([0.0, 1.0], rho=[1.0], vp=[1.0], vs=[1.0])
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="complex"):
        profile(ax, [constant_field((0.0, 1.0), 1.0 + 1j)])
    with pytest.raises(ValueError, match="radial field of rank 0"):
        profile(ax, [m.elastic_moduli(0)])
    plt.close(fig)


# -- slices drawn -----------------------------------------------------------

from planetmodel import AnalyticField, flattening, gauss_legendre  # noqa: E402
from planetmodel import slices  # noqa: E402
from planetmodel.plotting import (BOUNDARY_STYLE, DIVERGING, SEQUENTIAL,  # noqa: E402
                                  colour_scale, section)


def _lateral(r, t, p):
    return 3.0 + 0.3 * r * np.cos(t) + 0.2 * r * np.sin(t) * np.cos(p)


@pytest.fixture
def layered():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    return m.with_field(1, "vs", AnalyticField((0.5, 1.0), _lateral, name="vs"),
                        replace=True)


def test_a_profile_slice_is_drawn_layer_by_layer_with_its_step(layered):
    prof = slices.profile(layered, "rho", 1.0, 2.0, n=7)
    fig, ax = plt.subplots()
    lines = profile(ax, prof, scale=2.0)
    segments = [ln for ln in lines if ln.get_label() != "_joiner"]
    joiners = [ln for ln in lines if ln.get_label() == "_joiner"]
    assert len(segments) == 2 and len(joiners) == 1
    x, y = segments[1].get_data()
    assert np.allclose(x, prof.layers[1].values)
    assert np.allclose(y, 2.0 * prof.layers[1].radius)
    assert list(joiners[0].get_data()[0]) == [10.0, 4.0]
    mean = slices.mean_profile(layered, "vs", mean_grid=gauss_legendre(8), n=5)
    assert len(profile(fig.add_subplot(2, 1, 2), mean, joiners=False)) == 2
    plt.close(fig)


def test_a_section_is_a_wedge_of_the_annulus_on_polar_axes(layered):
    sec = slices.section(layered, "vs", (0.6, 0.2), azimuth=1.0, length=1.2, n=5,
                         m=31, relative_to="mean", mean_grid=gauss_legendre(10))
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    meshes = section(ax, sec, scale=10.0, colorbar=True)
    assert len(meshes) == 2
    # the layers share one scale, symmetric about zero for a departure
    lims = {mesh.get_clim() for mesh in meshes}
    assert len(lims) == 1 and np.isclose(-lims.pop()[0], meshes[0].get_clim()[1])
    assert meshes[0].get_cmap().name == DIVERGING
    coords = meshes[1].get_coordinates()
    assert np.allclose(coords[..., 0], sec.distance[None, :] - 0.6)
    assert np.allclose(coords[..., 1], 10.0 * sec.layers[1].radius)
    assert np.allclose(np.asarray(meshes[1].get_array()).reshape(5, 31),
                       sec.layers[1].values)
    assert np.isclose(ax.get_thetamin(), -np.degrees(0.6))
    assert np.isclose(ax.get_thetamax(), np.degrees(0.6))
    assert np.allclose(ax.get_ylim(), (0.0, 10.0))
    # the interfaces are faint arcs, and the colour bar names the slice
    arcs = [ln for ln in ax.get_lines() if ln.get_color() == BOUNDARY_STYLE["color"]]
    assert len(arcs) == 4
    assert fig.axes[-1].get_ylabel() == sec.label
    plt.close(fig)


def test_a_whole_circle_is_a_disc_and_the_body_can_be_drawn_as_it_is(layered):
    full = slices.section(layered, "rho", (0.6, 0.0), azimuth=0.5, length=2 * np.pi,
                          n=4, m=40)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    section(ax, full)
    assert np.isclose(ax.get_thetamax() - ax.get_thetamin(), 360.0)
    assert section(ax, full)[0].get_cmap().name == SEQUENTIAL
    plt.close(fig)
    flat = layered.stretched(flattening(0.1))
    sec = slices.section(flat, "rho", (0.3, 0.0), end=(2.0, 0.0), n=4, m=21)
    fig, ax = plt.subplots()
    meshes = section(ax, sec, physical=True)
    xy = meshes[0].get_coordinates()
    assert np.allclose(np.hypot(xy[..., 0], xy[..., 1]),
                       np.linalg.norm(sec.layers[0].physical, axis=-1))
    assert ax.get_aspect() == 1.0
    with pytest.raises(ValueError, match="polar"):
        section(ax, sec)
    polar = fig.add_subplot(2, 1, 2, projection="polar")
    with pytest.raises(ValueError, match="polar"):
        section(polar, sec, physical=True)
    at = slices.section(flat, "rho", (0.3, 0.0), end=(2.0, 0.0), n=4, m=21,
                        at="physical")
    assert len(section(polar, at)) == 2
    with pytest.raises(ValueError, match="physical points already"):
        section(ax, at, physical=True)
    plt.close(fig)


def test_the_colour_scale_is_the_callers_or_by_what_the_values_are(layered):
    dep = slices.sphere(layered, "vs", radius=0.8, grid=gauss_legendre(4),
                        relative_to="mean")
    cmap, lo, hi = colour_scale([dep.values], dep.shown)
    assert cmap == DIVERGING and lo == -hi == -np.max(np.abs(dep.values))
    plain = slices.sphere(layered, "rho", radius=0.8, grid=gauss_legendre(4))
    cmap, lo, hi = colour_scale([plain.values], plain.shown)
    assert cmap == SEQUENTIAL and lo < 4.0 < hi
    assert colour_scale([plain.values], plain.shown, cmap="magma", vmin=0.0,
                        vmax=9.0) == ("magma", 0.0, 9.0)
    with pytest.raises(ValueError, match="no finite"):
        colour_scale([np.full(3, np.nan)], plain.shown)
