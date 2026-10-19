"""Maps of sphere slices on cartopy's projections: the cells around the
nodes, the seam closed and the poles reached on every kind of grid."""
import io

import numpy as np
import pytest

from planetmodel import (AnalyticField, AngularGrid, LayeredIsotropicElastic,
                         driscoll_healy, equiangular, gauss_legendre)
from planetmodel import slices
from planetmodel.plotting import DIVERGING, cell_edges, sphere_map

pytestmark = pytest.mark.maps

GRIDS = [gauss_legendre(6), driscoll_healy(5), equiangular(9, 16, poles=True),
         equiangular(8, 16),
         AngularGrid(np.array([0.3, 1.0, 1.9, 2.8]),
                     np.array([0.2, 1.5, 3.0, 4.4, 5.9]))]


@pytest.mark.parametrize("grid", GRIDS, ids=lambda g: g.kind)
def test_the_cells_close_the_seam_and_reach_the_poles(grid):
    lon, lat = cell_edges(grid)
    assert lon.size == grid.nphi + 1 and lat.size == grid.ntheta + 1
    assert np.isclose(lon[-1] - lon[0], 360.0)
    assert lat[0] == 90.0 and lat[-1] == -90.0
    assert np.all(np.diff(lon) > 0) and np.all(np.diff(lat) < 0)
    # each node inside its own cell
    phi = np.degrees(grid.longitudes)
    assert np.all((lon[:-1] < phi) & (phi < lon[1:]))
    nodes = 90.0 - np.degrees(grid.colatitudes)
    assert np.all((lat[:-1] >= nodes) & (nodes >= lat[1:]))


def _model():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    return m.with_field(1, "vs", AnalyticField(
        (0.5, 1.0), lambda r, t, p: 3.0 + 0.2 * r * np.sin(t) * np.cos(p),
        name="vs"),
        replace=True)


def test_a_map_draws_the_slice_on_a_projection():
    ccrs = pytest.importorskip("cartopy.crs",
                               reason="needs the planetmodel[maps] extra")
    plt = pytest.importorskip("matplotlib.pyplot")
    m = _model()
    grid = gauss_legendre(8)
    s = slices.sphere(m, "vs", radius=0.8, grid=grid, relative_to="mean")
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    mesh = sphere_map(ax, s, coastlines=True, colorbar=True)
    assert np.allclose(np.asarray(mesh.get_array()).reshape(s.values.shape), s.values)
    lo, hi = mesh.get_clim()
    assert lo == -hi and mesh.get_cmap().name == DIVERGING
    # coastlines asked for, not fetched: nothing is drawn here
    assert any(type(a).__name__ == "FeatureArtist" for a in ax.get_children())
    assert fig.axes[-1].get_xlabel() == s.label
    plt.close(fig)
    # without coastlines it draws, on any projection, offline
    for projection in (ccrs.Mollweide(central_longitude=180.0),
                       ccrs.Orthographic(30, 20), ccrs.PlateCarree()):
        fig, ax = plt.subplots(subplot_kw={"projection": projection})
        sphere_map(ax, slices.sphere(m, "vs", radius=0.8, grid=driscoll_healy(6)))
        fig.savefig(io.BytesIO(), format="png")
        plt.close(fig)


def test_a_kernel_s_map_is_symmetric_clipped_and_normalised():
    ccrs = pytest.importorskip("cartopy.crs",
                               reason="needs the planetmodel[maps] extra")
    plt = pytest.importorskip("matplotlib.pyplot")
    from planetmodel import GriddedFunction
    grid = gauss_legendre(16)
    def kernel(t, p):
        return np.cos(t) + 40.0 * np.exp(-((t - 1.0) ** 2 + (p - 2.0) ** 2) / 0.01)

    spike = GriddedFunction.sampled(kernel, grid, lateral="linear")
    s = slices.surface(_model(), spike, 1, name="K")
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    mesh = sphere_map(ax, s, diverging=True, clip=95, colorbar=True)
    assert np.isclose(mesh.get_clim()[1], np.percentile(np.abs(s.values), 95))
    assert mesh.get_clim()[0] == -mesh.get_clim()[1] and mesh.colorbar.extend == "max"
    plt.close(fig)
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    mesh = sphere_map(ax, s, normalised=True, diverging=True, colorbar=True)
    assert np.allclose(np.asarray(mesh.get_array()).reshape(s.values.shape),
                       s.values / np.max(np.abs(s.values)))
    assert fig.axes[-1].get_xlabel() == "K, on interface 1, normalised"
    plt.close(fig)


def test_a_map_is_refused_on_axes_other_than_cartopys():
    plt = pytest.importorskip("matplotlib.pyplot")
    s = slices.sphere(_model(), "vs", radius=0.8, grid=gauss_legendre(4))
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="cartopy GeoAxes"):
        sphere_map(ax, s)
    geo = fig.add_subplot(2, 1, 2, projection="mollweide")
    with pytest.raises(ValueError, match="maps extra"):
        sphere_map(geo, s)
    plt.close(fig)
