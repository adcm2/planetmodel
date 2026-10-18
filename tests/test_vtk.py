"""VTK files of a model: parsed back, their points and arrays the model's,
the seam closed, the poles present, a failed write leaving nothing."""
import base64
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest

from planetmodel import (STRESS, VECTOR, AnalyticField, LayeredIsotropicElastic,
                         flattening, gauss_legendre)
from planetmodel.frames import cartesian_points, voigt_to_tensor
from planetmodel.plotting import write_vtk


def decode(text, dtype):
    """VTK's compressed binary: a UInt64 header, then the zlib blocks, each
    encoded in base64 on its own."""
    text = text.strip()
    first = np.frombuffer(base64.b64decode(text[:32]), dtype="<u8")
    blocks = int(first[0])
    head_bytes = 8 * (3 + blocks)
    head_chars = 4 * ((head_bytes + 2) // 3)
    header = np.frombuffer(base64.b64decode(text[:head_chars]), dtype="<u8")
    data = base64.b64decode(text[head_chars:])
    out, at = [], 0
    for size in header[3:]:
        out.append(zlib.decompress(data[at:at + int(size)]))
        at += int(size)
    return np.frombuffer(b"".join(out), dtype=dtype)


def read_block(path):
    root = ET.parse(path).getroot()
    assert root.get("compressor") == "vtkZLibDataCompressor"
    assert root.get("header_type") == "UInt64"
    grid = root.find("StructuredGrid")
    nx, ny, nz = (int(e) + 1 for e in grid.get("WholeExtent").split()[1::2])
    points = decode(grid.find("Piece/Points/DataArray").text, "<f8")
    points = points.reshape(nz, ny, nx, 3)
    arrays = {}
    for da in grid.findall("Piece/PointData/DataArray"):
        dtype = "<f8" if da.get("type") == "Float64" else "<i4"
        k = int(da.get("NumberOfComponents"))
        arrays[da.get("Name")] = decode(da.text, dtype).reshape(nz, ny, nx, k)
    return points, arrays


def stress(r, t, p):
    X = cartesian_points(r, t, p)
    return np.stack([1 + X[..., 0], 2 + X[..., 1], 3 + X[..., 2], 0.1 * X[..., 0],
                     0.2 * X[..., 1], 0.3 * X[..., 2]], axis=-1)


@pytest.fixture
def model():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    m = m.with_field(1, "vs", AnalyticField(
        (0.5, 1.0), lambda r, t, p: 3.0 + 0.2 * r * np.sin(t) * np.cos(p), name="vs"),
        replace=True)
    m = m.with_field(1, "flow", AnalyticField(
        (0.5, 1.0), lambda r, t, p: cartesian_points(r, t, p)[..., [1, 0, 2]],
        character=VECTOR, frame="cartesian"))
    return m.with_field(1, "sigma", AnalyticField((0.5, 1.0), stress, character=STRESS,
                                                  frame="cartesian"))


def test_the_files_hold_the_models_points_and_fields(model, tmp_path):
    grid = gauss_legendre(4)
    out = write_vtk(tmp_path / "earth", model, grid=grid, n=5, physical=False,
                    departures={"vs": "mean"}, mean_grid=gauss_legendre(8))
    assert out == tmp_path / "earth.vtm"
    sets = ET.parse(out).getroot().findall("vtkMultiBlockDataSet/DataSet")
    assert [d.get("file") for d in sets] == ["earth/00_layer.vts", "earth/01_layer.vts"]
    points, arrays = read_block(tmp_path / sets[1].get("file"))
    # the pole rows the grid lacks are added, the seam closed by a column
    theta = np.concatenate([[0.0], grid.colatitudes, [np.pi]])
    phi = np.append(grid.longitudes, grid.longitudes[0] + 2 * np.pi)
    r = np.linspace(0.5, 1.0, 5)
    R, T, P = np.meshgrid(r, theta, phi, indexing="ij")
    assert points.shape == (5, theta.size, phi.size, 3)
    assert np.allclose(points, cartesian_points(R, T, P))
    assert np.allclose(arrays["vs"][..., 0], 3.0 + 0.2 * R * np.sin(T) * np.cos(P))
    assert np.allclose(arrays["flow"], cartesian_points(R, T, P)[..., [1, 0, 2]])
    assert arrays["sigma"].shape[-1] == 9
    assert np.allclose(arrays["sigma"].reshape(R.shape + (3, 3)),
                       voigt_to_tensor(stress(R, T, P), rank=2))
    assert np.array_equal(arrays["layer"][..., 0], np.ones(R.shape, dtype=int))
    assert "vs_percent_from_mean" in arrays
    for name in ("vs", "flow", "sigma", "vs_percent_from_mean"):
        assert np.array_equal(arrays[name][:, :, -1], arrays[name][:, :, 0])
    inner, inner_arrays = read_block(tmp_path / sets[0].get("file"))
    assert "flow" not in inner_arrays and np.allclose(inner_arrays["rho"], 10.0)
    assert np.allclose(np.linalg.norm(inner[-1], axis=-1), 0.5)


def test_the_points_are_physical_by_default(model, tmp_path):
    flat = model.stretched(flattening(0.05))
    out = write_vtk(tmp_path / "flat.vtm", flat, grid=gauss_legendre(3), n=3)
    points, _ = read_block(tmp_path / "flat" / "01_layer.vts")
    grid = gauss_legendre(3)
    theta = np.concatenate([[0.0], grid.colatitudes, [np.pi]])
    phi = np.append(grid.longitudes, grid.longitudes[0] + 2 * np.pi)
    R, T, P = np.meshgrid(np.linspace(0.5, 1.0, 3), theta, phi, indexing="ij")
    assert np.allclose(points, flat.geometry.mapping(cartesian_points(R, T, P)))
    assert out.exists()


def test_a_failed_write_leaves_nothing_and_an_earlier_write_stands(model, tmp_path):
    grid = gauss_legendre(3)
    write_vtk(tmp_path / "m", model, grid=grid, n=3)
    before = (tmp_path / "m" / "01_layer.vts").read_bytes()
    with pytest.raises(ValueError, match="mean_grid"):
        write_vtk(tmp_path / "m", model, grid=grid, n=3, departures={"vs": "mean"})
    assert (tmp_path / "m" / "01_layer.vts").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m", "m.vtm"]
    # rewritten, the files are replaced and nothing is left beside them
    write_vtk(tmp_path / "m", model, grid=grid, n=4)
    assert (tmp_path / "m" / "01_layer.vts").read_bytes() != before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m", "m.vtm"]
    with pytest.raises(KeyError):
        write_vtk(tmp_path / "x", model, fields=["nonesuch"])
    with pytest.raises(ValueError, match="'mean' or 'reference'"):
        write_vtk(tmp_path / "x", model, departures={"vs": "median"})
