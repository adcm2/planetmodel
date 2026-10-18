"""Write the fixtures the C++ readers are held to: model files and
interpolation cases, from planetmodel's Python, which defines both.

Run as a script it writes into `fixtures/` beside it; `write_fixtures`
writes into any directory and returns the paths.  The fixtures are
written in CI, not committed.

**Model files** (`model_*.nc`), each with the model's fields at its own
nodes: PREM (radial, one part); PREM with a relative perturbation of
`vsv` over the lower mantle; a two-layer model with a laterally varying
density and a vector field; PREM in three parts on three kinds of grid;
a model placed by boundary reliefs (the `geometry` group); and one
placed by a flattening (the `displacement` field).

**Interpolation cases** (`case_lateral_*.nc`, `case_radial_*.nc`), one
netCDF file each: for a lateral case the grid (`colatitude`,
`longitude`, radians), its `kind`, the `values` at its nodes and the
`poles`, the `method`, the query points `query_colatitude` and
`query_longitude`, and the `expected` values with their derivatives
`expected_dtheta` and `expected_dphi`; for a radial case the element
`breaks`, `ngll`, the `values` at the nodes (element by element), the
`method`, the query radii and the `expected` values and derivatives.
Every case covers the poles, the longitude seam and points between
nodes.  The local rules only: the C++ side does no spherical-harmonic
work.

**JSON** (`json_*.json`, and `manifests/<case>/*.json`, the manifests
the mesher and the MFEM exports write on unit geometries, which need
gmsh and PyMFEM and are left out with a note without them), each with a
walk beside it (`<file>.walk`): one line per value in document order,
its path from the root as the C++ reader writes it, a tab, its type
(`null`, `bool`, `number`, `string`, `array`, `object`), a tab, and its
value as Python's `json` reads it (`true` or `false`; a number written
to round-trip; a string's UTF-8 bytes in hexadecimal; the length of an
array or an object).

**The facts the C++ restates** (`constants.txt`), one per line, a key
and its value separated by spaces: the version, `G_SI`,
`EARTH_MEAN_DENSITY`, and for each of a few scales, `scales <label>`
with its four values, `factor <label> <m l t T> <value>` and
`unit <label> <m l t T> <string>` for dimensions across the vocabulary,
floats written to round-trip exactly.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import planetmodel as pm
from planetmodel import units
from planetmodel.lateral import LateralInterpolant
from planetmodel.mesh1d import Mesh1D
from planetmodel.modelfile import validate_structure, write_model


def smooth(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """A smooth function on the sphere, the exponential of a Cartesian one."""
    st = np.sin(theta)
    return np.exp(st * np.cos(phi) + 0.5 * st * np.sin(phi) - 0.3 * np.cos(theta))


def _queries(grid: pm.AngularGrid, rng: np.random.Generator
             ) -> tuple[np.ndarray, np.ndarray]:
    """Random points, the poles, the seam and the nodes' own positions."""
    t = list(rng.uniform(0.0, np.pi, 40)) + [0.0, 0.0, np.pi, np.pi]
    p = list(rng.uniform(-np.pi, 3.0 * np.pi, 40)) + [0.3, 4.0, 1.1, 5.9]
    phi0 = float(grid.longitudes[0])
    t += [1.0, 1.0, float(grid.colatitudes[1])]
    p += [phi0 - 1e-9, phi0 + 1e-9, float(grid.longitudes[2])]
    return np.array(t), np.array(p)


def _write_case(path: Path, arrays: dict[str, np.ndarray],
                attributes: dict[str, object]) -> Path:
    """One case file: every array on its own dimension of its length."""
    import netCDF4
    with netCDF4.Dataset(path, "w") as ds:
        for key, value in attributes.items():
            ds.setncattr(key, value)
        for name, value in arrays.items():
            value = np.asarray(value, dtype=float)
            dims = tuple(f"{name}_{k}" for k in range(value.ndim))
            for d, n in zip(dims, value.shape):
                ds.createDimension(d, n)
            ds.createVariable(name, "f8", dims)[:] = value
    return path


def lateral_cases(directory: Path) -> list[Path]:
    """The lateral rules on every kind of grid."""
    rng = np.random.default_rng(20261018)
    offset = pm.AngularGrid(np.linspace(0.2, 2.9, 9),
                            np.sort(rng.uniform(0.3, 6.1, 11)))
    grids = {"gauss_legendre": pm.gauss_legendre(6),
             "driscoll_healy": pm.driscoll_healy(5),
             "equiangular": pm.equiangular(9, 16),
             "equiangular_poles": pm.equiangular(9, 16, poles=True),
             "custom_offset": offset}
    out = []
    for label, grid in grids.items():
        values = smooth(grid.colatitudes[:, None], grid.longitudes[None, :])
        poles = smooth(np.array([0.0, np.pi]), np.zeros(2))
        t, p = _queries(grid, rng)
        for method in ("linear", "cubic"):
            f = LateralInterpolant(values, grid, method=method, poles=poles)
            dt, dp = f.gradient(t, p)
            out.append(_write_case(
                directory / f"case_lateral_{label}_{method}.nc",
                {"colatitude": grid.colatitudes, "longitude": grid.longitudes,
                 "values": values, "poles": poles, "query_colatitude": t,
                 "query_longitude": p, "expected": f(t, p),
                 "expected_dtheta": dt, "expected_dphi": dp},
                {"kind": grid.kind, "method": method, "mean": float(f.mean())}))
    return out


def radial_cases(directory: Path) -> list[Path]:
    """The radial rules on elements of unequal widths."""
    mesh = Mesh1D([0.0, 0.35, 0.5, 1.0], ngll=5)
    values = np.sin(3.0 * mesh.r) + mesh.r ** 2
    r = np.concatenate([np.linspace(0.0, 1.0, 41), mesh.left])
    out = []
    for method in ("polynomial", "linear"):
        pp = mesh.to_ppoly(values, method=method)
        out.append(_write_case(
            directory / f"case_radial_{method}.nc",
            {"breaks": mesh.breakpoints, "values": values, "query_radius": r,
             "expected": pp(r), "expected_dr": pp.derivative()(r)},
            {"method": method, "ngll": np.int32(mesh.ngll)}))
    return out


def model_files(directory: Path) -> list[Path]:
    """The model files of the module docstring."""
    prem = pm.PREM()
    lo, hi = 3480e3, 5701e3

    def dlnvs(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 0.02 * np.sin(t) ** 2 * np.cos(2 * p) * (r - lo) / (hi - lo)

    perturbed = pm.perturbed(prem, "vsv", pm.AnalyticField((lo, hi), dlnvs),
                             kind="relative")
    base = ["rho", "vpv", "vsv", "vph", "vsh", "eta"]
    two = pm.LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0],
                                     vp=[9.0, 7.0], vs=[4.0, 3.0])
    grid = pm.gauss_legendre(8)

    def rho3d(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 4.0 + 0.2 * r * smooth(t, p)

    def wind(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        r, t, p = np.broadcast_arrays(r, t, p)
        return np.stack([r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p),
                         r * np.cos(t) + 1.0], axis=-1)

    lateral = two.with_field(1, "rho", pm.GriddedField.sampled(
        pm.AnalyticField((0.5, 1.0), rho3d, character=pm.DENSITY), grid,
        breaks=[0.5, 0.75, 1.0], ngll=4, lateral="cubic"), replace=True)
    lateral = lateral.with_field(1, "flow", pm.GriddedField.sampled(
        pm.AnalyticField((0.5, 1.0), wind, character=pm.VECTOR, frame="cartesian"),
        grid, ngll=3, lateral="linear"))
    parts = {i: pm.gauss_legendre(4) for i in range(2)}
    dh, eq = pm.driscoll_healy(5), pm.equiangular(9, 16, poles=True)
    parts.update({i: dh for i in range(2, 9)})
    parts.update({i: eq for i in range(9, 13)})

    def bump(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 0.02 * np.sin(t) ** 3 * np.cos(3 * p) + 0.01 * np.cos(t)

    relief = pm.GriddedFunction.sampled(bump, pm.gauss_legendre(16))
    stacked = two.stretched(pm.layer_linear(two.skeleton, [None, relief, relief]))
    flattened = two.stretched(pm.flattening(0.01))
    return [
        write_model(directory / "model_prem.nc", prem, grid=pm.gauss_legendre(2)),
        write_model(directory / "model_prem_perturbed.nc", perturbed,
                    grid=pm.gauss_legendre(8), fields=base,
                    interpolation={"vsv": ("polynomial", "cubic")}),
        write_model(directory / "model_lateral.nc", lateral, grid=grid),
        write_model(directory / "model_parts.nc", perturbed, grid=parts,
                    fields=base),
        write_model(directory / "model_stacked.nc", stacked,
                    grid=pm.gauss_legendre(4)),
        write_model(directory / "model_flattened.nc", flattened,
                    grid=pm.gauss_legendre(8)),
    ]


#: The JSON documents of the walk fixtures, as bytes on disk: the
#: grammar's corners, and a nesting deep but within the reader's limit.
_JSON_DOCUMENTS = {
    "corners": (
        '{"unicode": "caf\\u00e9 \\ud83d\\ude00", "raw": "caf\u00e9 \U0001f600",\n'
        ' "escapes": "a\\"b\\\\c\\/d\\b\\f\\n\\r\\t",\n'
        ' "numbers": [0, -0.0, 1e300, 5e-324, 0.1, 123456789012, -1.5E-3, 1E+2,\n'
        '             2e-1],\n'
        ' "empty": {}, "list": [], "nested": [[[{"x": null}]]],\n'
        ' "keys": {"a b": 1, "x.y": 2, "": 3, "q\\"uote": 4, "9lives": 5, "_ok9": 6},\n'
        ' "bools": [true, false], "spaced": [ 1 ,\t2\r\n, 3 ]}\n'),
    "deep": "[" * 200 + "7" + "]" * 200,
    "scalar": "  -12.5e3  ",
}


def _path(path: str, key: str | int) -> str:
    """The path of an element or a member, by the C++ reader's rule."""
    if isinstance(key, int):
        return f"{path}[{key}]"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return key if not path else f"{path}.{key}"
    quoted = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'{path}["{quoted}"]'


def _walk(value: Any, path: str = "") -> list[str]:
    """The walk lines of a parsed JSON value, by the module docstring."""
    if value is None:
        lines = [f"{path}\tnull\t"]
    elif isinstance(value, bool):
        lines = [f"{path}\tbool\t{'true' if value else 'false'}"]
    elif isinstance(value, (int, float)):
        lines = [f"{path}\tnumber\t{float(value)!r}"]
    elif isinstance(value, str):
        lines = [f"{path}\tstring\t{value.encode('utf-8').hex()}"]
    elif isinstance(value, list):
        lines = [f"{path}\tarray\t{len(value)}"]
        for i, v in enumerate(value):
            lines += _walk(v, _path(path, i))
    else:
        lines = [f"{path}\tobject\t{len(value)}"]
        for k, v in value.items():
            lines += _walk(v, _path(path, k))
    return lines


def _write_walk(path: Path) -> Path:
    """The walk of the JSON document at `path`, beside it."""
    walk = path.with_name(path.name + ".walk")
    walk.write_text("\n".join(_walk(json.loads(path.read_bytes()))) + "\n",
                    encoding="utf-8")
    return walk


def json_documents(directory: Path) -> list[Path]:
    """The documents of `_JSON_DOCUMENTS`, each with its walk."""
    out = []
    for label, text in _JSON_DOCUMENTS.items():
        path = directory / f"json_{label}.json"
        path.write_bytes(text.encode("utf-8"))
        out += [path, _write_walk(path)]
    return out


def manifests(directory: Path) -> list[Path]:
    """The manifests the Python writes, each with its walk: the mesher's
    (3D, and 2D sized towards a point cloud), `export_mfem_mesh`'s with no
    model (and a hollow 2D mesh with a shell), and `export_mfem`'s of an
    elastic model with a fluid core in both deliveries, one with a model
    file beside it and one non-dimensional.  None without gmsh and
    PyMFEM."""
    try:
        import mfem.ser  # noqa: F401

        from planetmodel.mesh3d import (MeshSpec, NearPoints, Shell, UniformInterfaces,
                                        build_layered_mesh, export_mfem,
                                        export_mfem_mesh)
    except ImportError as error:
        print(f"no manifests without gmsh and PyMFEM: {error}", file=sys.stderr)
        return []
    root = directory / "manifests"
    shutil.rmtree(root, ignore_errors=True)
    model = pm.LayeredIsotropicElastic([0.0, 0.4, 0.8, 1.0], rho=[12.0, 5.0, 3.0],
                                       vp=[10.0, 9.0, 6.0], vs=[0.0, 5.0, 3.5])
    model = model.with_geometry(model.geometry.renamed(
        layers=["core", "mantle", "crust"], interfaces=["cmb", "moho", "surface"]))
    flat = model.stretched(pm.flattening(0.02))
    coarse = UniformInterfaces(0.3, 0.5, 0.5)
    hollow = pm.Geometry(pm.Skeleton([0.5, 0.8, 1.0]), layer_names=["lower", "upper"],
                         interface_names=["inner", "mid", "outer"])
    cloud = NearPoints(np.array([[1.0, 0.0]]), 0.1, 0.5, 0.3)
    written = []
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)

        def built(label: str, spec: MeshSpec) -> Any:
            return build_layered_mesh(spec, work / label / label)

        sphere = built("mesher", MeshSpec(model.geometry, coarse, dimension=3))
        oblate = built("oblate", MeshSpec(flat.geometry, coarse, dimension=3, order=2))
        shell = built("shell", MeshSpec(hollow, coarse, dimension=2,
                                        shells=(Shell(ratio=0.3),)))
        refined = built("mesher_refined", MeshSpec(model.geometry, coarse, dimension=2,
                                                   refinements=[cloud]))
        cases = {
            "mesher": sphere.manifest_path,
            "mesher_refined": refined.manifest_path,
            "mesh_only": export_mfem_mesh(sphere, work / "mesh_only" / "mesh_only"
                                          ).manifest_path,
            "hollow_shell": export_mfem_mesh(
                shell, work / "hollow_shell" / "hollow_shell").manifest_path,
            "referential": export_mfem(oblate, work / "referential" / "referential",
                                       model=flat, delivery="referential",
                                       model_file=pm.gauss_legendre(4)).manifest_path,
            "physical": export_mfem(oblate, work / "physical" / "physical", model=flat,
                                    delivery="physical").manifest_path,
            "nondimensional": export_mfem(
                sphere, work / "nondimensional" / "nondimensional",
                model=model.nondimensionalised()).manifest_path,
        }
        for label, manifest in cases.items():
            target = root / label
            shutil.copytree(manifest.parent, target)
            path = target / manifest.name
            written += [path, _write_walk(path)]
    return written


#: The dimensions the units fixture covers: every named one, and powers
#: and products the named ones do not reach.
_DIMENSIONS = (units.DIMENSIONLESS, units.MASS, units.LENGTH, units.TIME,
               units.FREQUENCY, units.DENSITY, units.VELOCITY, units.GRAVITY,
               units.MODULUS, units.VISCOSITY, units.GRAVITATIONAL_CONSTANT,
               units.TEMPERATURE, units.VELOCITY ** 2,
               units.DENSITY * units.TEMPERATURE ** -1, units.LENGTH ** -7)


def constants(directory: Path) -> list[Path]:
    """The facts of the module docstring, under the SI scales, the Earth's
    geophysical ones, and scales of no physical meaning with a
    temperature."""
    scales = {"si": pm.Scales.SI,
              "earth": pm.Scales.geophysical(6371e3),
              "odd": pm.Scales(length=3.5e4, mass=2.25e19, time=17.0,
                               temperature=273.15)}
    lines = [f"version {pm.__version__}", f"G_SI {units.G_SI!r}",
             f"EARTH_MEAN_DENSITY {units.EARTH_MEAN_DENSITY!r}"]
    for label, s in scales.items():
        lines.append(f"scales {label} {s.length!r} {s.mass!r} {s.time!r} "
                     f"{s.temperature!r}")
        for d in _DIMENSIONS:
            exps = f"{d.mass} {d.length} {d.time} {d.temperature}"
            lines.append(f"factor {label} {exps} {s.factor(d)!r}")
            lines.append(f"unit {label} {exps} {units.scaled_unit(d, s)}")
    path = directory / "constants.txt"
    path.write_text("\n".join(lines) + "\n")
    return [path]


def write_fixtures(directory: str | Path, *, meshes: bool = True) -> list[Path]:
    """Every fixture into `directory`, each model file validated; the
    manifests, which build meshes, only where `meshes` is true."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    models = model_files(directory)
    for path in models:
        validate_structure(path)
    return (models + lateral_cases(directory) + radial_cases(directory)
            + constants(directory) + json_documents(directory)
            + (manifests(directory) if meshes else []))


if __name__ == "__main__":
    here = Path(__file__).parent / "fixtures"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else here
    for written in write_fixtures(target):
        print(written)
