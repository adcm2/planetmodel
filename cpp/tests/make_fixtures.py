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
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import planetmodel as pm
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


def write_fixtures(directory: str | Path) -> list[Path]:
    """Every fixture into `directory`, each model file validated."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    models = model_files(directory)
    for path in models:
        validate_structure(path)
    return models + lateral_cases(directory) + radial_cases(directory)


if __name__ == "__main__":
    here = Path(__file__).parent / "fixtures"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else here
    for written in write_fixtures(target):
        print(written)
