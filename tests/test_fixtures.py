"""The fixtures the C++ readers are held to are written and validate."""
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

SCRIPT = Path(__file__).resolve().parents[1] / "cpp" / "tests" / "make_fixtures.py"


def test_the_fixtures_are_written_and_validate(tmp_path):
    spec = importlib.util.spec_from_file_location("make_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = module.write_fixtures(tmp_path, meshes=False)
    names = {p.name for p in written}
    assert {"model_prem.nc", "model_parts.nc", "model_stacked.nc",
            "model_flattened.nc", "case_radial_polynomial.nc",
            "case_lateral_driscoll_healy_cubic.nc"} <= names
    from planetmodel.modelfile import read_model
    assert read_model(tmp_path / "model_parts.nc").nlayers == 13
    with netCDF4.Dataset(tmp_path / "case_lateral_custom_offset_cubic.nc") as ds:
        assert ds.method == "cubic"
        assert ds.variables["expected"].shape == ds.variables["query_colatitude"].shape
    assert {"constants.txt", "json_corners.json", "json_corners.json.walk"} <= names
    assert not (tmp_path / "manifests").exists()


@pytest.mark.gmsh
@pytest.mark.mfem
def test_the_manifests_are_written_with_their_walks(tmp_path):
    """The manifests the C++ reader is held to, one per way the Python
    writes one, each valid and walked."""
    pytest.importorskip("gmsh")
    pytest.importorskip("mfem.ser")
    from planetmodel.mesh3d import manifest
    spec = importlib.util.spec_from_file_location("make_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = module.manifests(tmp_path)
    manifests = [p for p in written if p.suffix == ".json"]
    assert {p.parent.name for p in manifests} == {
        "mesher", "mesher_refined", "mesh_only", "hollow_shell", "referential",
        "physical", "nondimensional"}
    for path in manifests:
        manifest.read(path)          # which validates
        assert path.with_name(path.name + ".walk").stat().st_size > 0
