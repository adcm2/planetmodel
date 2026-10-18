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
    written = module.write_fixtures(tmp_path)
    names = {p.name for p in written}
    assert {"model_prem.nc", "model_parts.nc", "model_stacked.nc",
            "model_flattened.nc", "case_radial_polynomial.nc",
            "case_lateral_driscoll_healy_cubic.nc"} <= names
    from planetmodel.modelfile import read_model
    assert read_model(tmp_path / "model_parts.nc").nlayers == 13
    with netCDF4.Dataset(tmp_path / "case_lateral_custom_offset_cubic.nc") as ds:
        assert ds.method == "cubic"
        assert ds.variables["expected"].shape == ds.variables["query_colatitude"].shape
