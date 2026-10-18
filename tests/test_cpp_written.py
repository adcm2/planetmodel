"""The model files the C++ writer wrote, read by the Python: each
validates, reads as a model, and holds the values the C++ reader reads
from it; a fixture the C++ read and wrote back reads as the fixture does.

The C++ tests leave the files in their build tree; PLANETMODEL_CPP_WRITTEN
names that directory (`build/cpp/tests/written` for the build of
CLAUDE.md), and without it these tests are skipped.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

from planetmodel.modelfile import read_model, validate_structure  # noqa: E402

WRITTEN = os.environ.get("PLANETMODEL_CPP_WRITTEN")
FIXTURES = Path(__file__).resolve().parents[1] / "cpp" / "tests" / "fixtures"
needs_written = pytest.mark.skipif(
    WRITTEN is None,
    reason="PLANETMODEL_CPP_WRITTEN names no directory of C++-written files")


def _written(pattern: str) -> list[Path]:
    return [] if WRITTEN is None else sorted(Path(WRITTEN).glob(pattern))


def _check_expect(model, expect: Path, *, rtol: float) -> int:
    """The lines of an expectation file met by a model: values at points
    in either frame, and the mapping's displacement."""
    checked = 0
    for line in expect.read_text().splitlines():
        words = line.split()
        if words[0] == "value":
            layer, name, frame = int(words[1]), words[2], words[3]
            r, t, p = (float(x) for x in words[4:7])
            want = np.array([float(x) for x in words[7:]])
            got = np.asarray(model.layer(layer)[name].evaluate(
                r, t, p, frame="spherical" if frame == "s" else "cartesian"),
                dtype=float).ravel()
            scale = max(1.0, float(np.max(np.abs(want))))
            assert np.allclose(got, want, rtol=0, atol=rtol * scale), line
            checked += 1
        elif words[0] == "displacement":
            X = np.array([float(x) for x in words[1:4]])
            want = np.array([float(x) for x in words[4:7]])
            got = np.asarray(model.geometry.mapping.displacement(X[None, :]),
                             dtype=float)[0]
            outer = float(model.skeleton.boundaries[-1])
            assert np.allclose(got, want, rtol=0, atol=1e-12 * outer), line
            checked += 1
    return checked


@needs_written
def test_the_cpp_wrote_its_files():
    names = {p.name for p in _written("*.nc")}
    assert {"written_si.nc", "written_custom.nc", "written_tensors.nc",
            "roundtrip_model_prem.nc", "roundtrip_model_harmonic.nc"} <= names


@needs_written
@pytest.mark.parametrize("path", _written("written_*.nc"), ids=lambda p: p.name)
def test_a_file_the_cpp_wrote_reads_as_the_cpp_reads_it(path):
    validate_structure(path)
    model = read_model(path)
    assert _check_expect(model, path.with_suffix(".expect"), rtol=1e-10) > 10


@needs_written
@pytest.mark.parametrize("path", _written("roundtrip_*.nc"), ids=lambda p: p.name)
def test_a_fixture_written_back_reads_as_the_fixture(path):
    validate_structure(path)
    expect = FIXTURES / (path.stem.removeprefix("roundtrip_") + ".expect")
    if not expect.exists():
        return                              # the harmonic copy: structure alone
    model = read_model(path)
    assert _check_expect(model, expect, rtol=1e-11) > 10
