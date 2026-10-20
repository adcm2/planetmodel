"""Every example and mini-app runs headless to completion, under the
markers its own header chooses: the extras of its `Needs:` line, each
skipped where not installed, and the `network` marker where its
`Network:` line says it fetches anything."""
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
SCRIPTS = sorted(p for p in EXAMPLES.rglob("*.py")
                 if "tutorials" not in p.relative_to(EXAMPLES).parts)

#: The pytest marker each extra's tests carry, where it has one.
MARKERS = {"netcdf": "netcdf", "maps": "maps", "meshing": "gmsh", "mfem": "mfem"}

#: Extras whose packages warn when used (netCDF4 under numpy 2.5 as it
#: writes, PyMFEM as it imports), so their examples are not run strictly.
WARN = {"netcdf", "mfem"}


def extras() -> dict[str, list[str]]:
    """Each extra and the modules it brings, read from `pyproject.toml`."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        table = tomllib.load(fh)["project"]["optional-dependencies"]
    return {name: [re.match(r"[A-Za-z0-9_.-]+", req).group(0) for req in reqs]
            for name, reqs in table.items()}


def header(path: Path) -> dict[str, str]:
    """The `Needs:`, `Network:` and `Run:` lines of an example's first cell."""
    out = {}
    for line in path.read_text().splitlines()[:40]:
        m = re.match(r"#\s*(Needs|Network|Run):\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def case(path: Path) -> object:
    head = header(path)
    needs = [n.strip() for n in head.get("Needs", "").split(",")
             if n.strip() and n.strip() != "none"]
    marks = [getattr(pytest.mark, MARKERS[n]) for n in needs if n in MARKERS]
    if head.get("Network", "none") != "none":
        marks.append(pytest.mark.network)
    return pytest.param(path, needs, id=str(path.relative_to(EXAMPLES)), marks=marks)


def test_every_example_states_its_header():
    known = extras()
    assert SCRIPTS
    for path in SCRIPTS:
        head = header(path)
        assert set(head) == {"Needs", "Network", "Run"}, path
        assert path.relative_to(ROOT).as_posix() in head["Run"], path
        for n in head["Needs"].split(","):
            assert n.strip() in known or n.strip() == "none", (path, n)


@pytest.mark.parametrize(("path", "needs"), [case(p) for p in SCRIPTS])
def test_the_example_runs(path, needs):
    known = extras()
    for n in needs:
        for module in known[n]:
            pytest.importorskip(module)
    env = dict(os.environ, MPLBACKEND="Agg")
    flags = [] if WARN & set(needs) else ["-W", "error"]
    r = subprocess.run([sys.executable, *flags, str(path)], cwd=path.parent,
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
