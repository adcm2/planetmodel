"""Fixtures shared by the whole suite."""
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def extra_modules() -> tuple[str, ...]:
    """The importable names of every optional dependency, read from the
    extras of `pyproject.toml`, which is their one declaration: each
    distribution there imports under its own name."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        extras = tomllib.load(fh)["project"]["optional-dependencies"]
    names = {re.match(r"[A-Za-z0-9_.-]+", req).group(0)
             for reqs in extras.values() for req in reqs}
    return tuple(sorted(names))
