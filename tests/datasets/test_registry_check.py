"""The registry check: every shipped entry fetched from its host, its hashes
checked, read, and its product held to its protocol.

These tests touch the network and are selected only with `-m network`
(by hand, and by the registry workflow on demand and monthly); a changed
file upstream fails here first.  The cache is `PLANETMODEL_DATA`, or the
platform's.
"""
import pytest

from planetmodel import datasets
from planetmodel.datasets import REGISTRY, GridReader
from planetmodel.testing import check_dataset

pytestmark = pytest.mark.network


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_entry_fetches_reads_and_holds_to_its_contract(name):
    entry = REGISTRY[name]
    if isinstance(entry.reader, GridReader):
        pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    paths = datasets.fetch(entry)
    check_dataset(entry, paths=paths)
