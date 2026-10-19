"""Radial references read from decks: the reader, and the shipped entries."""
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

import planetmodel as pm
from planetmodel import ND, MINEOS, MineosModel, datasets
from planetmodel.datasets import REGISTRY, Dataset, DeckReader, Source
from planetmodel.deck import nd_names
from planetmodel.testing import check_dataset

PREM_DECK = Path(__file__).resolve().parents[2] / "examples" / "data" / "prem.200"

WEBER_LIKE = """mantle
0 1 0.5 2.6
40 1 0.5 2.6
mantle-two
40 7.7 4.4 3.3
1407.1 7.5 3.2 3.4
outer-core
1407.1 4.1 0.0 5.1
1497.1 4.1 0.0 5.1
inner-core
1497.1 4.3 2.3 8.0
1737.1 4.3 2.3 8.0
"""


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANETMODEL_DATA", str(tmp_path / "cache"))


def local(tmp_path, name, data, reader, **kw):
    path = tmp_path / name
    path.write_bytes(data)
    kw.setdefault("title", "t")
    kw.setdefault("citation", ("c",))
    kw.setdefault("licence", "l")
    return Dataset(name.replace("_", "-"), (Source(path.as_uri(),
                                                   hashlib.sha256(data).hexdigest()),),
                   reader, **kw)


def test_an_nd_body_is_a_model_whose_radius_is_its_deepest_depth(tmp_path):
    d = local(tmp_path, "weber.nd", WEBER_LIKE.encode(),
              DeckReader(ND, names=nd_names, kind="linear"), body="Moon")
    m = datasets.load(d)
    assert m.skeleton.boundaries[0] == 0.0
    assert m.skeleton.boundaries[-1] == pytest.approx(1737.1e3)
    assert [lay.name for lay in m.layers] == ["inner_core", "outer_core",
                                             "mantle_two", "mantle"]
    assert [f.name for f in m.geometry.interfaces] == ["icb", "cmb", None, "surface"]
    core = m.layer("outer_core")
    assert core["vp"](0.5 * (240e3 + 330e3)) == pytest.approx(4.1e3)
    two = m.layer("mantle_two")
    lo, hi = two.interval
    assert two["vp"](0.5 * (lo + hi)) == pytest.approx(0.5 * (7.7e3 + 7.5e3))  # linear
    pm.testing.check_model(m)
    check_dataset(d, paths=datasets.fetch(d))
    km = datasets.load(d, scales=pm.Scales(length=1e3))
    assert km.skeleton.boundaries[-1] == pytest.approx(1737.1)


def test_a_mineos_deck_is_read_by_its_model_type(tmp_path):
    d = local(tmp_path, "prem.200", PREM_DECK.read_bytes(),
              DeckReader(MINEOS, model=MineosModel))
    m = datasets.load(d)
    assert isinstance(m, MineosModel) and m.layers[0].name == "inner_core"
    want = MineosModel(PREM_DECK)
    assert m.layer(3)["vsv"](5e6) == want.layer(3)["vsv"](5e6)
    natural = datasets.load(d, kind="natural")
    knots = m.knots[3]
    between = 0.5 * (knots[:-1] + knots[1:])          # the kinds agree at knots
    assert not np.allclose(natural.layer(3)["vsv"](between),
                           m.layer(3)["vsv"](between), rtol=1e-12, atol=0.0)
    with pytest.raises(ValueError, match="a deck is one file"):
        DeckReader(MINEOS).read((PREM_DECK, PREM_DECK), d)


def test_a_reference_named_in_the_registry_is_loaded(tmp_path, monkeypatch):
    from planetmodel.datasets import entries
    ref = local(tmp_path, "ref.nd", WEBER_LIKE.encode(),
                DeckReader(ND, names=nd_names, kind="linear"), body="Moon")
    monkeypatch.setattr(entries, "REGISTRY", {**REGISTRY, ref.name: ref})
    perturbed = Dataset("P", ref.sources, DeckReader(ND), title="t", citation=("c",),
                        licence="l", reference=ref.name)
    got = datasets.reference_of(perturbed)
    assert got.skeleton.boundaries[-1] == pytest.approx(1737.1e3)
    other = Dataset("Q", ref.sources, DeckReader(ND), title="t", citation=("c",),
                    licence="l", reference="nowhere")
    with pytest.raises(ValueError, match="neither a model of the catalogue"):
        datasets.reference_of(other)


def test_a_member_of_an_archive_is_read(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("models/weber.nd", WEBER_LIKE)
        zf.writestr("models/readme.txt", "not wanted")
    archive = tmp_path / "moon.zip"
    archive.write_bytes(buffer.getvalue())
    d = Dataset("moon", (Source(archive.as_uri(),
                                hashlib.sha256(buffer.getvalue()).hexdigest(),
                                members=("models/weber.nd",)),),
                DeckReader(ND, names=nd_names, kind="linear"), title="t",
                citation=("c",), licence="l", body="Moon")
    assert datasets.load(d).skeleton.boundaries[-1] == pytest.approx(1737.1e3)


REFERENCE_ENTRIES = ("STW105", "ak135-F", "iasp91", "VPREMOON", "MoonWeber2011",
                     "ISSI-MOON-M1", "ISSI-MOON-M2", "ISSI-MOON-M3")


@pytest.mark.parametrize("name", REFERENCE_ENTRIES)
def test_the_shipped_references_hold_to_their_contract(name):
    d = REGISTRY[name]
    check_dataset(d)
    assert isinstance(d.reader, DeckReader)
    moon = name not in ("STW105", "ak135-F", "iasp91")
    assert d.body == ("Moon" if moon else "Earth")
    if moon:
        (source,) = d.sources
        assert source.members[0].endswith(".nd") and d.licence == "CC BY 4.0"
    assert np.all([len(s.sha256) == 64 for s in d.sources])
