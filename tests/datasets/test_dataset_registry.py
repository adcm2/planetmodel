"""Entries of data sets: what a Source and a Dataset accept and refuse."""
import pytest

from planetmodel.datasets import (REGISTRY, Dataset, Source, archive_stem, as_dataset,
                                  cite, describe)
from planetmodel.testing import check_dataset

SHA = "ab" * 32


class Echo:
    """A reader that gives back what it was handed."""

    def read(self, paths, dataset, **options):
        return {"paths": paths, "name": dataset.name, **options}


def entry(*sources, **kw):
    kw.setdefault("title", "A model")
    kw.setdefault("citation", ("Someone, 2026, A model, J. Models 1, 1",))
    kw.setdefault("licence", "CC BY 4.0")
    return Dataset("A-model+1", sources or (Source("https://example.org/m.sph", SHA),),
                   Echo(), **kw)


def test_a_source_names_its_file_from_its_url():
    s = Source("https://zenodo.org/records/1/files/S40RTS.sph", SHA.upper())
    assert s.name == "S40RTS.sph" and s.sha256 == SHA and s.members is None
    assert Source("https://h.org/a/S362ANI%2BM.nc", SHA).name == "S362ANI+M.nc"
    named = Source("https://h.org/x/content", SHA, name="S20RTS.sph")
    assert named.name == "S20RTS.sph"
    mirrored = Source("file:///tmp/m.sph", SHA, mirrors=["https://m.org/m.sph"])
    assert mirrored.mirrors == ("https://m.org/m.sph",)


@pytest.mark.parametrize("url, kw, words", [
    ("ftp://h.org/m.sph", {}, "served over"),
    ("https://h.org/m.sph", {"mirrors": ("ftp://m.org/m.sph",)}, "served over"),
    ("https://h.org/dir/", {}, "cached under"),
    ("https://h.org/m.sph", {"name": "../m.sph"}, "cached under"),
    ("https://h.org/m.sph", {"name": ".hidden"}, "cached under"),
    ("https://h.org/m.sph", {"members": ("a",)}, "not an archive"),
    ("https://h.org/m.tar.gz", {"members": ("../a",)}, "relative path"),
    ("https://h.org/m.tar.gz", {"members": ("/etc/a",)}, "relative path"),
    ("https://h.org/m.zip", {"members": ("a//b",)}, "relative path"),
    ("https://h.org/m.zip", {"members": ()}, "once each"),
    ("https://h.org/m.zip", {"members": ("a", "a")}, "once each"),
])
def test_a_source_refuses_what_it_cannot_cache_safely(url, kw, words):
    with pytest.raises(ValueError, match=words):
        Source(url, SHA, **kw)


def test_a_source_refuses_a_malformed_hash():
    for bad in ("ab" * 31, "zz" * 32, ""):
        with pytest.raises(ValueError, match="64 hexadecimal"):
            Source("https://h.org/m.sph", bad)


def test_archive_stems():
    assert archive_stem("crust1.0.tar.gz") == "crust1.0"
    assert archive_stem("GRAIL.zip") == "GRAIL" and archive_stem("a.TGZ") == "a"
    assert archive_stem("S40RTS.sph") is None and archive_stem(".zip") is None


def test_a_dataset_holds_its_entry():
    d = entry(citation="One work, cited")
    assert d.citation == ("One work, cited",) and d.body == "Earth"
    assert d.reference is None and d.sources[0].name == "m.sph"
    assert as_dataset(d) is d


@pytest.mark.parametrize("change, error, words", [
    ({"name": "a/b"}, ValueError, "not a data set name"),
    ({"name": ""}, ValueError, "not a data set name"),
    ({"sources": ()}, TypeError, "one or more Sources"),
    ({"sources": ("https://h.org/m.sph",)}, TypeError, "one or more Sources"),
    ({"reader": object()}, TypeError, "read"),
    ({"licence": ""}, ValueError, "licence"),
    ({"sources": (Source("https://h.org/m.sph", SHA),
                  Source("https://g.org/m.sph", SHA))}, ValueError, "share a name"),
    ({"sources": (Source("https://h.org/m.tar.gz", SHA, members=("a",)),
                  Source("https://g.org/m", SHA))}, ValueError, "share a name"),
])
def test_a_dataset_refuses_a_malformed_entry(change, error, words):
    args = {"name": "m", "sources": (Source("https://h.org/m.sph", SHA),),
            "reader": Echo(), "title": "t", "citation": ("c",), "licence": "l"}
    args.update(change)
    with pytest.raises(error, match=words):
        Dataset(args.pop("name"), args.pop("sources"), args.pop("reader"), **args)


def test_title_citation_and_licence_are_required():
    with pytest.raises(TypeError):
        Dataset("m", (Source("https://h.org/m.sph", SHA),), Echo())


def test_the_registry_is_read_only_and_names_what_it_lacks():
    with pytest.raises(TypeError):
        REGISTRY["mine"] = entry()
    with pytest.raises(KeyError, match="no shipped data set is named 'nothing'"):
        as_dataset("nothing")


def test_describe_and_cite():
    d = entry(Source("https://h.org/m.tar.gz", SHA, mirrors=("https://g.org/m.tar.gz",),
                     members=("m/a.txt", "m/b.txt")),
              reference="PREM", body="Moon", notes="a test entry",
              citation=("First, 2026", "Second, 2027"))
    text = describe(d)
    for words in ("A-model+1: A model", "body: Moon", "relative to: PREM",
                  "file: m.tar.gz from https://h.org/m.tar.gz", "mirror https://g.org",
                  "members m/a.txt, m/b.txt", "licence: CC BY 4.0", "cite: First, 2026",
                  "notes: a test entry", "sha256 abababababab..."):
        assert words in text, words
    assert cite(d) == "First, 2026\nSecond, 2027"


def test_the_contract_holds_an_entry_alone():
    check_dataset(entry())
