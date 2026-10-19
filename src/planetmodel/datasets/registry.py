"""What a data set is to the library: an entry naming its files and its reader.

A `Source` is one file of a published data set: the URL it is served
from, its sha256, the name it is cached under, the mirrors that serve it
too, and for an archive the members its reader needs.  A `Dataset` is
the entry for the whole: its name, its sources, its reader, and what the
source does not state in machine-readable form (the model its
perturbations are relative to, its title, how it asks to be cited, its
licence in its own words or "none stated", the body it describes).  The
reader is an object the entry holds, as a `DeckFormat` is the knowledge
of a deck: it carries the format and whatever is particular to this
source (which variable is which name, a unit it mislabels, a coordinate
it glitches), and `read(paths, dataset, **options)` gives the object the
data set is.  Nothing dispatches on a data set's name.

The shipped entries are in `planetmodel.datasets.entries`.  The library
holds addresses and hashes, never data.
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

__all__ = ["Source", "Dataset", "Reader", "ARCHIVES", "archive_stem"]

#: The suffixes of the archives a source may name members of.
ARCHIVES = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar",
            ".zip")

#: The URL schemes a source may be served by.
_SCHEMES = ("https", "http", "file")

#: A name a data set or a cached file may have: no path, nothing hidden.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


def _member(name: str, archive: str) -> str:
    """A member name checked to stay inside its archive's directory."""
    parts = name.split("/")
    if (not name or name.startswith("/") or "\\" in name
            or any(p in ("", "..") for p in parts)):
        raise ValueError(f"the member {name!r} of {archive!r} must be a relative "
                         "path inside the archive, without '..'")
    return name


def archive_stem(name: str) -> str | None:
    """The name an archive's members are extracted under, its archive suffix
    removed, or None for a file that is not an archive."""
    for suffix in ARCHIVES:
        if name.lower().endswith(suffix) and len(name) > len(suffix):
            return name[:-len(suffix)]
    return None


@dataclass(frozen=True)
class Source:
    """One file of a data set: served at `url` (https, http or file), of
    the given sha256, cached as `name` (the URL's last path segment by
    default), served too by `mirrors`, and for an archive (`ARCHIVES`)
    the `members` a reader needs, extracted and given in that order;
    None gives the archive itself."""

    url: str
    sha256: str
    _: KW_ONLY
    name: str | None = None
    mirrors: tuple[str, ...] = ()
    members: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for url in (self.url, *self.mirrors):
            if urlsplit(url).scheme not in _SCHEMES:
                raise ValueError(f"a source is served over {_SCHEMES}, not {url!r}")
        digest = str(self.sha256).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"sha256 must be 64 hexadecimal digits, got "
                             f"{self.sha256!r}")
        object.__setattr__(self, "sha256", digest)
        name = self.name
        if name is None:
            name = unquote(posixpath.basename(urlsplit(self.url).path))
        if not _NAME.fullmatch(name or ""):
            raise ValueError(
                f"{name!r} is not a file name a source can be cached under (from "
                f"{self.url!r}); give `name=`")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "mirrors", tuple(self.mirrors))
        if self.members is not None:
            if archive_stem(name) is None:
                raise ValueError(f"{name!r} is not an archive ({ARCHIVES}), so it "
                                 "has no members")
            members = tuple(_member(m, name) for m in self.members)
            if not members or len(set(members)) != len(members):
                raise ValueError(f"the members of {name!r} must be named, once each")
            object.__setattr__(self, "members", members)


class Reader(Protocol):
    """What reads a data set's fetched files into the object it is."""

    def read(self, paths: tuple[Path, ...], dataset: Dataset, **options: Any
             ) -> object:
        """The data set, from the paths `fetch` gives, in its sources' order."""
        ...


@dataclass(frozen=True)
class Dataset:
    """The entry for a published data set: `name` (letters, digits and
    `._+-`), its `sources`, and the `reader` of their files; `title`,
    `citation` (each work the source asks to be cited) and `licence` (as
    the source states it, or "none stated") are required; `body` is what
    it describes, `reference` the name of the entry of the model its
    perturbations are relative to, and `notes` what else a user should
    know of it."""

    name: str
    sources: tuple[Source, ...]
    reader: Reader
    _: KW_ONLY
    title: str
    citation: tuple[str, ...]
    licence: str
    body: str = "Earth"
    reference: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"{self.name!r} is not a data set name: letters, digits "
                             "and ._+- , starting with a letter or digit")
        sources = tuple(self.sources)
        if not sources or not all(isinstance(s, Source) for s in sources):
            raise TypeError(f"{self.name}: sources must be one or more Sources")
        names = [s.name for s in sources]
        stems = [archive_stem(s.name) for s in sources if s.members is not None]
        if len(set(names + stems)) != len(names) + len(stems):
            raise ValueError(f"{self.name}: two sources would share a name in the "
                             f"cache: {names}")
        object.__setattr__(self, "sources", sources)
        if not callable(getattr(self.reader, "read", None)):
            raise TypeError(f"{self.name}: the reader must have a read(paths, "
                            f"dataset, **options) method, got {self.reader!r}")
        citation = (self.citation,) if isinstance(self.citation, str) else self.citation
        object.__setattr__(self, "citation", tuple(citation))
        if not self.licence:
            raise ValueError(f"{self.name}: state the licence as the source does, "
                             "or 'none stated'")
