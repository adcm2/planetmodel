"""The shipped data sets, each entry written once, and the lookup by name.

`REGISTRY` maps each shipped entry's name to it, read-only; a caller's
own `Dataset` is used exactly as a shipped one, passed where a name
would go (`DatasetLike`).  Each entry states what its source states: the
original host first (mirrors after it), the sha256 of the file as
published, the licence in the source's own words, and every work the
source asks to be cited.  `describe` and `cite` give that text.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from .registry import Dataset, Source
from .sph import SphReader

__all__ = ["DatasetLike", "REGISTRY", "as_dataset", "describe", "cite"]

#: A data set, or the name of a shipped one.
type DatasetLike = Dataset | str

#: Where the Woodhouse-Ritsema files are published, as one record.
_RTS = "https://zenodo.org/records/11212270/files"

#: SPECFEM3D_GLOBE's copies of S20RTS and S40RTS, at a release tag.
_SPECFEM = "https://raw.githubusercontent.com/SPECFEM/specfem3d_globe/v8.1.0/DATA"

#: The record of the Woodhouse-Ritsema files, to be cited beside each model.
_RTS_RECORD = ("Ritsema, J. and P. Koelemeijer (2026).  Tomographic filters for "
               "models S20RTS, S40RTS and SP12RTS.  Zenodo, "
               "doi:10.5281/zenodo.11212270")

#: What every entry of the Woodhouse-Ritsema files says of them.
_RTS_NOTES = ("Relative perturbations of the isotropic velocity, as fractions, "
              "between the CMB (3480 km) and the Moho (6346.6 km; 6346.619 km in "
              "Ritsema's code), relative to PREM's Voigt average; degree 0 is kept "
              "as the file holds it (Ritsema's maps leave it out).")

_ENTRIES: tuple[Dataset, ...] = (
    Dataset(
        "S20RTS",
        (Source(f"{_RTS}/S20RTS.sph",
                "642d5c826950a61d7ae324fbc9b58844d2e96f7d5277b232f5acbd1d428dcbe2",
                mirrors=(f"{_SPECFEM}/s20rts/S20RTS.sph",)),),
        SphReader(),
        title="A degree-20 shear-velocity model of the mantle",
        citation=("Ritsema, J., H. J. van Heijst and J. H. Woodhouse (1999).  "
                  "Complex shear wave velocity structure imaged beneath Africa and "
                  "Iceland.  Science 286, 1925-1928, "
                  "doi:10.1126/science.286.5446.1925", _RTS_RECORD),
        licence="CC BY 4.0", reference="PREM", notes=_RTS_NOTES),
    Dataset(
        "S40RTS",
        (Source(f"{_RTS}/S40RTS.sph",
                "95cd2ee3fb523189f52d2e8ae8ef822628d5e69b283cb1ffe8b29a6577750cf9",
                mirrors=(f"{_SPECFEM}/s40rts/s40rts.sph",)),),
        SphReader(),
        title="A degree-40 shear-velocity model of the mantle",
        citation=("Ritsema, J., A. Deuss, H. J. van Heijst and J. H. Woodhouse "
                  "(2011).  S40RTS: a degree-40 shear-velocity model for the mantle "
                  "from new Rayleigh wave dispersion, teleseismic traveltime and "
                  "normal-mode splitting function measurements.  Geophys. J. Int. "
                  "184, 1223-1236, doi:10.1111/j.1365-246X.2010.04884.x",
                  _RTS_RECORD),
        licence="CC BY 4.0", reference="PREM", notes=_RTS_NOTES),
    Dataset(
        "S12RTS",
        (Source(f"{_RTS}/S12RTS.sph",
                "2cbd52fe7eb15fcbaf5cfdbb3c13291b23c4e1252ba664916c59b980d938dc48"),),
        SphReader(),
        title="A degree-12 shear-velocity model of the mantle",
        citation=("Koelemeijer, P., J. Ritsema, A. Deuss and H. J. van Heijst "
                  "(2016).  SP12RTS: a degree-12 model of shear- and "
                  "compressional-wave velocity for Earth's mantle.  Geophys. J. "
                  "Int. 204, 1024-1039, doi:10.1093/gji/ggv481", _RTS_RECORD),
        licence="CC BY 4.0", reference="PREM", notes=_RTS_NOTES),
    Dataset(
        "SP12RTS",
        (Source(f"{_RTS}/SP12RTS..ES.sph",
                "ceb039fef9e19c45c4ebb98b62abb95b1cd9f51e0a56984304acfac693667d6c"),
         Source(f"{_RTS}/SP12RTS..EP.sph",
                "4575805bff1714568b7240715dc7168b8aefd305da5a9cee537b57e8fbfa1baf")),
        SphReader(parameters=("vs", "vp")),
        title="A degree-12 model of shear and compressional velocity of "
              "the mantle",
        citation=("Koelemeijer, P., J. Ritsema, A. Deuss and H. J. van Heijst "
                  "(2016).  SP12RTS: a degree-12 model of shear- and "
                  "compressional-wave velocity for Earth's mantle.  Geophys. J. "
                  "Int. 204, 1024-1039, doi:10.1093/gji/ggv481", _RTS_RECORD),
        licence="CC BY 4.0", reference="PREM",
        notes=_RTS_NOTES + "  The S part (..ES) is vs, the P part (..EP) vp."),
)


def _registry(entries: tuple[Dataset, ...]) -> Mapping[str, Dataset]:
    """The entries by name, read-only, a name used twice refused."""
    by_name: dict[str, Dataset] = {}
    for entry in entries:
        if entry.name in by_name:
            raise ValueError(f"two entries are named {entry.name!r}")
        by_name[entry.name] = entry
    return MappingProxyType(by_name)


#: The shipped data sets by name.
REGISTRY: Mapping[str, Dataset] = _registry(_ENTRIES)


def as_dataset(dataset: DatasetLike) -> Dataset:
    """The entry itself, or the shipped entry of that name."""
    if isinstance(dataset, Dataset):
        return dataset
    try:
        return REGISTRY[dataset]
    except KeyError:
        raise KeyError(f"no shipped data set is named {dataset!r}; the shipped "
                       f"ones are {sorted(REGISTRY)}, and a Dataset of one's own "
                       "is passed as it is") from None


def cite(dataset: DatasetLike) -> str:
    """The works a data set asks to be cited, one to a line."""
    return "\n".join(as_dataset(dataset).citation)


def describe(dataset: DatasetLike) -> str:
    """What the entry says of a data set, as lines of text: its title and
    body, its reference, its files, its licence, its citation and notes."""
    d = as_dataset(dataset)
    lines = [f"{d.name}: {d.title}", f"  body: {d.body}"]
    if d.reference is not None:
        lines.append(f"  relative to: {d.reference}")
    for s in d.sources:
        lines.append(f"  file: {s.name} from {s.url} (sha256 {s.sha256[:12]}...)")
        lines += [f"        mirror {m}" for m in s.mirrors]
        if s.members is not None:
            lines.append(f"        members {', '.join(s.members)}")
    lines.append(f"  licence: {d.licence}")
    lines += [f"  cite: {c}" for c in d.citation]
    if d.notes:
        lines.append(f"  notes: {d.notes}")
    return "\n".join(lines)
