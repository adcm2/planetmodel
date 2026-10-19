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

from ..catalogue import MineosModel
from ..deck import MINEOS, ND, TVEL, nd_names
from .decks import DeckReader
from .grids import GridReader
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

#: TauP's standard models, at a release tag.
_TAUP = ("https://raw.githubusercontent.com/crotwell/TauP/v3.2.1/src/main/resources/"
         "edu/sc/seis/TauP/StdModels")

#: TauP, cited where its files are used.
_TAUP_CITE = ("Crotwell, H. P., T. J. Owens and J. Ritsema (1999).  The TauP "
              "Toolkit: flexible seismic travel-time and ray-path utilities.  "
              "Seismol. Res. Lett. 70, 154-160, doi:10.1785/gssrl.70.2.154")

#: The licence of a model in TauP's repository, which states none of its own.
_TAUP_LICENCE = ("none stated for the model; the file is from TauP's repository, "
                 "LGPL-3.0")

#: The compilation of the Moon's models, and its archive's members.
_MOON = ("https://zenodo.org/records/3372489/files/"
         "SSR_2019_Garcia_Moon_Seismological_structure_supplementary_zenodo.zip")
_MOON_SHA = "dfdb5cb68db402d989e953f48f1a82cb8c4c7104d15897861f25e304ad4f3136"
_MOON_DIR = ("SSR_2019_Garcia_Moon_Seismological_structure_supplementary_zenodo/"
             "Sup_Material_SSR_Internal_Moon_Seismological_models/")
_MOON_CITE = ("Garcia, R. F., A. Khan, M. Drilleau, L. Margerin, T. Kawamura, D. Sun, "
              "M. A. Wieczorek, A. Rivoldini, C. Nunn, R. C. Weber, A. G. Marusiak, "
              "P. Lognonné, Y. Nakamura and P. Zhu (2019).  Lunar seismology: an "
              "update on interior structure models.  Space Sci. Rev. 215, 50, "
              "doi:10.1007/s11214-019-0613-y; the files at Zenodo, "
              "doi:10.5281/zenodo.3372489")
_MOON_NOTES = ("TauP's named discontinuities, joined linearly as TauP joins them; the "
               "body's radius is the deepest depth.")


#: Where the Earth Model Collaboration serves its netCDF files.
_EMC = "https://data.earthscope.org/archive/seismology/products/emc/netcdf"

#: The licence of an EMC file, stated in its DataCite record.
_EMC_LICENCE = "CC BY 4.0 (stated in the EMC's DataCite records)"


def _emc_cites(doi: str) -> tuple[str, str]:
    """What the EMC asks cited beside a model's paper: its page and itself."""
    return (f"EarthScope Data Services, Earth Model Collaboration: the model's page, "
            f"doi:{doi}; and the EMC, doi:10.17611/DP/EMC.1",
            "Trabant, C., A. R. Hutko, M. Bahavar, R. Karstens, T. Ahern and R. Aster "
            "(2012).  Data products at the IRIS DMC: stepping stones for research "
            "and other applications.  Seismol. Res. Lett. 83, 846-854, "
            "doi:10.1785/0220120032")


def _moon(name: str, member: str, title: str, cite: str, *,
          notes: str = "") -> Dataset:
    """An entry of one model of the Moon from the compilation's archive."""
    return Dataset(name, (Source(_MOON, _MOON_SHA, members=(_MOON_DIR + member,)),),
                   DeckReader(ND, names=nd_names, kind="linear"), title=title,
                   citation=(cite, _MOON_CITE), licence="CC BY 4.0", body="Moon",
                   notes=" ".join(filter(None, (_MOON_NOTES, notes))))


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
    Dataset(
        "STW105",
        (Source("https://zenodo.org/records/8357380/files/STW105",
                "ccc2060674502b8032ef7bb11d2ea08fd4c8532f19455d62a784907bb5588352"),),
        DeckReader(MINEOS, model=MineosModel),
        title="The radial reference of S362ANI, S362ANI+M and GLAD-M25 (REF)",
        citation=("Kustowski, B., G. Ekström and A. M. Dziewoński (2008).  "
                  "Anisotropic shear-wave velocity structure of the Earth's mantle: "
                  "a global model.  J. Geophys. Res. 113, B06306, "
                  "doi:10.1029/2007JB005169",
                  "Moulik, P. and G. Ekström (2014).  An anisotropic shear velocity "
                  "model of the Earth's mantle using normal modes, body waves, "
                  "surface waves and long-period waveforms.  Geophys. J. Int. 199, "
                  "1713-1738, doi:10.1093/gji/ggu356; the files at Zenodo, "
                  "doi:10.5281/zenodo.8357380"),
        licence="none stated",
        notes="A mineos deck in SI, its discontinuities repeated radii; the third "
              "header line's two extra integers are kept.  EMC's copy moves the "
              "upper side of each discontinuity up by 0.1 m and is not this file."),
    Dataset(
        "ak135-F",
        (Source(f"{_TAUP}/ak135favg.nd",
                "d08d19dbaad8d90a51c663011239cc45ad48ea2ed9b4ec9e7647b1a196f1eb19"),),
        DeckReader(ND, names=nd_names, kind="linear"),
        title="ak135 with density and attenuation, the average Earth",
        citation=("Kennett, B. L. N., E. R. Engdahl and R. Buland (1995).  "
                  "Constraints on seismic velocities in the Earth from traveltimes.  "
                  "Geophys. J. Int. 122, 108-124, "
                  "doi:10.1111/j.1365-246X.1995.tb03540.x",
                  "Montagner, J.-P. and B. L. N. Kennett (1996).  How to reconcile "
                  "body-wave and normal-mode reference Earth models.  Geophys. J. "
                  "Int. 125, 229-248, doi:10.1111/j.1365-246X.1996.tb06548.x",
                  _TAUP_CITE),
        licence=_TAUP_LICENCE,
        notes="TauP's ak135favg.nd: the average of the oceanic and continental "
              "forms, with its 3 km ocean, joined linearly as TauP joins it; its "
              "Q columns are TauP's Qp (the published table's 'Q kappa') and Qs "
              "(qmu)."),
    Dataset(
        "iasp91",
        (Source(f"{_TAUP}/iasp91.tvel",
                "c16b31d1eeae292e857a4224ac98719abcad7aff5c0068204fcadeb2868dd6f2"),),
        DeckReader(TVEL, kind="linear"),
        title="The IASPEI 1991 seismological tables' velocity model",
        citation=("Kennett, B. L. N. and E. R. Engdahl (1991).  Traveltimes for "
                  "global earthquake location and phase identification.  Geophys. "
                  "J. Int. 105, 429-465, doi:10.1111/j.1365-246X.1991.tb06724.x",
                  _TAUP_CITE),
        licence=_TAUP_LICENCE,
        notes="TauP's iasp91.tvel, joined linearly; its density column is not "
              "part of the published model and its source is not stated."),
    Dataset(
        "S362ANI+M",
        (Source(f"{_EMC}/S362ANI%2BM_kmps.r0.0-n4c.nc",
                "7bf5076e70ab9cc59b6d0f5a20d513c34d9309eecd33873b9d30e2b47188b425"),),
        GridReader({"vs": ("vs", None), "vsv": ("vsv", None), "vsh": ("vsh", None)}),
        title="A radially anisotropic shear-velocity model of the mantle, on EMC's "
              "2-degree grid",
        citation=("Moulik, P. and G. Ekström (2014).  An anisotropic shear velocity "
                  "model of the Earth's mantle using normal modes, body waves, "
                  "surface waves and long-period waveforms.  Geophys. J. Int. 199, "
                  "1713-1738, doi:10.1093/gji/ggu356",
                  *_emc_cites("10.17611/DP/9991784")),
        licence=_EMC_LICENCE, reference="STW105",
        notes="Absolute velocities in km/s at 25 depths from 25 to 2890 km on a "
              "2-degree grid with both poles, from the native model on spherical "
              "splines; relative to STW105 as published."),
    Dataset(
        "SEMUCB-WM1",
        (Source(f"{_EMC}/semucb-2014-ucb-vs-viz-only.r0.1-n4c.nc",
                "92b191f909c05e32151044695ff2be956ca25dfa0f62470f38c0ea976857cd43"),),
        GridReader({"vs": ("vs", None)}),
        title="A whole-mantle shear-velocity model from spectral-element waveform "
              "tomography, on EMC's grid",
        citation=("French, S. W. and B. A. Romanowicz (2014).  Whole-mantle radially "
                  "anisotropic shear velocity structure from spectral-element "
                  "waveform tomography.  Geophys. J. Int. 199, 1303-1327, "
                  "doi:10.1093/gji/ggu334",
                  *_emc_cites("10.17611/dp/emc.2023.semucbwm1.1")),
        licence=_EMC_LICENCE,
        notes="Voigt-average vs in km/s at 74 depths from 30 to 2891 km, on a grid of "
              "about 0.99 degrees with a short last interval (a custom grid with "
              "both poles); EMC calls this grid one for viewing.  Its own 1D "
              "reference is not shipped, so it is read as values."),
    Dataset(
        "GLAD-M25",
        (Source(f"{_EMC}/glad-m25-vs.r0.0-n4c.nc",
                "8916f703b07b2c23587dc8f6a7eddd89505d5a5491bc6712cd9ca70fe1e23d93"),),
        GridReader({"vsv": ("vsv", None), "vsh": ("vsh", None)}, stride=2,
                   vertical=(50.0, None), seam="mean"),
        title="A global model from adjoint tomography, its shear velocities on EMC's "
              "half-degree grid",
        citation=("Lei, W., Y. Ruan, E. Bozdağ, D. Peter, M. Lefebvre, D. Komatitsch, "
                  "J. Tromp, J. Hill, N. Podhorszki and D. Pugmire (2020).  Global "
                  "adjoint tomography: model GLAD-M25.  Geophys. J. Int. 223, 1-21, "
                  "doi:10.1093/gji/ggaa253",
                  *_emc_cites("10.17611/dp/emc.2020.gladm2500.1")),
        licence=_EMC_LICENCE, reference="STW105",
        notes="vsv and vsh in km/s at 342 depths from -5 to 2891 km on a half-degree "
              "grid with both poles (505 MB), discontinuities at 410 and 650 km.  Read "
              "by default from 50 km down, every second node (one degree, about 1.3 "
              "GB as a model): its crust, with zeros in water and air, differs by up "
              "to 3.2 km/s between the columns at -180 and 180 degrees, which differ "
              "by at most 0.04 % from 50 km down, where they are averaged.  "
              "load(..., stride=1) reads the half-degree grid.  Its starting model's "
              "reference is STW105."),
    Dataset(
        "SGLOBE-rani",
        (Source(f"{_EMC}/SGLOBE-rani-voigt_percent.r0.0-n4c.nc",
                "78a90fa796659ae99539612126c8e4ec72ebf9c3d3dad1f0c45a754319b32113"),),
        GridReader({"dvs": ("vs", "relative")}),
        title="A joint model of isotropic and radially anisotropic mantle structure, "
              "its Voigt-average shear velocity on EMC's 2-degree grid",
        citation=("Chang, S.-J., A. M. G. Ferreira, J. Ritsema, H. J. van Heijst and "
                  "J. H. Woodhouse (2015).  Joint inversion for global isotropic and "
                  "radially anisotropic mantle structure including crustal thickness "
                  "perturbations.  J. Geophys. Res. Solid Earth 120, 4278-4300, "
                  "doi:10.1002/2014JB011824",
                  *_emc_cites("10.17611/dp/emc2015sgloberani")),
        licence=_EMC_LICENCE, reference="PREM",
        notes="The relative perturbation of the Voigt-average vs, in percent, at 37 "
              "depths from 25 to 2800 km, latitudes -88 to 88 every 2 degrees (the "
              "poles by the nearest ring); relative to PREM."),
    Dataset(
        "GyPSuM",
        (Source(f"{_EMC}/GYPSUMS_kmps.r0.0-n4c.nc",
                "acb1e8d9563dfdeffb016ededdbba9890ceab0efefb0139554856d4c97799cc5"),),
        GridReader({"vs": ("vs", None)}, seam="mean"),
        title="A joint tomographic model of mantle density and wave speeds: its "
              "shear velocity on EMC's 1-degree grid",
        citation=("Simmons, N. A., A. M. Forte, L. Boschi and S. P. Grand (2010).  "
                  "GyPSuM: a joint tomographic model of mantle density and seismic "
                  "wave speeds.  J. Geophys. Res. 115, B12310, "
                  "doi:10.1029/2010JB007631", *_emc_cites("10.17611/DP/9991624")),
        licence=_EMC_LICENCE,
        notes="Absolute vs in km/s, a layered model sampled at the top and bottom of "
              "each block 1 km apart (read linearly, a step a 1 km ramp).  Its "
              "columns at -180 and 180 degrees differ by up to 1.1 %, the source's "
              "gridding not being periodic; they are averaged.  The file's id says "
              "percent; it holds km/s."),
    Dataset(
        "CSEM-North-Atlantic",
        (Source(f"{_EMC}/csem-north-atlantic-2019.12.01.r0.0-n4c.nc",
                "d4151db9dbfdacbb90086140aea86f528225730eab7051f2ac3c731dbd8f1f04"),),
        GridReader({"vsv": ("vsv", None), "vsh": ("vsh", None)}),
        title="The North Atlantic of the Collaborative Seismic Earth Model, a regional "
              "grid",
        citation=("Fichtner, A., D.-P. van Herwaarden, M. Afanasiev, S. Simutė, L. "
                  "Krischer, Y. Çubuk-Sabuncu, T. Taymaz, L. Colli, E. Saygin, A. "
                  "Villaseñor, J. Trampert, P. Cupillard, H.-P. Bunge and H. Igel "
                  "(2018).  The Collaborative Seismic Earth Model: generation 1.  "
                  "Geophys. Res. Lett. 45, 4007-4016, doi:10.1029/2018GL077338",
                  "Krischer, L., A. Fichtner, C. Boehm and H. Igel (2018).  Automated "
                  "large-scale full seismic waveform inversion for North America and "
                  "the North Atlantic.  J. Geophys. Res. Solid Earth 123, 5902-5928, "
                  "doi:10.1029/2017JB015289",
                  "Rickers, F., A. Fichtner and J. Trampert (2013).  The Iceland-Jan "
                  "Mayen plume system and its impact on mantle dynamics in the North "
                  "Atlantic region: evidence from full-waveform inversion.  Earth "
                  "Planet. Sci. Lett. 367, 39-51, doi:10.1016/j.epsl.2013.02.022",
                  *_emc_cites("10.17611/dp/emccsemnatl20191201")),
        licence=_EMC_LICENCE,
        notes="vsv and vsh in km/s from the surface to 500 km, latitudes 20 to 75 and "
              "longitudes -55 to 10 every half degree: a regional grid, for merging "
              "into a global model."),
    _moon("VPREMOON", "MoonGarcia_PEPI_2011.nd",
          "VPREMOON, a very preliminary reference Moon model",
          "Garcia, R. F., J. Gagnepain-Beyneix, S. Chevrot and P. Lognonné (2011).  "
          "Very preliminary reference Moon model.  Phys. Earth Planet. Inter. 188, "
          "96-113, doi:10.1016/j.pepi.2011.06.015"),
    _moon("MoonWeber2011", "MoonWeber_Science_2011.nd",
          "A seismic model of the Moon's core",
          "Weber, R. C., P.-Y. Lin, E. J. Garnero, Q. Williams and P. Lognonné "
          "(2011).  Seismic detection of the lunar core.  Science 331, 309-312, "
          "doi:10.1126/science.1199375"),
    _moon("ISSI-MOON-M1", "ISSI_MOON_M1.nd",
          "The ISSI lunar team's model M1, the median of an ensemble",
          "Garcia et al. (2019), as below: the ISSI lunar team's models",
          notes="Its crust-mantle boundary is marked for the codes that need one; "
                "the ensemble does not define it."),
    _moon("ISSI-MOON-M2", "ISSI_MOON_M2.nd",
          "The ISSI lunar team's model M2, the median of an ensemble",
          "Garcia et al. (2019), as below: the ISSI lunar team's models"),
    _moon("ISSI-MOON-M3", "ISSI_MOON_M3.nd",
          "The ISSI lunar team's model M3, the median of an ensemble",
          "Garcia et al. (2019), as below: the ISSI lunar team's models",
          notes="Its crust-mantle boundary is marked for the codes that need one; "
                "it varies across the ensemble."),
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
