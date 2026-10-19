"""Published models, fetched on demand and read into the library's own objects.

A data set is a model published elsewhere: a tomographic model, a crust
or lithosphere model, a radial reference, a surface of some body.  The
library keeps an entry for it, never its data: a `Dataset` names its
files (`Source`s, each with its URL, sha256 and mirrors, and for an
archive the members wanted), the reader that turns them into the object
the data set is, the model its perturbations are relative to, and its
title, citation and licence as the source states them.  `REGISTRY`
holds the shipped entries by name; a caller's own `Dataset` is used
exactly as a shipped one, passed where a name would go.

    fetch("S40RTS")              # paths of its files, cached and checked
    load("S40RTS")               # the data set as the object it is: a model
    describe("S40RTS"), cite("S40RTS")

A data set of perturbations reads as the model it is (`over_reference`):
the reference's geometry from the inner end of the perturbations'
support to the surface, each perturbed name a `PerturbedField` over the
reference's field on the layers the support covers, nothing beyond, so
that slices, pictures and the model file take it as it stands.  The
reference is the entry's (`REFERENCES` from the catalogue, or another
data set) or the caller's.  The Woodhouse-Ritsema `.sph` files (S20RTS,
S40RTS, S12RTS, SP12RTS) are read exactly, as harmonic fields on their
radial splines (`read_sph`, `sph_field`, `SphReader`); radial references
published as decks (STW105, ak135-F, iasp91, the Moon's models) read
through `planetmodel.deck` by `DeckReader`; netCDF grids of latitude and
longitude, EMC's among them, normalised by `read_grid` into a model of a
global grid (`grid_model`, `GridReader`) or a `RegionalGrid`.  The
processing that fits them to a model of one's own (`rereferenced`,
`spread`, `filtered`, `merged`, one taper `raised_cosine`) is in
`planetmodel.datasets.processing`, and `transplanted` takes them in their
order to put a data set's perturbations on a model of one's own:

    transplanted(PREM(), "S40RTS", lmax=20)    # vsv, vsh perturbed; L, N with them

`fetch` downloads into `cache_directory()` (`PLANETMODEL_DATA`, or the
platform's per-user cache), checks each file's sha256 every time it is
used, refuses one that differs unless asked to keep it, never turns off
certificate checks, and leaves nothing behind when interrupted; a file
placed in the cache by hand is used as fetched.  `load` hands the paths
to the entry's reader.  Everything here is the standard library, numpy
and scipy; a reader needing an extra imports it where it is used.  The
sub-package is reached by module, off the top-level surface.
"""
from .building import PAIRS, transplanted
from .decks import DeckReader, Names
from .entries import REGISTRY, DatasetLike, as_dataset, cite, describe
from .grids import (EARTH_RADIUS, SAME_RTOL, VERTICAL_RTOL, GridData, GridReader,
                    RegionalGrid, Variables, grid_model, read_grid)
from .fetch import (CACHE_VARIABLE, OFFLINE_VARIABLE, TIMEOUT, cache_directory, fetch,
                    load, sha256_of)
from .models import (REFERENCES, Segments, over_reference, reference_field,
                     reference_of, standalone)
from .processing import (MODES, Perturbation, demeaned, filtered, merged,
                         raised_cosine, rereferenced, resampled, spread)
from .registry import ARCHIVES, Dataset, Reader, Source, archive_stem
from .sph import (MANTLE_SLOTS, SLOTS, SPH_RADII, SPLINE_KNOTS, SphFile, SphReader,
                  read_sph, sph_field, sph_splines)

__all__ = ["Source", "Dataset", "Reader", "DatasetLike", "REGISTRY", "ARCHIVES",
           "as_dataset", "archive_stem", "describe", "cite", "cache_directory",
           "fetch", "load", "sha256_of", "CACHE_VARIABLE", "OFFLINE_VARIABLE",
           "TIMEOUT", "REFERENCES", "reference_of", "reference_field",
           "over_reference", "SphFile", "SphReader", "read_sph", "sph_field",
           "sph_splines", "SPLINE_KNOTS", "SPH_RADII", "SLOTS", "MANTLE_SLOTS",
           "DeckReader", "Names", "Segments", "standalone", "GridData", "GridReader",
           "RegionalGrid", "Variables", "grid_model", "read_grid", "EARTH_RADIUS",
           "SAME_RTOL", "VERTICAL_RTOL", "MODES", "Perturbation", "filtered",
           "merged", "raised_cosine", "rereferenced", "spread", "resampled",
           "demeaned", "transplanted", "PAIRS"]
