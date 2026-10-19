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
    load("S40RTS")               # the data set as the object it is
    describe("S40RTS"), cite("S40RTS")

`fetch` downloads into `cache_directory()` (`PLANETMODEL_DATA`, or the
platform's per-user cache), checks each file's sha256 every time it is
used, refuses one that differs unless asked to keep it, never turns off
certificate checks, and leaves nothing behind when interrupted; a file
placed in the cache by hand is used as fetched.  `load` hands the paths
to the entry's reader.  Everything here is the standard library, numpy
and scipy; a reader needing an extra imports it where it is used.  The
sub-package is reached by module, off the top-level surface.
"""
from .fetch import (CACHE_VARIABLE, OFFLINE_VARIABLE, TIMEOUT, cache_directory, fetch,
                    load, sha256_of)
from .registry import (ARCHIVES, REGISTRY, Dataset, DatasetLike, Reader, Source,
                       archive_stem, as_dataset, cite, describe)

__all__ = ["Source", "Dataset", "Reader", "DatasetLike", "REGISTRY", "ARCHIVES",
           "as_dataset", "archive_stem", "describe", "cite", "cache_directory",
           "fetch", "load", "sha256_of", "CACHE_VARIABLE", "OFFLINE_VARIABLE",
           "TIMEOUT"]
