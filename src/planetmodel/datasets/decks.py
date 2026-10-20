"""Radial models published as decks, read into models.

A radial reference (STW105, ak135-F, iasp91, the Moon's models) is a deck
in some format of `planetmodel.deck`: a mineos deck, TauP's named
discontinuities, its tabled velocities.  `DeckReader` is the reader of
such a data set: the deck read by its format, its columns joined layer
by layer by the interpolation the source intends (cubic splines for a
mineos deck, straight lines for TauP's tables, which TauP itself
interpolates linearly), and the model built either by a model type that
takes a deck (`MineosModel`, with the elastic behaviours) or as a plain
`Model` of the deck's fields, its layers and interfaces named by the
format's rule (`deck.nd_names`).  SI, as decks are, or the caller's
scales.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..deck import Deck, DeckFormat, deck_layers, read_deck
from ..geometry import Geometry
from ..model import Model
from ..units import Scales

if TYPE_CHECKING:
    from .registry import Dataset

__all__ = ["DeckReader", "Names"]

#: A rule naming a deck's layers and interfaces, as `deck.mineos_names` does.
type Names = Callable[[Deck], tuple[list[str | None], list[str | None]]]


@dataclass(frozen=True)
class DeckReader:
    """The reader of a data set that is one deck in `format`: built by
    `model` (a model type taking the deck and `kind=`), or as a plain
    model of its fields named by `names`, the knots joined by `kind`."""

    format: DeckFormat
    _: KW_ONLY
    model: Callable[..., Model] | None = None
    names: Names | None = None
    kind: str = "cubic"

    def read(self, paths: tuple[Path, ...], dataset: Dataset, *,
             kind: str | None = None, scales: Scales | None = None) -> Model:
        """The data set's model: its one file read as a deck, joined by
        `kind` (the reader's own by default), in SI or `scales`."""
        if len(paths) != 1:
            raise ValueError(f"{dataset.name}: a deck is one file, given {len(paths)}")
        deck = read_deck(paths[0], self.format)
        kind = self.kind if kind is None else kind
        if self.model is not None:
            model = self.model(deck, kind=kind)
        else:
            skeleton, layers = deck_layers(deck, kind=kind)
            layer_names, interface_names = (self.names(deck) if self.names is not None
                                            else (None, None))
            geometry = Geometry(skeleton, layer_names=layer_names,
                                interface_names=interface_names)
            model = Model(geometry, layers, name=dataset.name)
        model = model.with_citation(*dataset.citation)
        return model if scales is None else model.converted(scales)
