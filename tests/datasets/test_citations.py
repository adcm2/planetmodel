"""What a data set's products cite, and what a model built from them
cites: the entry's works on every product a reader gives, a model of
perturbations its reference's after them, and a model with a data set
transplanted or merged into it its own and then the data set's."""
import numpy as np
import pytest

import planetmodel as pm
from planetmodel.datasets import (GridData, RegionalGrid, Strata, Surface, merged,
                                  over_reference, standalone, transplanted)

PREM_WORK = pm.PREM().citation[0]


def flat(value):
    return lambda t, p: np.full(np.broadcast(t, p).shape, value)


def test_a_model_of_perturbations_cites_its_own_then_its_reference_s():
    pert = pm.AnalyticField((3480e3, 6346.6e3),
                            lambda r, t, p: 0.01 * np.cos(t) + 0 * r)
    s = over_reference(pm.PREM(), {"vs": pert}, kind="relative", citation=["S."])
    assert s.citation == ("S.", PREM_WORK)
    m = transplanted(pm.PREM(), s)
    assert m.citation == (PREM_WORK, "S.")
    other = pm.LayeredIsotropicElastic([0.0, 3480e3, 6346.6e3, 6371e3],
                                       rho=[1e4, 4e3, 3e3], vp=[8e3, 1e4, 7e3],
                                       vs=[0.0, 6e3, 4e3])
    assert transplanted(other, s).citation == ("S.", PREM_WORK)
    v = pm.AnalyticField((3480e3, 6346.6e3), lambda r, t, p: 5e3 + 0 * r * t,
                         name="vs")
    assert standalone({"vs": v}, surface=6371e3, citation=("V.",)).citation == ("V.",)


def test_a_merged_model_cites_the_regional_grid_after_its_own():
    lats, lons = np.arange(40.0, 60.1), np.arange(-20.0, 20.1)
    radius = 6371e3 - np.array([500e3, 25e3])
    theta = np.radians(90.0 - lats[::-1])
    phi = np.radians(lons)
    values = np.full((2, theta.size, phi.size), 5.0)
    data = GridData(theta, phi, radius, {"vsv": values}, {"vsv": "km/s"})
    box = RegionalGrid(data, {"vsv": ("vsv", None)}, citation=["Box, 2026."])
    assert box.citation == ("Box, 2026.",) and box.cite() == "Box, 2026."
    m = merged(pm.PREM(), box, name="vsv", width=np.radians(4.0),
               grid=pm.equiangular(45, 90))
    assert m.citation == (PREM_WORK, "Box, 2026.")


def test_surfaces_and_strata_cite_and_keep_it():
    top = Surface(flat(0.0), "depth", datum=6371e3, name="top", citation=["T."])
    assert top.cite() == "T." and top.with_datum(6370e3).citation == ("T.",)
    assert top.sampled(pm.equiangular(8, 16)).citation == ("T.",)
    assert top.with_citation("U.", "T.").citation == ("T.", "U.")
    with pytest.raises(TypeError, match="not one string"):
        Surface(flat(0.0), "radius", citation="T.")
    bottom = Surface(flat(30e3), "depth", datum=6371e3, name="bottom")
    strata = Strata.from_surfaces([top, bottom], ["crust"], {}, name="one")
    assert strata.citation == ()
    cited = strata.with_citation("C.")
    assert cited.citation == ("C.",) and cited.cite() == "C."
    assert [s.citation for s in cited.surfaces] == [("T.", "C."), ("C.",)]
    assert cited["crust"].bottom is cited.surfaces[1]
    assert np.allclose(cited["crust"].thickness(1.0, 2.0), 30e3)
