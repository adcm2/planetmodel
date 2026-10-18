"""A radial reference with a perturbation.

The three kinds evaluate as their formulas and pass the field contract;
the perturbed field is radial when its perturbation is; units convert
and come back; perturbing a model wraps whole layers, refuses a layer
covered in part, and on PREM re-derives the Love moduli from a perturbed
velocity; unperturbing gives the model's fields back.
"""
import numpy as np
import pytest

from planetmodel import (PREM, AnalyticField, GriddedField, PerturbedField,
                         RadialField, constant_field, gauss_legendre, perturbed,
                         sample, unperturbed)
from planetmodel.character import DENSITY, SCALAR, VECTOR
from planetmodel.testing import check_field, check_model, check_sample

INTERVAL = (0.5, 1.0)
REF = RadialField(INTERVAL, lambda r: 3.0 + r ** 2, character=DENSITY, name="rho")


def wobble(r, t, p):
    return 0.1 * r * np.sin(t) * np.cos(p) + 0.05 * np.cos(t)


FORMULA = {"absolute": lambda a, d: a + d, "relative": lambda a, d: a * (1 + d),
           "log": lambda a, d: a * np.exp(d)}


@pytest.mark.parametrize("kind", ["absolute", "relative", "log"])
def test_the_kinds_are_their_formulas(kind):
    char = DENSITY if kind == "absolute" else SCALAR
    d = AnalyticField(INTERVAL, wobble, character=char)
    f = PerturbedField(REF, d, kind=kind)
    check_field(f)
    assert f.reference is REF and f.perturbation is d and f.kind == kind
    assert f.character == DENSITY and f.name == "rho" and not f.is_radial
    rng = np.random.default_rng(0)
    r, t, p = (rng.uniform(*INTERVAL, 40), rng.uniform(0, np.pi, 40),
               rng.uniform(0, 6, 40))
    assert np.allclose(f(r, t, p), FORMULA[kind](REF(r), wobble(r, t, p)))


def test_a_radial_perturbation_gives_a_radial_field():
    d = RadialField(INTERVAL, lambda r: 0.1 * r)
    f = PerturbedField(REF, d, kind="log", name="q")
    check_field(f)
    assert f.is_radial and f.name == "q"
    assert np.allclose(f(np.array([0.6, 0.9])), REF(np.array([0.6, 0.9]))
                       * np.exp(0.1 * np.array([0.6, 0.9])))


def test_a_gridded_perturbation():
    grid = gauss_legendre(8)
    d = GriddedField.sampled(AnalyticField(INTERVAL, wobble), grid, ngll=3,
                             lateral="cubic")
    f = PerturbedField(REF, d, kind="relative")
    check_field(f)
    r, t, p = 0.75, grid.colatitudes[2], grid.longitudes[3]
    assert np.isclose(f(r, t, p), REF(r) * (1 + wobble(r, t, p)), rtol=1e-12)


def test_rescaled_and_restated():
    d = AnalyticField(INTERVAL, wobble)
    f = PerturbedField(REF, d, kind="relative")
    s = f.rescaled(k=2.0, v=3.0)
    assert s.interval == (1.0, 2.0) and s.kind == "relative"
    assert np.isclose(s(1.6, 1.0, 2.0), 3.0 * f(0.8, 1.0, 2.0))
    assert np.isclose(s.perturbation(1.6, 1.0, 2.0), d(0.8, 1.0, 2.0))
    a = PerturbedField(REF, AnalyticField(INTERVAL, wobble, character=DENSITY),
                       kind="absolute")
    assert np.isclose(a.rescaled(k=2.0, v=3.0).perturbation(1.6, 1.0, 2.0),
                      3.0 * wobble(0.8, 1.0, 2.0))
    part = f.on_interval(0.6, 0.9)
    assert part.interval == (0.6, 0.9) and part.reference.interval == (0.6, 0.9)
    assert f.renamed("x").name == "x"


def test_refusals():
    d = AnalyticField(INTERVAL, wobble)
    with pytest.raises(ValueError, match="kind"):
        PerturbedField(REF, d, kind="ratio")
    with pytest.raises(ValueError, match="radial field of rank 0"):
        PerturbedField(AnalyticField(INTERVAL, wobble), d, kind="relative")
    with pytest.raises(ValueError, match="perturbed once"):
        PerturbedField(PerturbedField(REF, d, kind="relative"), d, kind="relative")
    with pytest.raises(ValueError, match="rank 0"):
        PerturbedField(REF, AnalyticField(INTERVAL, lambda r, t, p: np.stack(
            np.broadcast_arrays(r, t, p), -1), character=VECTOR), kind="relative")
    with pytest.raises(ValueError, match="pure number"):
        PerturbedField(REF, AnalyticField(INTERVAL, wobble, character=DENSITY),
                       kind="log")
    with pytest.raises(ValueError, match="character"):
        PerturbedField(REF, d, kind="absolute")
    with pytest.raises(ValueError, match="on_interval"):
        PerturbedField(REF, constant_field((0.5, 0.9), 0.1), kind="relative")


# -- on a model ---------------------------------------------------------------------

LO, HI = 3480e3, 5701e3          # PREM's three lower-mantle layers, whole


def dlnvs(r, t, p):
    return 0.02 * np.sin(t) ** 2 * np.cos(2 * p) * (r - LO) / (HI - LO)


@pytest.fixture(scope="module")
def prem():
    return PREM()


def test_a_perturbed_velocity_rederives_the_moduli(prem):
    m = perturbed(prem, "vsv", AnalyticField((LO, HI), dlnvs), kind="relative")
    check_model(m)
    rng = np.random.default_rng(1)
    for i in (2, 3, 4):
        layer, base = m.layer(i), prem.layer(i)
        assert isinstance(layer["vsv"], PerturbedField)
        assert layer["vsv"].interval == base.interval
        a, b = base.interval
        r = rng.uniform(a, b, 20)
        t, p = rng.uniform(0, np.pi, 20), rng.uniform(0, 6, 20)
        vsv = base["vsv"](r) * (1 + dlnvs(r, t, p))
        assert np.allclose(layer["vsv"](r, t, p), vsv, rtol=1e-12)
        assert np.allclose(layer["L"](r, t, p), base["rho"](r) * vsv ** 2, rtol=1e-12)
        assert np.allclose(layer["N"](r), base["N"](r), rtol=1e-12)
    for i in (0, 1, 5, 12):
        assert m.layer(i)["vsv"] is prem.layer(i)["vsv"]


def test_unperturbed_gives_the_fields_back(prem):
    m = perturbed(prem, "vsv", AnalyticField((LO, HI), dlnvs), kind="relative")
    m = perturbed(m, "rho", AnalyticField((LO, HI), wobble, character=DENSITY),
                  kind="absolute", layers=[3])
    back = unperturbed(m)
    rng = np.random.default_rng(2)
    for i in (2, 3, 4):
        a, b = prem.layer(i).interval
        r = rng.uniform(a, b, 10)
        for name in ("vsv", "rho", "L", "A"):
            f = back.layer(i)[name]
            assert f.is_radial, (i, name)
            assert np.allclose(f(r), prem.layer(i)[name](r), rtol=1e-12), (i, name)


def test_whole_layers_only(prem):
    with pytest.raises(ValueError, match="only in part"):
        perturbed(prem, "vsv", AnalyticField((LO, 5650e3), dlnvs), kind="relative")
    with pytest.raises(ValueError, match="only in part"):
        perturbed(prem, "vsv", AnalyticField((3500e3, 3600e3), dlnvs), kind="relative")
    with pytest.raises(ValueError, match="covers no layer"):
        perturbed(prem, "vsv", AnalyticField((7.0e6, 8.0e6), dlnvs), kind="relative")
    with pytest.raises(ValueError, match="not covered"):
        perturbed(prem, "vsv", AnalyticField((LO, HI), dlnvs), kind="relative",
                  layers=["lid"])
    with pytest.raises(KeyError, match="qmu"):
        perturbed(prem, "qmu", AnalyticField((1221.5e3, 3480e3), dlnvs),
                  kind="relative")
    with pytest.raises(ValueError, match="kind"):
        perturbed(prem, "vsv", AnalyticField((LO, HI), dlnvs), kind="ratio")


def test_units_convert_and_come_back(prem):
    m = perturbed(prem, "vsv", AnalyticField((LO, HI), dlnvs), kind="log")
    nd = m.nondimensionalised()
    back = nd.in_si()
    r, t, p = 4.4e6, 1.1, 2.3
    k = 1.0 / nd.scales.length
    assert np.isclose(nd.layer(3)["vsv"].perturbation(r * k, t, p), dlnvs(r, t, p))
    assert np.isclose(back.layer(3)["vsv"](r, t, p), m.layer(3)["vsv"](r, t, p),
                      rtol=1e-12)


def test_a_perturbed_model_samples(prem):
    m = perturbed(prem, "vsv", AnalyticField((LO, HI), dlnvs), kind="relative")
    s = sample(m, gauss_legendre(3), fields=["vsv", "L", "rho"], ngll=3, drmax=1e6)
    check_sample(s, m)
    assert s.fields["vsv"].ndim == 3 and s.fields["rho"].ndim == 1
