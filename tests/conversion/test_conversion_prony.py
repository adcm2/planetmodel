"""Prony branches: the vocabulary's families, freezing a generalised
Maxwell body, the Maxwell, Burgers and fitted Andrade series against
their closed forms, the branches on a model, its refusals, its units and
the model file."""
import math

import numpy as np
import pytest

import planetmodel as pm
from planetmodel import rheology, vocabulary
from planetmodel.conversion import prony
from planetmodel.units import DIMENSIONLESS, TIME


def viscous_prem():
    prem = pm.PREM()
    for lay in prem.layers:
        lo, hi = lay.interval
        if 3480e3 <= lo and hi <= 6291e3:
            prem = prem.with_field(lay.index, "viscosity", pm.constant_field(
                (lo, hi), 2e21 if hi <= 5701e3 else 5e20, name="viscosity"))
    return prem


def test_the_vocabulary_s_families():
    assert vocabulary.member("relaxation_time_3") == ("relaxation_time", 3)
    assert vocabulary.member("relaxation_strength_12") == ("relaxation_strength", 12)
    for name in ("relaxation_time_0", "relaxation_time_03", "relaxation_time_",
                 "relaxation_time", "relaxation_time_x", "viscosity_1", "vs"):
        assert vocabulary.member(name) is None, name
    assert vocabulary.shipped("relaxation_time_7").dimensions == TIME
    assert vocabulary.spec("relaxation_strength_2").dimensions == DIMENSIONLESS
    assert vocabulary.shipped("vs") is vocabulary.VOCABULARY["vs"]
    with pytest.raises(KeyError, match="families"):
        vocabulary.spec("relaxation_time_0")
    assert rheology.is_rheology_name("relaxation_time_2")
    assert rheology.is_rheology_name("viscosity")
    assert not rheology.is_rheology_name("vsv")


def test_one_branch_freezes_as_the_viscosity_does():
    m = viscous_prem()
    b = prony.with_branches(m, prony.maxwell_series())
    i = m.layer("lower_mantle").index
    assert "viscosity" not in b.layer(i) and "relaxation_time_1" in b.layer(i)
    r = np.linspace(*m.layer(i).interval, 7)
    for omega in (2 * np.pi / (1000 * 3.15e7), 2 * np.pi / 3.15e9):
        a = m.frozen(omega).layer(i)
        c = b.frozen(omega).layer(i)
        for k in ("A", "C", "F", "L", "N"):
            assert np.allclose(c[k](r), a[k](r), rtol=1e-13, atol=0), k
    pm.testing.check_model(b)


def test_burgers_is_its_closed_form():
    strength, ratio = 5.0, 5.0
    s = prony.burgers_series(strength=strength, viscosity_ratio=ratio)
    assert len(s.strengths) == 2 and sum(s.strengths) == pytest.approx(1.0, abs=1e-15)
    w = np.logspace(-4, 4, 41)
    # J / J_M with mu_M = eta_M = 1, mu_K = 1 / strength, eta_K = 1 / ratio
    j = 1.0 + 1.0 / (1j * w) + 1.0 / (1.0 / strength + 1j * w / ratio)
    assert np.allclose(s.modulus(w), 1.0 / j, rtol=1e-13, atol=1e-15)
    # on a model: mu(omega) = mu_U / J(i omega) with the layer's own Maxwell time
    m = viscous_prem()
    b = prony.with_branches(m, s, layers=["lower_mantle"])
    i = m.layer("lower_mantle").index
    r = np.array([5000e3])
    omega = 2 * np.pi / (100 * 3.15e7)
    mu = pm.kappa_mu(m.layer(i))[1](r)
    tau = 2e21 / mu
    want = mu / (1.0 + 1.0 / (1j * omega * tau)
                 + 1.0 / (1.0 / strength + 1j * omega * tau / ratio))
    frozen = b.frozen(omega).layer(i)
    assert np.allclose(frozen["N"](r) - m.layer(i)["N"](r) + mu, want, rtol=1e-12)
    with pytest.raises(ValueError, match="positive"):
        prony.burgers_series(strength=0.0, viscosity_ratio=1.0)


def test_an_andrade_fit_holds_its_misfit_and_flows():
    s = prony.andrade_series(1.0 / 3.0)
    assert s.misfit < 0.02
    assert sum(s.strengths) == pytest.approx(1.0, abs=1e-6)
    assert sum(w * c for w, c in zip(s.strengths, s.times)) == pytest.approx(1.0,
                                                                            abs=1e-6)
    w = np.logspace(-3, 3, 61)
    g = 1.0 + math.gamma(4.0 / 3.0) * (1j * w) ** (-1.0 / 3.0) + 1.0 / (1j * w)
    assert np.max(np.abs(s.modulus(w) * g - 1.0)) <= s.misfit * 1.01
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        prony.andrade_series(1.0)
    with pytest.raises(ValueError, match="band"):
        prony.fitted_series(lambda x: 1.0 + 0 * x, band=(1.0, 0.1))


def test_the_series_refuses_what_is_not_a_body():
    with pytest.raises(ValueError, match="more than the whole"):
        prony.PronySeries((0.7, 0.5), (1.0, 2.0), name="x")
    with pytest.raises(ValueError, match="positive"):
        prony.PronySeries((0.5,), (0.0,), name="x")
    with pytest.raises(ValueError, match="as many"):
        prony.PronySeries((0.5,), (1.0, 2.0), name="x")


def test_freezing_refuses_a_muddled_layer():
    m = viscous_prem()
    i = m.layer("lower_mantle").index
    lay = m.layer(i)
    one = pm.constant_field(lay.interval, 0.5, name="relaxation_strength_1")
    time = pm.constant_field(lay.interval, 1e10, name="relaxation_time_1")
    both = m.with_field(i, "relaxation_strength_1", one).with_field(
        i, "relaxation_time_1", time)
    with pytest.raises(ValueError, match="viscosity and Prony branches"):
        both.frozen(1e-9)
    alone = both.without_field("viscosity", layers=[i]).without_field(
        "relaxation_time_1", layers=[i])
    with pytest.raises(ValueError, match="branch 1 holds"):
        alone.frozen(1e-9)
    gap = both.without_field("viscosity", layers=[i]).with_field(
        i, "relaxation_strength_3", one).with_field(i, "relaxation_time_3", time)
    with pytest.raises(ValueError, match=r"numbered \[1, 3\]"):
        gap.frozen(1e-9)


def test_branches_are_rheology_and_have_units():
    m = prony.with_branches(viscous_prem(), prony.burgers_series(strength=2.0,
                                                                 viscosity_ratio=10.0))
    i = m.layer("lower_mantle").index
    assert "relaxation_time_2" not in m.elastic().layer(i)
    assert m.layer(i).fields.keys() >= {"relaxation_strength_1", "relaxation_time_2"}
    nd = m.nondimensionalised()
    r = np.array([5000e3])
    k = nd.scales.factor(TIME)
    assert np.allclose(k * nd.layer(i)["relaxation_time_2"](r / nd.scales.length),
                       m.layer(i)["relaxation_time_2"](r), rtol=1e-12)
    with pytest.raises(ValueError, match="frozen"):
        m.frozen(1e-9).with_field(i, "relaxation_time_1", pm.constant_field(
            m.layer(i).interval, 1.0, name="relaxation_time_1"), replace=True)
    with pytest.raises(ValueError, match="no solid layer"):
        prony.with_branches(pm.PREM(), prony.maxwell_series())


@pytest.mark.netcdf
def test_branches_write_and_read_back(tmp_path):
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.modelfile import read_model, write_model
    m = prony.with_branches(viscous_prem(), prony.burgers_series(strength=5.0,
                                                                 viscosity_ratio=5.0))
    path = write_model(tmp_path / "prony.nc", m, grid=pm.gauss_legendre(2))
    pm.testing.check_model_file(path, m)
    back = read_model(path)
    i = m.layer("lower_mantle").index
    assert back.spec("relaxation_time_1").dimensions == TIME
    assert len(rheology.branches(back.layer(i))) == 2
