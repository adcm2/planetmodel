"""Power-law creep's two parameters: in the vocabulary, read as rheology
but not frozen, set directly on a model's creeping layers in its units,
and through the model file."""
import numpy as np
import pytest

import planetmodel as pm
from planetmodel import rheology
from planetmodel.conversion import building, profiles, prony
from planetmodel.units import DIMENSIONLESS, MODULUS


def viscous_prem():
    prem = pm.PREM()
    for lay in prem.layers:
        lo, hi = lay.interval
        if 3480e3 <= lo and hi <= 6291e3:
            prem = prem.with_field(lay.index, "viscosity", pm.constant_field(
                (lo, hi), 2e21, name="viscosity"))
    return prem


def test_the_names():
    assert pm.vocabulary.spec("stress_exponent").dimensions == DIMENSIONLESS
    assert pm.vocabulary.spec("transition_stress").dimensions == MODULUS
    assert rheology.is_rheology_name("transition_stress")
    lay = pm.PREM().layer("lower_mantle")
    only = {"stress_exponent": pm.constant_field(lay.interval, 3.0)}
    assert not rheology.is_viscoelastic(only)


def test_set_on_the_creeping_layers_in_the_model_s_units():
    m = viscous_prem()
    stress = profiles.tabulated([3480e3, 6291e3], [3e6, 1e5])
    p = building.with_power_law(m, exponent=3.5, transition_stress=stress)
    held = [lay.index for lay in p.layers if "stress_exponent" in lay]
    assert held == [lay.index for lay in m.layers if "viscosity" in lay]
    i = p.layer("lower_mantle").index
    r = np.array([4000e3, 5500e3])
    assert np.allclose(p.layer(i)["stress_exponent"](r), 3.5)
    assert np.allclose(p.layer(i)["transition_stress"](r), stress(r), rtol=1e-14)
    assert p.layer(i)["transition_stress"].is_radial
    nd = m.nondimensionalised()
    q = building.with_power_law(nd, exponent=3.5, transition_stress=stress)
    k = nd.scales.factor(MODULUS)
    assert np.allclose(k * q.layer(i)["transition_stress"](r / nd.scales.length),
                       stress(r), rtol=1e-12)
    # nonlinear: no part of a frozen model, dropped with the rheology
    omega = 2 * np.pi / 3.15e9
    assert np.allclose(p.frozen(omega).layer(i)["N"](r),
                       m.frozen(omega).layer(i)["N"](r), rtol=1e-14)
    assert "transition_stress" not in p.elastic().layer(i)
    burgers = prony.with_branches(m, prony.maxwell_series())
    assert building.with_power_law(burgers, exponent=3.0,
                                   transition_stress=1e6).layer(i)["stress_exponent"]
    with pytest.raises(ValueError, match="no layer holds"):
        building.with_power_law(pm.PREM(), exponent=3.0, transition_stress=1e6)


@pytest.mark.netcdf
def test_through_the_model_file(tmp_path):
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.modelfile import read_model, write_model
    p = building.with_power_law(viscous_prem(), exponent=3.5, transition_stress=1e6)
    path = write_model(tmp_path / "power.nc", p, grid=pm.gauss_legendre(2))
    pm.testing.check_model_file(path, p)
    back = read_model(path)
    i = p.layer("lower_mantle").index
    assert back.layer(i)["transition_stress"](np.array([5e6])) == pytest.approx(1e6)
