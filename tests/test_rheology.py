"""Rheology from a layer's fields, and a model frozen at a frequency."""
import numpy as np
import pytest

from planetmodel import (constant_field, frozen, frozen_moduli, LayeredIsotropicElastic,
                         is_viscoelastic, kappa_mu, moduli, PREM, testing)
from planetmodel.units import FREQUENCY
from planetmodel.character import DENSITY, Symmetry
from planetmodel.materials import voigt_matrix


def test_is_viscoelastic():
    model = PREM(ocean=False)
    assert all(is_viscoelastic(layer) for layer in model.layers)   # qkappa everywhere
    plain = LayeredIsotropicElastic.homogeneous(1.0, rho=1.0, vp=2.0, vs=1.0)
    assert not is_viscoelastic(plain.layer(0))
    assert frozen_moduli(plain.layer(0), 1.0, reference_omega=1.0)["A"].dtype == float


def test_maxwell_layer_limits():
    rho, mu, a, eta = 5500.0, 1.0e11, 6371e3, 1e21
    model = LayeredIsotropicElastic.homogeneous(a, rho=rho, vp=8000.0,
                                                vs=np.sqrt(mu / rho))
    model = model.with_field(0, "viscosity",
                             constant_field((0.0, a), eta, name="viscosity"))
    layer = model.layer(0)
    base = moduli(layer)
    kappa, mu_f = kappa_mu(layer)
    r = np.array([1e6, 4e6])
    tau = eta / mu
    fast = frozen_moduli(layer, 1e8 / tau, reference_omega=1.0)
    assert fast["L"].dtype == complex
    assert np.allclose(fast["L"](r), base["L"](r), rtol=1e-7)
    assert np.allclose(fast["A"](r), base["A"](r), rtol=1e-7)
    slow = frozen_moduli(layer, 1e-8 / tau, reference_omega=1.0)
    assert np.allclose(slow["L"](r), 0.0, atol=1e-7 * mu)
    assert np.allclose(slow["A"](r), kappa(r), rtol=1e-7)
    mid = frozen_moduli(layer, 1.0 / tau, reference_omega=1.0)
    z = 1j
    want = mu * z / (1.0 + z)
    assert np.allclose(mid["N"](r), want)
    assert np.allclose(mid["F"](r), kappa(r) - 2.0 * want / 3.0)
    for f in mid.values():
        testing.check_field(f)


def test_constant_q_band_on_prem():
    model = PREM(ocean=False)
    omega0 = 2.0 * np.pi
    layer = model.layer("lower_mantle")
    kappa, mu = kappa_mu(layer)
    base = moduli(layer)
    r = np.array([4.0e6, 5.0e6])
    at_ref = frozen_moduli(layer, omega0, reference_omega=omega0)
    qmu, qkappa = layer["qmu"](r), layer["qkappa"](r)
    assert np.allclose(at_ref["L"](r).real, base["L"](r))
    assert np.allclose(at_ref["L"](r).imag, mu(r) / qmu)
    assert np.allclose(at_ref["A"](r).imag,
                       kappa(r) / qkappa + 4.0 * mu(r) / (3.0 * qmu))
    assert np.allclose(at_ref["F"](r).imag,
                       kappa(r) / qkappa - 2.0 * mu(r) / (3.0 * qmu))
    assert np.allclose(at_ref["C"](r).real - at_ref["A"](r).real,
                       base["C"](r) - base["A"](r))               # anisotropy kept real
    lower = frozen_moduli(layer, omega0 / np.e, reference_omega=omega0)
    assert np.allclose(lower["L"](r).real, base["L"](r) - (2.0 / np.pi) * mu(r) / qmu)
    core = model.layer("outer_core")
    fc = frozen_moduli(core, omega0, reference_omega=omega0)
    assert np.allclose(fc["L"](r[:1] * 0.6), 0.0) and fc["A"].dtype == complex


def test_fixed_ratio_attenuation_preserves_voigt_targets_and_ti_ratios():
    radius = 10.0
    model = LayeredIsotropicElastic.homogeneous(radius, rho=4.0, vp=3.0, vs=1.0)
    values = {"A": 30.0, "C": 24.0, "F": 17.0, "L": 3.0, "N": 5.0}
    for name, value in values.items():
        model = model.with_field(0, name, constant_field((0.0, radius), value,
                                                         character=DENSITY, name=name),
                                 replace=True)
    for name, value in (("qkappa", 8.0), ("qmu", 12.0)):
        model = model.with_field(0, name, constant_field((0.0, radius), value, name=name))
    layer = model.layer(0)
    r = np.array([1.0, 5.0])
    additive = frozen_moduli(layer, 4.0, reference_omega=2.0)
    fixed = frozen_moduli(layer, 4.0, reference_omega=2.0,
                          ti_prescription="fixed_ratios")
    explicit_default = frozen_moduli(layer, 4.0, reference_omega=2.0,
                                     ti_prescription="additive")
    assert all(np.array_equal(additive[n](r), explicit_default[n](r))
               for n in values)
    before = {n: values[n] for n in values}
    after = {n: fixed[n](r) for n in values}
    assert np.allclose(after["C"] / after["A"], before["C"] / before["A"])
    assert np.allclose(after["N"] / after["L"], before["N"] / before["L"])
    assert np.allclose(after["F"] / (after["A"] - 2 * after["L"]),
                       before["F"] / (before["A"] - 2 * before["L"]))
    base_k, base_mu = kappa_mu(layer)
    fixed_k, fixed_mu = kappa_mu(fixed)
    add_k, add_mu = kappa_mu(additive)
    assert np.allclose(fixed_k(r) - base_k(r), add_k(r) - base_k(r))
    assert np.allclose(fixed_mu(r) - base_mu(r), add_mu(r) - base_mu(r))
    loss_matrix = voigt_matrix(Symmetry.VTI, {n: fixed[n](r).imag for n in values})
    assert np.all(np.linalg.eigvalsh(loss_matrix) >= -1e-12)
    frozen_model = frozen(model, 4.0, reference_omega=2.0,
                          ti_prescription="fixed_ratios")
    assert all(np.allclose(frozen_model.layer(0)[n](r), fixed[n](r)) for n in values)

    isotropic = LayeredIsotropicElastic.homogeneous(radius, rho=4.0, vp=3.0, vs=1.0)
    isotropic = isotropic.with_field(0, "qkappa", constant_field((0.0, radius), 8.0,
                                                                  name="qkappa"))
    isotropic = isotropic.with_field(0, "qmu", constant_field((0.0, radius), 12.0,
                                                               name="qmu"))
    additive_iso = frozen_moduli(isotropic.layer(0), 4.0, reference_omega=2.0)
    fixed_iso = frozen_moduli(isotropic.layer(0), 4.0, reference_omega=2.0,
                              ti_prescription="fixed_ratios")
    assert all(np.allclose(additive_iso[n](r), fixed_iso[n](r)) for n in values)


def test_fixed_ratio_fluid_and_invalid_prescriptions():
    radius = 10.0
    fluid = LayeredIsotropicElastic.homogeneous(radius, rho=4.0, vp=3.0, vs=0.0)
    fluid = fluid.with_field(0, "qkappa", constant_field((0.0, radius), 10.0,
                                                         name="qkappa"))
    moduli_fixed = frozen_moduli(fluid.layer(0), 1.0, reference_omega=1.0,
                                 ti_prescription="fixed_ratios")
    assert np.allclose(moduli_fixed["A"](5.0), 36.0 + 3.6j)
    assert np.allclose(moduli_fixed["C"](5.0), 36.0 + 3.6j)
    assert moduli_fixed["L"](5.0) == 0.0 and moduli_fixed["N"](5.0) == 0.0
    with pytest.raises(ValueError, match="ti_prescription"):
        frozen_moduli(fluid.layer(0), 1.0, reference_omega=1.0,
                      ti_prescription="other")

    bad = LayeredIsotropicElastic.homogeneous(radius, rho=4.0, vp=3.0, vs=1.0)
    for name, value in {"A": 30.0, "C": 24.0, "F": 17.0, "L": 3.0, "N": 5.0}.items():
        bad = bad.with_field(0, name, constant_field((0.0, radius), value,
                                                      character=DENSITY, name=name),
                             replace=True)
    for name, value in (("qkappa", 0.0), ("qmu", 20.0)):
        bad = bad.with_field(0, name, constant_field((0.0, radius), value, name=name))
    with pytest.raises(ValueError, match="non-passive"):
        frozen_moduli(bad.layer(0), 1.0, reference_omega=1.0,
                      ti_prescription="fixed_ratios")["A"](5.0)


def test_frozen_model():
    model = PREM(ocean=False)
    omega = 2.0 * np.pi / 43200.0
    cold = frozen(model, omega)
    testing.check_model(cold)
    assert cold.constant("omega") == omega and cold.G == model.G
    for layer in cold.layers:
        assert all(n in layer for n in ("A", "C", "F", "L", "N"))
        assert layer["A"].dtype == complex
        assert "vpv" in layer
    # dispersion from the 1 s reference to 12 h softens the mantle by a per cent
    warm = moduli(model.layer("lower_mantle"))["A"](5e6)
    cold_A = moduli(cold.layer("lower_mantle"))["A"](5e6)
    assert 0.97 < cold_A.real / warm < 0.995 and cold_A.imag > 0.0
    nd = frozen(model.nondimensionalised(), omega / 3.0)
    assert nd.constant("omega") == omega / 3.0
    assert np.isclose(nd.in_si().constant("omega"),
                      omega / 3.0 * nd.scales.factor(FREQUENCY))
    plain = LayeredIsotropicElastic.homogeneous(1.0, rho=1.0, vp=2.0, vs=1.0)
    same = frozen(plain, 3.0)
    assert same.constant("omega") == 3.0 and same.layer(0)["vp"].dtype == float
    assert same.layer(0)["A"].dtype == float
    with pytest.raises(ValueError):
        frozen(model, 0.0)
    with pytest.raises(ValueError):
        frozen(model, 1.0, reference_omega=-1.0)


def test_a_frozen_model_is_not_frozen_again():
    cold = frozen(PREM(ocean=False), 1e-3)
    with pytest.raises(ValueError, match="already frozen"):
        frozen(cold, 1e-3)


def test_a_frozen_model_refuses_what_would_leave_its_moduli_stale():
    """Replacing rho, a velocity or a Q field would rebuild or outdate the
    complex moduli, and coarsening would refit them from the real
    velocities; a field the moduli do not depend on is still attached."""
    from planetmodel.rheology import is_frozen

    warm = PREM(ocean=False)
    cold = frozen(warm, 2 * np.pi / 3600.0)
    assert is_frozen(cold) and not is_frozen(warm)
    lay = cold.layer("lower_mantle")
    iv = lay.interval
    for name in ("rho", "vsv", "L", "qmu"):
        with pytest.raises(ValueError, match=f"frozen at omega.*derived from '{name}'"):
            cold.with_field("lower_mantle", name, lay[name], replace=True)
    with pytest.raises(ValueError, match="frozen at omega.*viscosity"):
        cold.with_field("lower_mantle", "viscosity",
                        constant_field(iv, 1e21, name="viscosity"))
    with pytest.raises(ValueError, match="frozen at omega.*coarsening"):
        cold.coarsened(drop=["d771"])
    tagged = cold.with_field("lower_mantle", "tracer", constant_field(iv, 1.0))
    assert tagged.layer("lower_mantle")["L"] is lay["L"]
    # the way out the refusal names: change the elastic model, freeze again
    vsv = warm.layer("lower_mantle")["vsv"]
    slow = warm.with_field("lower_mantle", "vsv", 0.9 * vsv, replace=True)
    assert frozen(slow, 2 * np.pi / 3600.0).layer("lower_mantle")["L"].dtype == complex


def test_moduli_conversions_keep_complex_numbers():
    from planetmodel.materials import kappa_mu_from_moduli, moduli_from_velocities

    cold = frozen(PREM(ocean=False), 1e-3).layer("lower_mantle")
    r = 5e6
    five = [cold[n](r) for n in ("A", "C", "F", "L", "N")]
    kappa, mu = kappa_mu_from_moduli(*five)
    k_field, mu_field = kappa_mu(cold)
    assert np.iscomplexobj(mu) and np.isclose(mu, mu_field(r), rtol=1e-12)
    assert np.isclose(kappa, k_field(r), rtol=1e-12)
    out = moduli_from_velocities(1.0, 2.0 + 0.1j, 1.0)
    assert np.isclose(out["C"], (2.0 + 0.1j) ** 2)
