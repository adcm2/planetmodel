"""The Woodhouse-Ritsema `.sph` files, against their own definitions.

The oracles are independent statements: Woodhouse's `rspln` and `rsple`
transliterated line by line for the radial splines, and the harmonics
written out from their definition with scipy's associated Legendre
functions (which carry the Condon-Shortley phase, as the files' do).
Files are written here in the format; no published file is used.
"""
import hashlib
from math import factorial

import numpy as np
import pytest
from scipy.special import lpmv

import planetmodel as pm
from planetmodel import datasets
from planetmodel.datasets import (MANTLE_SLOTS, REGISTRY, SPH_RADII, SPLINE_KNOTS,
                                  Dataset, Source, SphReader, read_sph, sph_field,
                                  sph_splines)
from planetmodel.materials import voigt_vp, voigt_vs
from planetmodel.testing import check_dataset

CMB, MOHO = SPH_RADII


# -- Woodhouse's routines, transliterated (one strictly increasing segment) --

def rspln(x, y):
    """q(3, n) of rspln for strictly increasing x, 1-based as the Fortran."""
    n = len(x)
    x = np.concatenate([[np.nan], x])
    y = np.concatenate([[np.nan], y])
    q = np.zeros((4, n + 1))
    f = np.zeros((4, n + 1))
    j1, j2 = 1, n - 2
    a0 = 0.0
    h = x[j1 + 1] - x[j1]
    h2 = x[j1 + 2] - x[j1]
    y0 = h * h2 * (h2 - h)
    h, h2 = h * h, h2 * h2
    b0 = (y[j1] * (h - h2) + y[j1 + 1] * h2 - y[j1 + 2] * h) / y0
    b1 = b0
    for i in range(j1, j2 + 1):
        h = x[i + 1] - x[i]
        y0 = y[i + 1] - y[i]
        h2 = h * h
        ha = h - a0
        h2a = h - 2.0 * a0
        h3a = 2.0 * h - 3.0 * a0
        h2b = h2 * b0
        q[1, i] = h2 / ha
        q[2, i] = -ha / (h2a * h2)
        q[3, i] = -h * h2a / h3a
        f[1, i] = (y0 - h * b0) / (h * ha)
        f[2, i] = (h2b - y0 * (2.0 * h - a0)) / (h * h2 * h2a)
        f[3, i] = -(h2b - 3.0 * y0 * ha) / (h * h3a)
        a0 = q[3, i]
        b0 = f[3, i]
    i = j2 + 1
    h = x[i + 1] - x[i]
    y0 = y[i + 1] - y[i]
    h2 = h * h
    ha = h - a0
    h2a = h * ha
    h2b = h2 * b0 - y0 * (2.0 * h - a0)
    q[1, i] = h2 / ha
    f[1, i] = (y0 - h * b0) / h2a
    ha = x[j2] - x[i + 1]
    y0 = -h * ha * (ha + h)
    ha = ha * ha
    y0 = (y[i + 1] * (h2 - ha) + y[i] * ha - y[j2] * h2) / y0
    q[3, i] = (y0 * h2a + h2b) / (h * h2 * (h - 2.0 * a0))
    q[2, i] = f[1, i] - q[1, i] * q[3, i]
    for _ in range(j1, j2 + 1):
        k = i - 1
        q[1, i] = f[3, k] - q[3, k] * q[2, i]
        q[3, k] = f[2, k] - q[2, k] * q[1, i]
        q[2, k] = f[1, k] - q[1, k] * q[3, k]
        i = k
    q[1, i] = b1
    j2 = j2 + 2
    q[1, j2], q[2, j2], q[3, j2] = y0, 0.0, 0.0
    return q


def rsple(x, y, q, s):
    """rsple at points s inside [x_1, x_n]."""
    n = len(x)
    xs = np.concatenate([[np.nan], x])
    ys = np.concatenate([[np.nan], y])
    i = np.clip(np.searchsorted(x, s, side="right"), 1, n - 1)
    h = s - xs[i]
    return ys[i] + h * (q[1, i] + h * (q[2, i] + h * q[3, i]))


def woodhouse_splines(xs):
    """S_k(x) for every knot k, as the Fortran builds them: (21, len(xs))."""
    out = []
    for k in range(SPLINE_KNOTS.size):
        y = np.zeros(SPLINE_KNOTS.size)
        y[k] = 1.0
        out.append(rsple(SPLINE_KNOTS, y, rspln(SPLINE_KNOTS, y), xs))
    return np.array(out)


def x_of(r):
    return -1.0 + 2.0 * (r - CMB) / (MOHO - CMB)


def test_the_radial_splines_are_woodhouses():
    r = np.concatenate([np.linspace(CMB, MOHO, 401), CMB + 0.5 * (SPLINE_KNOTS + 1)
                        * (MOHO - CMB)])
    got = sph_splines()(r)                           # (points, 21)
    want = woodhouse_splines(x_of(r))
    # the same splines, Woodhouse's elimination in x against scipy's in metres:
    # they agree to rounding against the system's conditioning (4e-12 measured)
    assert np.allclose(got.T, want, rtol=0.0, atol=1e-11)
    # cardinal: one at their own knot, zero at the others; and C2 inside
    knots = CMB + 0.5 * (SPLINE_KNOTS + 1) * (MOHO - CMB)
    assert np.allclose(sph_splines()(knots), np.eye(21), atol=1e-13)
    s = sph_splines()
    for nu in (1, 2):
        d = s.derivative(nu)
        inner = knots[1:-1]
        assert np.allclose(d(inner - 1e-3), d(inner + 1e-3), rtol=1e-6, atol=1e-12)


# -- the format ----------------------------------------------------------------

def sph_lines(blocks, lmax, *, degrees=None, slots=MANTLE_SLOTS, header=True):
    """A .sph file's lines: the header, then each present slot's block,
    each present degree starting a line, eleven values to a line."""
    degrees = degrees or [True] * (lmax + 1)
    present = "".join("1" if i + 1 in slots else "0" for i in range(24))
    lines = []
    if header == "lmax":
        lines.append(f"{'':10}{lmax:5d}")
    elif header:
        lines.append(f"{'':10}{lmax:5d} {''.join('1' if d else '0' for d in degrees)}"
                     f" {24:3d} {present}")
    for block in blocks:
        i = 0
        for l in range(lmax + 1):
            row = block[i:i + 2 * l + 1]
            i += 2 * l + 1
            if not degrees[l]:
                continue
            for j in range(0, row.size, 11):
                lines.append("".join(f"{v:16.8e}" for v in row[j:j + 11]))
    return lines


def file_field(blocks, lmax, r, theta, phi):
    """The file's model from its definition: the rspln splines times
    sum (a cos + b sin) X_lm, X_lm with the Condon-Shortley phase, no sqrt 2."""
    S = woodhouse_splines(x_of(r))[::-1]            # blocks run from the Moho down
    out = np.zeros_like(r)
    for k, block in enumerate(blocks):
        i = 0
        for l in range(lmax + 1):
            for m in range(l + 1):
                X = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m)
                            / factorial(l + m)) * lpmv(m, l, np.cos(theta))
                if m == 0:
                    out += S[k] * block[i] * X
                    i += 1
                else:
                    out += S[k] * X * (block[i] * np.cos(m * phi)
                                       + block[i + 1] * np.sin(m * phi))
                    i += 2
    return out


def random_blocks(lmax, rng):
    return 1e-2 * rng.standard_normal((21, (lmax + 1) ** 2))


def points(rng, n=300):
    r = rng.uniform(CMB, MOHO, n)
    return r, np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, 2 * np.pi, n)


def test_a_file_reads_to_its_own_definition():
    rng = np.random.default_rng(7)
    lmax = 6
    blocks = random_blocks(lmax, rng)
    data = read_sph(sph_lines(blocks, lmax))
    assert data.lmax == lmax and data.slots == MANTLE_SLOTS and data.nslots == 24
    assert np.array_equal(data.blocks, np.array([[float(f"{v:16.8e}") for v in b]
                                                  for b in blocks]))
    field = sph_field(data)
    assert field.interval == (CMB, MOHO)
    r, theta, phi = points(rng)
    want = file_field(data.blocks, lmax, r, theta, phi)
    assert np.allclose(field(r, theta, phi), want, rtol=0.0,
                       atol=1e-10 * np.abs(want).max())


def test_an_absent_degree_reads_as_zero_and_lmax_alone_means_all():
    rng = np.random.default_rng(8)
    lmax = 4
    blocks = random_blocks(lmax, rng)
    degrees = [True, False, True, True, False]
    data = read_sph(sph_lines(blocks, lmax, degrees=degrees))
    for l, kept in enumerate(degrees):
        cut = slice(l * l, (l + 1) ** 2)
        if kept:
            assert np.allclose(data.blocks[:, cut], blocks[:, cut], rtol=1e-7)
        else:
            assert not data.blocks[:, cut].any()
    whole = read_sph(sph_lines(blocks, lmax, header="lmax"))
    assert whole.degrees == (True,) * 5 and whole.slots == MANTLE_SLOTS
    bare = read_sph(sph_lines(blocks, lmax, header=False), lmax=lmax)
    assert np.array_equal(bare.blocks, whole.blocks)


def test_malformed_files_are_refused_by_name(tmp_path):
    rng = np.random.default_rng(9)
    blocks = random_blocks(3, rng)
    lines = sph_lines(blocks, 3)
    with pytest.raises(ValueError, match="give lmax="):
        read_sph(sph_lines(blocks, 3, header=False))
    with pytest.raises(ValueError, match="says lmax 3, not 4"):
        read_sph(lines, lmax=4)
    with pytest.raises(ValueError, match="asks 21 blocks of 16"):
        read_sph(lines[:-1])
    with pytest.raises(ValueError, match="degree mask"):
        read_sph([lines[0].replace("1111 ", "1121 ")] + lines[1:])
    with pytest.raises(ValueError, match="no lines"):
        read_sph([])
    crust = read_sph(sph_lines(random_blocks(2, rng)[:3], 2, slots=(1, 2, 3)))
    assert crust.slots == (1, 2, 3)
    with pytest.raises(ValueError, match="21 mantle slots"):
        sph_field(crust)
    path = tmp_path / "m.sph"
    path.write_text("\n".join(lines) + "\n")
    assert read_sph(path).source == str(path)


# -- the data set as a model ---------------------------------------------------

def sph_dataset(tmp_path, files, parameters=("vs",), reference="PREM", name="T"):
    sources = []
    for i, lines in enumerate(files):
        path = tmp_path / f"{name}{i}.sph"
        path.write_text("\n".join(lines) + "\n")
        sources.append(Source(path.as_uri(),
                              hashlib.sha256(path.read_bytes()).hexdigest()))
    return Dataset(name, tuple(sources), SphReader(parameters=parameters),
                   title="t", citation=("c",), licence="l", reference=reference)


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANETMODEL_DATA", str(tmp_path / "cache"))


def test_a_data_set_loads_as_the_model_it_is(tmp_path):
    rng = np.random.default_rng(10)
    blocks = random_blocks(5, rng)
    d = sph_dataset(tmp_path, [sph_lines(blocks, 5)])
    m = datasets.load(d)
    assert isinstance(m, pm.Model) and m.name == "T"
    assert m.skeleton.boundaries[0] == CMB and m.skeleton.boundaries[-1] == 6371e3
    prem = pm.PREM()
    mantle = [lay for lay in m.layers if lay.interval[1] <= MOHO]
    assert [lay.name for lay in mantle] == [lay.name for lay in prem.layers[2:10]]
    assert all(not lay.names for lay in m.layers[len(mantle):])
    field = sph_field(read_sph(sph_lines(blocks, 5)))
    r, theta, phi = points(rng, 50)
    for lay in mantle:
        vs = lay["vs"]
        assert isinstance(vs, pm.PerturbedField) and vs.kind == "relative"
        ref = prem.layer(lay.name)
        rr = np.clip(r, *lay.interval)
        want = voigt_vs(ref["vsv"](rr), ref["vsh"](rr))
        assert np.allclose(vs.reference(rr), want, rtol=1e-14)
        assert np.allclose(vs.perturbation(rr, theta, phi), field(rr, theta, phi),
                           rtol=0.0, atol=1e-15)
        assert np.allclose(vs(rr, theta, phi), want * (1 + field(rr, theta, phi)),
                           rtol=1e-13)
    pm.testing.check_model(m)
    check_dataset(d, paths=datasets.fetch(d))


def test_scales_and_a_reference_of_ones_own(tmp_path):
    rng = np.random.default_rng(11)
    blocks = random_blocks(3, rng)
    d = sph_dataset(tmp_path, [sph_lines(blocks, 3)])
    scales = pm.Scales.geophysical(6371e3)
    m = datasets.load(d, scales=scales)
    assert m.scales == scales
    assert m.skeleton.boundaries[0] == pytest.approx(CMB / 6371e3)
    simple = pm.LayeredIsotropicElastic([0.0, 3480e3, 5000e3, 6371e3],
                                        rho=[1e4, 5e3, 3e3], vp=[9e3, 1.1e4, 8e3],
                                        vs=[0.0, 6e3, 4.5e3])
    own = datasets.load(d, reference=simple)
    assert [lay.interval for lay in own.layers] == [(3480e3, 5000e3), (5000e3, MOHO),
                                                    (MOHO, 6371e3)]
    assert own.layers[1]["vs"].reference(5500e3) == pytest.approx(4.5e3)
    density_only = pm.Model(pm.Geometry(pm.Skeleton([0.0, 6371e3])),
                            [{"rho": pm.constant_field((0.0, 6371e3), 5e3,
                                                       character=pm.DENSITY)}])
    with pytest.raises(ValueError, match="holds no 'vs'"):
        datasets.load(d, reference=density_only)
    with pytest.raises(ValueError, match="names no reference"):
        datasets.load(sph_dataset(tmp_path, [sph_lines(blocks, 3)], reference=None,
                                  name="N"))


def test_a_p_and_s_model_holds_both(tmp_path):
    rng = np.random.default_rng(12)
    s, p = random_blocks(2, rng), random_blocks(2, rng)
    d = sph_dataset(tmp_path, [sph_lines(s, 2), sph_lines(p, 2)],
                    parameters=("vs", "vp"), name="SP")
    m = datasets.load(d)
    lay = m.layer("lower_mantle")
    assert lay.names == ("vs", "vp")
    ref = pm.PREM().layer("lower_mantle")
    assert lay["vp"].reference(5e6) == pytest.approx(
        float(voigt_vp(ref["vpv"](5e6), ref["vph"](5e6))), rel=1e-14)
    r, theta, phi = points(rng, 20)
    rr = np.clip(r, *lay.interval)
    assert np.allclose(lay["vp"].perturbation(rr, theta, phi),
                       sph_field(read_sph(sph_lines(p, 2)))(rr, theta, phi), atol=1e-15)
    with pytest.raises(ValueError, match="2 files for the parameters"):
        SphReader().read(datasets.fetch(d), d)


def test_a_support_inside_layers_is_cut_out_of_them():
    pert = pm.HarmonicField((4000e3, 6000e3), np.full((2, 2, 2), 0.0))
    m = datasets.over_reference(pm.PREM(), {"vs": pert}, kind="relative", name="cut")
    assert m.skeleton.boundaries[0] == 4000e3
    assert 6000e3 in m.skeleton.boundaries
    held = [lay.interval for lay in m.layers if lay.names]
    assert held[0] == (4000e3, 5600e3) and held[-1][1] == 6000e3
    assert not m.layer(len(held)).names
    with pytest.raises(ValueError, match="different intervals"):
        datasets.over_reference(pm.PREM(), {
            "vs": pert, "vp": pm.HarmonicField((4000e3, 6100e3), np.zeros((2, 2, 2)))},
            kind="relative")


def test_the_voigt_averages():
    assert voigt_vs(4.0, 4.0) == 4.0 and voigt_vp(8.0, 8.0) == 8.0
    assert voigt_vs(3.0, 6.0) == pytest.approx(np.sqrt((18 + 36) / 3))
    assert voigt_vp(5.0, 10.0) == pytest.approx(np.sqrt((25 + 400) / 5))


def test_the_shipped_entries_hold_to_their_contract():
    for name in ("S20RTS", "S40RTS", "S12RTS", "SP12RTS"):
        d = REGISTRY[name]
        check_dataset(d)
        assert d.reference == "PREM" and d.licence == "CC BY 4.0"
        assert all(s.url.startswith("https://zenodo.org/records/11212270/")
                   for s in d.sources)
    assert REGISTRY["SP12RTS"].reader.parameters == ("vs", "vp")
