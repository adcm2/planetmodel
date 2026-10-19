"""Surfaces and their coefficient tables: tables the tests write, in each
format and convention, read to the function the convention evaluates;
partial reads; what a surface measures and from what; the reader of an
entry's tables.  Nothing here touches the network."""
import gzip
from math import factorial

import numpy as np
import pytest
from scipy.special import lpmv

from planetmodel import GriddedFunction, HarmonicFunction, gauss_legendre
from planetmodel.datasets import (CoefficientReader, Dataset, Source, Surface,
                                  read_coefficients)
from planetmodel.harmonics import SCIPY_LMAX, synthesise, to_convention
from planetmodel.testing import check_dataset, check_surface

L = 8


def library_coefficients(lmax=L, seed=0):
    """Random coefficients of the library's basis, zero where no harmonic is."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(2, lmax + 1, lmax + 1))
    l, m = np.meshgrid(np.arange(lmax + 1), np.arange(lmax + 1), indexing="ij")
    c[:, m > l] = 0.0
    c[1, :, 0] = 0.0
    return c


def own_evaluation(C, S, normalization, csphase, theta, phi):
    """sum C_lm Y_lm^c + S_lm Y_lm^s in a convention, by its own statement:
    the associated Legendre functions from scipy's `lpmv`, which carry the
    Condon-Shortley phase, times the convention's normalisation."""
    out = np.zeros(np.broadcast(theta, phi).shape)
    x = np.cos(theta)
    for l in range(C.shape[0]):
        for m in range(l + 1):
            p = lpmv(m, l, x) * (-1.0) ** m          # without the phase
            if csphase == -1:
                p = p * (-1.0) ** m
            ratio = factorial(l - m) / factorial(l + m)
            if normalization == "4pi":
                n = np.sqrt((2 - (m == 0)) * (2 * l + 1) * ratio)
            elif normalization == "schmidt":
                n = np.sqrt((2 - (m == 0)) * ratio)
            else:                                     # "ortho"
                n = np.sqrt((2 - (m == 0)) * (2 * l + 1) / (4 * np.pi) * ratio)
            out = out + n * p * (C[l, m] * np.cos(m * phi) + S[l, m] * np.sin(m * phi))
    return out


def write_shtools(path, C, S, *, header=True, fortran=False, sigmas=True):
    lines = (["# a header, R0 = 1737151.0, GM", "1737151.0, 4.9e12, 310"] if header
             else [])
    for l in range(C.shape[0]):
        for m in range(l + 1):
            row = f"{l:5d} {m:4d} {C[l, m]: .15E} {S[l, m]: .15E}"
            if fortran:
                row = row.replace("E", "D")
            if sigmas:
                row += "  1.0E-3  1.0E-3"
            lines.append(row)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_bshc(path, C, S, *, lmin=0, gz=False):
    lmax = C.shape[0] - 1
    l, m = np.tril_indices(lmax + 1)
    keep = l >= lmin
    data = np.concatenate([[lmin, lmax], C[l[keep], m[keep]],
                           S[l[keep], m[keep]]]).astype("<f8").tobytes()
    if gz:
        with gzip.open(path, "wb") as f:
            f.write(data)
    else:
        path.write_bytes(data)
    return path


def in_convention(c, normalization, csphase):
    k = to_convention(c, normalization=normalization, csphase=csphase)
    return k[0], k[1]


@pytest.mark.parametrize("normalization, csphase", [("4pi", 1), ("4pi", -1),
                                                    ("schmidt", 1), ("ortho", 1)])
@pytest.mark.parametrize("form", ["shtools", "shtools-fortran", "bshc", "bshc.gz"])
def test_a_table_reads_to_the_function_its_convention_evaluates(tmp_path, form,
                                                                normalization,
                                                                csphase):
    c = library_coefficients()
    C, S = in_convention(c, normalization, csphase)
    if form.startswith("shtools"):
        path = write_shtools(tmp_path / "t.sh", C, S, fortran=form.endswith("fortran"))
        fmt = "shtools"
    else:
        path = write_bshc(tmp_path / f"t.{form}", C, S, gz=form.endswith("gz"))
        fmt = "bshc"
    f = read_coefficients(path, format=fmt, normalization=normalization,
                          csphase=csphase)
    assert isinstance(f, HarmonicFunction) and f.lmax == L
    assert np.allclose(f.coefficients, c, rtol=0, atol=1e-13)
    rng = np.random.default_rng(1)
    t, p = np.arccos(rng.uniform(-1, 1, 40)), rng.uniform(0, 2 * np.pi, 40)
    assert np.allclose(f(t, p), own_evaluation(C, S, normalization, csphase, t, p),
                       rtol=0, atol=1e-12)


def test_a_table_is_read_only_as_far_as_asked(tmp_path):
    c = library_coefficients()
    C, S = in_convention(c, "4pi", 1)
    for path, fmt in ((write_bshc(tmp_path / "t.bshc.gz", C, S, gz=True), "bshc"),
                      (write_bshc(tmp_path / "t.bshc", C, S), "bshc"),
                      (write_shtools(tmp_path / "t.sh", C, S), "shtools")):
        f = read_coefficients(path, format=fmt, lmax=3)
        assert f.lmax == 3 and np.allclose(f.coefficients, c[:, :4, :4], atol=1e-13)
        assert read_coefficients(path, format=fmt, lmax=50).lmax == L
    # a table from degree 2: the degrees below are zero
    path = write_bshc(tmp_path / "from2.bshc", C, S, lmin=2)
    f = read_coefficients(path, format="bshc")
    assert np.all(f.coefficients[:, :2] == 0.0)
    assert np.allclose(f.coefficients[:, 2:], c[:, 2:], atol=1e-13)


def test_a_malformed_table_is_refused(tmp_path):
    c = library_coefficients()
    C, S = in_convention(c, "4pi", 1)
    whole = write_bshc(tmp_path / "t.bshc", C, S).read_bytes()
    (tmp_path / "short.bshc").write_bytes(whole[:-80])
    with pytest.raises(ValueError, match="ends before"):
        read_coefficients(tmp_path / "short.bshc", format="bshc")
    (tmp_path / "head.bshc").write_bytes(np.array([0.5, 3.0], "<f8").tobytes())
    with pytest.raises(ValueError, match="not a bshc"):
        read_coefficients(tmp_path / "head.bshc", format="bshc")
    (tmp_path / "words.sh").write_text("no numbers here\nnor here\n")
    with pytest.raises(ValueError, match="no line"):
        read_coefficients(tmp_path / "words.sh", format="shtools")
    (tmp_path / "order.sh").write_text("0 0 1.0 0.0\n1 2 1.0 1.0\n")
    with pytest.raises(ValueError, match="order"):
        read_coefficients(tmp_path / "order.sh", format="shtools")
    with pytest.raises(ValueError, match="format"):
        read_coefficients(tmp_path / "order.sh", format="icgem")


# -- surfaces -------------------------------------------------------------------

def shape(t, p):
    return 1000.0 * (np.cos(t) ** 2 - 1.0 / 3.0) + 200.0 * np.sin(t) * np.cos(p)


def test_a_surface_is_placed_by_what_it_measures_from_its_datum():
    t, p = np.array([0.3, 1.2, 2.9]), np.array([0.1, 4.0, 2.0])
    top = Surface(lambda t, p: 6371e3 + shape(t, p), "radius", name="top")
    assert np.allclose(top.radius(t, p), 6371e3 + shape(t, p))
    up = Surface(shape, "elevation", datum=6371e3)
    down = Surface(shape, "depth", datum=6371e3)
    thick = Surface(lambda t, p: 30e3 + shape(t, p), "thickness", datum=top)
    assert np.allclose(up.radius(t, p), 6371e3 + shape(t, p))
    assert np.allclose(down.radius(t, p), 6371e3 - shape(t, p))
    assert np.allclose(thick.radius(t, p), 6371e3 - 30e3)     # below a moving top
    alone = Surface(shape, "thickness", name="ice")
    with pytest.raises(ValueError, match="with_datum"):
        alone.radius(t, p)
    assert np.allclose(alone.with_datum(top).radius(t, p),
                       6371e3 + shape(t, p) - shape(t, p))
    assert np.allclose(alone(t, p), shape(t, p))
    with pytest.raises(ValueError, match="measures"):
        Surface(shape, "height")
    with pytest.raises(ValueError, match="no datum"):
        Surface(shape, "radius", datum=1.0)
    with pytest.raises(TypeError, match="function"):
        Surface(3.0, "radius")
    for s in (top, up, down, thick, alone):
        check_surface(s)
    assert "thickness from the surface 'top'" in repr(thick)


def test_a_surface_s_mean_and_its_values_on_a_grid():
    c = library_coefficients(seed=3)
    c[0, 0, 0] = 1737e3 * np.sqrt(4 * np.pi)
    s = Surface(HarmonicFunction(c), "radius", name="moon")
    assert np.isclose(s.mean(), 1737e3)
    grid = gauss_legendre(12)
    t, p = np.meshgrid(grid.colatitudes, grid.longitudes, indexing="ij")
    assert np.isclose(grid.mean(s(t, p)), 1737e3, rtol=1e-15)
    held = s.sampled(grid)
    assert isinstance(held.function, GriddedFunction) and held.name == "moon"
    assert np.allclose(held.function.values, s(t, p), rtol=1e-14)
    assert np.isclose(held.mean(), 1737e3, rtol=1e-9)      # its interpolant's mean
    plain = Surface(lambda t, p: s(t, p), "radius")
    with pytest.raises(ValueError, match="grid="):
        plain.mean()
    assert np.isclose(plain.mean(grid=grid), 1737e3, rtol=1e-15)
    check_surface(s)
    check_surface(held)


def test_a_harmonic_function_above_scipy_s_band_goes_through_ducc0():
    pytest.importorskip("ducc0", reason="needs the planetmodel[harmonics] extra")
    lmax = SCIPY_LMAX + 10
    c = np.zeros((2, lmax + 1, lmax + 1))
    c[0, 0, 0], c[0, lmax, 3], c[1, 2, 1] = 2.0, 0.5, -0.25
    f = HarmonicFunction(c)
    t, p = np.array([0.4, 1.9]), np.array([0.2, 5.0])     # a few points
    values = f(t, p)
    assert np.all(np.isfinite(values))
    low = c[:, :3, :3]                        # the low degrees, which scipy reads
    rest = values - synthesise(low, t, p)
    assert np.all(np.abs(rest) < 0.5 * 4.0) and np.all(np.isfinite(rest))
    with pytest.raises(ValueError, match="overflow"):
        synthesise(c, t, p)


# -- the reader of an entry -----------------------------------------------------

def test_the_reader_reads_the_table_asked_for(tmp_path):
    a = library_coefficients(seed=1)
    a[0, 0, 0] = 3389.5e3 * np.sqrt(4 * np.pi)
    b = a.copy()
    b[0, 0, 0] = 3300e3 * np.sqrt(4 * np.pi)
    paths = (write_shtools(tmp_path / "Model1_cmi.sh", *in_convention(a, "4pi", 1)),
             write_shtools(tmp_path / "Model2_cmi.sh", *in_convention(b, "4pi", 1)))
    entry = Dataset("Mars-test", (Source("https://example.org/archive.zip", "0" * 64,
                                         members=("Model1_cmi.sh", "Model2_cmi.sh")),),
                    CoefficientReader(format="shtools", measures="radius"),
                    title="A test", citation=("A, 2026",), licence="CC0",
                    body="Mars")
    first = entry.reader.read(paths, entry)
    assert first.name == "Mars-test, Model1_cmi.sh"
    assert np.isclose(first.mean(), 3389.5e3)
    second = entry.reader.read(paths, entry, which="Model2_cmi.sh", lmax=2)
    assert np.isclose(second.mean(), 3300e3) and second.function.lmax == 2
    with pytest.raises(KeyError, match="no file"):
        entry.reader.read(paths, entry, which="Model5_cmi.sh")
    with pytest.raises(IndexError):
        entry.reader.read(paths, entry, which=2)
    with pytest.raises(ValueError, match="measures"):
        CoefficientReader(format="bshc", measures="height")
    check_dataset(entry)
    up = CoefficientReader(format="shtools", measures="elevation", datum=3389.5e3,
                           name="relief").read(paths[:1], entry)
    assert up.name == "relief" and up.datum == 3389.5e3
