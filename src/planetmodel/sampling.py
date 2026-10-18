"""A model evaluated on radial times angular nodes.

A consumer's unknowns live on a product of a radial node set and an
angular node set, and a model is evaluated there once, in one
vectorised call per layer and field; the result is a `Sample`.

**Layout.**  Every sampled array is `(node, colatitude, longitude,
components...)`, node outermost and longitude fastest.  The radial
nodes are the per-element GLL nodes of a `RadialMesh`, flattened
element by element, so the flat `radius` array repeats every element
boundary: an interface is two nodes at one radius, one on each side,
and each carries its own layer's one-sided value.  That is how a
discontinuous field survives sampling without a second array saying
where the jumps are.  A field that does not depend on direction, one
whose layers all declare `is_radial`, is sampled on `(node,)` alone
and the shape is the mark; a consumer broadcasts.  A radial field of
any rank is direction-free in this sense, since its components in the
local spherical frame are functions of the radius alone.

**Components** are in the spherical frame (e_r, e_theta, e_phi) at the
node, with ranks 2 and 4 in Voigt form where the character has one.
The displacement of the geometry's mapping, `m(X) - X`, is sampled the
same way: a vector on the reference body in the spherical frame at X,
which for a radial stretch is `(h, 0, 0)`.  It is None when the
geometry is the identity.

**Missing fields.**  A name a layer does not hold is refused, naming
the layer and the name; `missing="nan"` fills NaN on that layer's
nodes instead, and `fields=None` samples the names every layer holds.

**Angular grids** are node arrays first.  `kind` and `lmax` say what
family the nodes came from and what band the grid resolves; the arrays
are the truth, and a consumer with its own grid hands its nodes over as
a `custom` grid.  Colatitudes lie in [0, pi], so either pole may be a
node, and longitudes in [0, 2 pi).  Three families are shipped, each
with its colatitude weights:

    gauss_legendre(lmax)       lmax + 1 Gauss-Legendre colatitudes, no pole
    driscoll_healy(lmax)       2 lmax + 2 equally spaced from the north
                               pole, which is a node, to one step short of
                               the south pole (Driscoll and Healy's grid)
    equiangular(ntheta, nphi)  equally spaced midpoints, no pole (Fejer's
                               first rule), or both poles with poles=True
                               (Clenshaw-Curtis)

The weight convention is

    integral over S^2 of f dOmega  ~=  sum_i w_i sum_j (2 pi / nphi) f_ij,

so `weights` are colatitude weights against sin(theta) dtheta, the
Legendre weights on x = cos(theta) for a Gauss grid, and the longitude
rule is the consumer's own trapezoid, exact for `exp(i (m - m') phi)`
only while `|m - m'| < nphi`.  The band is the highest degree the grid
determines, so that a field of that band is recovered exactly by
analysis on it: `lmax` for the Gauss-Legendre and Driscoll-Healy grids,
ntheta - 1 for the equiangular midpoints and ntheta - 2 with the poles,
and never more than (nphi - 1) // 2, since resolving orders `|m| <=
lmax` needs `nphi >= 2 lmax + 1`.  Every shipped weight rule integrates
the polynomials in cos(theta) of degree below ntheta exactly, so the
weights integrate every field of the grid's band exactly.  On the
Gauss-Legendre and Driscoll-Healy grids they also integrate the product
of two fields of the band; the equiangular grids reach their band only
through the upsampling analysis of `planetmodel.harmonics`.

At a pole node the frame (e_theta, e_phi) depends on the longitude, so
a component of rank 1 or above is sampled there as the limit along its
own longitude's meridian: the columns differ while the vector does not.
"""
from __future__ import annotations

from collections import abc
from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from .character import Character
from .fields import Field
from .fields import stored_shape as _stored_shape
from .frames import spherical_frame
from .mapping import Mapping
from .mesh1d.mesh import RadialMesh
from .model import Model
from .units import Dimensions, Scales

__all__ = ["AngularGrid", "gauss_legendre", "driscoll_healy", "equiangular",
           "Sample", "sample", "KINDS", "MISSING"]

#: The families an angular grid can come from.
KINDS = ("gauss_legendre", "driscoll_healy", "equiangular", "custom")

#: What `sample` does on a layer lacking a field.
MISSING = ("refuse", "nan")


def _readonly(a: ArrayLike, *, name: str) -> np.ndarray:
    """A read-only float64 1-d copy of a node array, or a ValueError."""
    a = np.array(a, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-d array, got shape "
                         f"{a.shape}")
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class AngularGrid:
    """A product angular node set: colatitudes times longitudes.

    `colatitudes` are strictly increasing in [0, pi], so that either
    pole may be a node, and `longitudes` strictly increasing in
    [0, 2 pi), both in radians and stored read-only.  `kind` names the
    family the nodes came from and `lmax` the band the grid resolves,
    when it is that kind of grid; `weights`, when present, are
    colatitude weights in the convention of the module docstring.
    `gauss_legendre`, `driscoll_healy` and `equiangular` build the
    shipped families; anything else is a `custom` grid, and nothing
    downstream treats one differently.
    """

    colatitudes: np.ndarray
    longitudes: np.ndarray
    _: KW_ONLY
    kind: str = "custom"
    lmax: int | None = None
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        theta = _readonly(self.colatitudes, name="colatitudes")
        phi = _readonly(self.longitudes, name="longitudes")
        if not np.all(np.diff(theta) > 0.0):
            raise ValueError("colatitudes must be strictly increasing")
        if theta[0] < 0.0 or theta[-1] > np.pi:
            raise ValueError("colatitudes must lie in [0, pi]")
        if not np.all(np.diff(phi) > 0.0):
            raise ValueError("longitudes must be strictly increasing")
        if phi[0] < 0.0 or phi[-1] >= 2.0 * np.pi:
            raise ValueError("longitudes must lie in [0, 2 pi)")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        lmax = self.lmax
        if lmax is not None:
            lmax = int(lmax)
            if lmax < 0:
                raise ValueError(f"lmax must be non-negative, got {lmax}")
        w = self.weights
        if w is not None:
            w = _readonly(w, name="weights")
            if w.shape != theta.shape:
                raise ValueError(f"weights must have one entry per colatitude, "
                                 f"{theta.size}, got {w.size}")
        object.__setattr__(self, "colatitudes", theta)
        object.__setattr__(self, "longitudes", phi)
        object.__setattr__(self, "lmax", lmax)
        object.__setattr__(self, "weights", w)

    @property
    def ntheta(self) -> int:
        """The number of colatitudes."""
        return self.colatitudes.size

    @property
    def nphi(self) -> int:
        """The number of longitudes."""
        return self.longitudes.size

    @property
    def pole_nodes(self) -> tuple[bool, bool]:
        """Whether the north pole and the south pole are nodes, in that order."""
        return bool(self.colatitudes[0] == 0.0), bool(self.colatitudes[-1] == np.pi)

    def __repr__(self) -> str:
        band = "" if self.lmax is None else f", lmax={self.lmax}"
        return f"AngularGrid({self.kind}, {self.ntheta} x {self.nphi}{band})"


def gauss_legendre(lmax: int, *, nphi: int | None = None) -> AngularGrid:
    """The Gauss-Legendre grid resolving degrees and orders up to `lmax`.

    `lmax + 1` Gauss-Legendre colatitudes in x = cos(theta), sorted so
    that theta increases, with the Legendre weights alongside, and
    `nphi` equally spaced longitudes from zero, 2 lmax + 1 by default,
    the fewest that resolve the orders.  An explicit `nphi` below that
    bound is refused: at 2 lmax the orders +lmax and -lmax are one
    discrete mode and cannot be told apart.
    """
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    x, w = np.polynomial.legendre.leggauss(lmax + 1)
    theta = np.arccos(x)
    order = np.argsort(theta)
    need = 2 * lmax + 1
    if nphi is None:
        nphi = need
    elif int(nphi) < need:
        raise ValueError(
            f"nphi={nphi} cannot resolve orders |m| <= {lmax}: the trapezoid "
            f"rule separates them only for nphi >= 2 lmax + 1 = {need}")
    nphi = int(nphi)
    phi = 2.0 * np.pi * np.arange(nphi) / nphi
    return AngularGrid(theta[order], phi, kind="gauss_legendre", lmax=lmax,
                       weights=w[order])


def driscoll_healy(lmax: int, *, sampling: int = 1) -> AngularGrid:
    """The Driscoll-Healy grid resolving degrees and orders up to `lmax`.

    n = 2 lmax + 2 equally spaced colatitudes theta_i = pi i / n, from
    the north pole, which is a node, to one step short of the south
    pole, which is not; n equally spaced longitudes from zero for
    `sampling=1`, or 2 n for `sampling=2`, the grid equally spaced in
    both angles.  The weights are Driscoll and Healy's, exact for the
    polynomials in cos(theta) of degree below n; the north pole's is
    zero.
    """
    lmax, sampling = int(lmax), int(sampling)
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    if sampling not in (1, 2):
        raise ValueError(f"sampling must be 1 or 2, got {sampling}")
    n = 2 * lmax + 2
    theta = np.pi * np.arange(n) / n
    nphi = sampling * n
    phi = 2.0 * np.pi * np.arange(nphi) / nphi
    return AngularGrid(theta, phi, kind="driscoll_healy", lmax=lmax,
                       weights=_driscoll_healy_weights(n))


def equiangular(ntheta: int, nphi: int, *, poles: bool = False) -> AngularGrid:
    """Equally spaced colatitudes and longitudes, with weights and a band.

    By default the colatitudes are the midpoints theta_i = pi (i + 1/2)
    / ntheta, both poles left out, with the weights of Fejer's first
    rule; `poles=True` makes both poles nodes, theta_i = pi i / (ntheta
    - 1) for ntheta of at least two, with the Clenshaw-Curtis weights.
    Either rule is exact for the polynomials in cos(theta) of degree
    below ntheta.  The longitudes are phi_j = 2 pi j / nphi.  The band
    is ntheta - 1 for the midpoints and ntheta - 2 with the poles, and
    at most (nphi - 1) // 2.
    """
    ntheta, nphi = int(ntheta), int(nphi)
    if ntheta < 1 or nphi < 1:
        raise ValueError("ntheta and nphi must be positive")
    if poles:
        if ntheta < 2:
            raise ValueError("an equiangular grid with both poles needs ntheta "
                             f">= 2, got {ntheta}")
        theta = np.linspace(0.0, np.pi, ntheta)          # the poles exactly
        weights, band = _clenshaw_curtis_weights(ntheta), ntheta - 2
    else:
        theta = np.pi * (np.arange(ntheta) + 0.5) / ntheta
        weights, band = _fejer_weights(ntheta), ntheta - 1
    phi = 2.0 * np.pi * np.arange(nphi) / nphi
    return AngularGrid(theta, phi, kind="equiangular",
                       lmax=min(band, (nphi - 1) // 2), weights=weights)


# -- colatitude weights against sin(theta) dtheta, summing to 2 ------------
# Each is a cosine sum accumulated term by term, O(n^2) work in O(n) memory.

def _fejer_weights(n: int) -> np.ndarray:
    """Fejer's first rule on the n midpoints theta_k = pi (k + 1/2) / n:
    w_k = (2 / n) (1 - 2 sum_{j=1}^{n // 2} cos(2 j theta_k) / (4 j^2 - 1))."""
    theta = np.pi * (np.arange(n) + 0.5) / n
    s = np.zeros(n)
    for j in range(1, n // 2 + 1):
        s += np.cos(2 * j * theta) / (4 * j * j - 1)
    return (2.0 / n) * (1.0 - 2.0 * s)


def _clenshaw_curtis_weights(n: int) -> np.ndarray:
    """Clenshaw-Curtis on the n nodes theta_k = pi k / N, N = n - 1:
    w_k = (c_k / N) (1 - sum_{j=1}^{N // 2} b_j cos(2 j theta_k) /
    (4 j^2 - 1)), with c_k = 1 at the poles and 2 between, b_j = 1 for
    j = N / 2 and 2 otherwise."""
    N = n - 1
    theta = np.pi * np.arange(n) / N
    s = np.zeros(n)
    for j in range(1, N // 2 + 1):
        b = 1.0 if 2 * j == N else 2.0
        s += b * np.cos(2 * j * theta) / (4 * j * j - 1)
    c = np.full(n, 2.0)
    c[0] = c[-1] = 1.0
    return (c / N) * (1.0 - s)


def _driscoll_healy_weights(n: int) -> np.ndarray:
    """Driscoll and Healy's weights on the n nodes theta_k = pi k / n, n
    even: w_k = (4 / n) sin(theta_k) sum_{j=0}^{n/2 - 1} sin((2 j + 1)
    theta_k) / (2 j + 1)."""
    theta = np.pi * np.arange(n) / n
    s = np.zeros(n)
    for j in range(n // 2):
        s += np.sin((2 * j + 1) * theta) / (2 * j + 1)
    return (4.0 / n) * np.sin(theta) * s


@dataclass(frozen=True)
class Sample:
    """A model's fields and mapping evaluated on radial times angular nodes.

    `fields[name]` has shape `(nnode, ntheta, nphi) + c` for a field
    that depends on direction and `(nnode,) + c` for one that does not,
    with `c` the stored shape of its character: Voigt where the
    character has one, the component shape otherwise; the components
    are in the spherical frame at the node.  A field a layer lacks is
    NaN on the nodes of that layer's elements.  `displacement` is
    `m(X) - X` in the spherical frame at X, shape `(nnode, ntheta, nphi,
    3)`, or None for an identity geometry.  `characters` and
    `dimensions` say what each array is, `scales` what its numbers are
    in, and `layer_names` names the layers the nodes refer to.  Every
    array and mapping is read-only.
    """

    radial: RadialMesh
    angular: AngularGrid
    fields: abc.Mapping[str, np.ndarray]
    displacement: np.ndarray | None
    characters: abc.Mapping[str, Character]
    dimensions: abc.Mapping[str, Dimensions | None]
    scales: Scales
    layer_names: tuple[str | None, ...]

    def __post_init__(self) -> None:
        for attr in ("fields", "characters", "dimensions"):
            object.__setattr__(self, attr, MappingProxyType(dict(getattr(self, attr))))
        object.__setattr__(self, "layer_names", tuple(self.layer_names))

    @property
    def nnode(self) -> int:
        """The flat node count, nspec * ngll: element boundaries counted twice."""
        return self.radial.nspec * self.radial.ngll

    @property
    def radius(self) -> np.ndarray:
        """The radius of every flat node: the per-element GLL nodes flattened."""
        r = self.radial.r.ravel()
        r.setflags(write=False)
        return r

    @property
    def element_layer(self) -> np.ndarray:
        """The skeleton layer of every element, shape (nspec,)."""
        return self.radial.layer

    @property
    def node_layer(self) -> np.ndarray:
        """The skeleton layer of every flat node, shape (nnode,)."""
        return np.repeat(self.radial.layer, self.radial.ngll)

    def stored_shape(self, name: str) -> tuple[int, ...]:
        """The trailing axes a field's array carries beyond the nodes."""
        return _stored_shape(self.characters[name])

    def is_radial(self, name: str) -> bool:
        """Whether `name` was sampled on `(node,)` alone."""
        return self.fields[name].shape == (self.nnode,) + self.stored_shape(name)

    def __repr__(self) -> str:
        names = ", ".join(self.fields)
        disp = "" if self.displacement is None else ", displacement"
        return (f"Sample({self.nnode} nodes x {self.angular.ntheta} x "
                f"{self.angular.nphi}; fields: {names}{disp})")


# -- the sampler ----------------------------------------------------------

def _radial_mesh(model: Model, grid: AngularGrid, radial: RadialMesh | None,
                 ngll: int, drmax: float | None) -> RadialMesh:
    """The caller's mesh, or one sized by `drmax`, or by the grid's band."""
    if radial is not None:
        if not isinstance(radial, RadialMesh):
            raise TypeError(f"radial must be a RadialMesh, got "
                            f"{type(radial).__name__}")
        if radial.skeleton != model.skeleton:
            raise ValueError("radial is a mesh over another skeleton; a "
                             "sample's nodes lie on the model it samples")
        if drmax is not None:
            raise ValueError("give radial= or drmax=, not both: a mesh already "
                             "fixes its element widths")
        return radial
    if drmax is not None:
        return RadialMesh(model.geometry, ngll=ngll, drmax=drmax)
    if grid.lmax is None:
        raise ValueError(
            f"a {grid.kind!r} angular grid carries no band, so nothing sizes "
            "the radial mesh: pass radial= (a RadialMesh over this model's "
            "skeleton) or drmax=")
    return RadialMesh(model.geometry, ngll=ngll, lmax=grid.lmax)


def _names(model: Model, fields: Iterable[str] | None) -> tuple[str, ...]:
    """The names to sample: every common name, or the sequence given."""
    if fields is None:
        return model.common_names()
    if isinstance(fields, str):
        raise TypeError("fields is a sequence of names, not a single string")
    names = tuple(str(n) for n in fields)
    known = model.field_names()
    for name in names:
        if name not in known:
            raise KeyError(f"no layer holds a field {name!r}; the model holds "
                           f"{list(known)}")
    return names


def _layer_fields(model: Model, name: str, layers: Iterable[int],
                  missing: str) -> dict[int, Field | None]:
    """The field of `name` on each of `layers`, or None where allowed missing."""
    out = {}
    for L in layers:
        layer = model.layer(int(L))
        if name in layer:
            out[int(L)] = layer[name]
        elif missing == "nan":
            out[int(L)] = None
        else:
            layer[name]                       # raises, naming layer and name
    chars = {f.character for f in out.values() if f is not None}
    if len(chars) > 1:
        raise ValueError(f"{name!r} has different characters on different "
                         f"layers: {sorted(map(str, chars))}")
    return out


def _expect(values: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    """A float64 array of exactly the promised shape."""
    if np.iscomplexobj(values):
        raise TypeError(f"{name!r} evaluated to complex values; a sample is real")
    values = np.asarray(values, dtype=float)
    if values.shape != shape:
        raise ValueError(f"{name!r} evaluated to shape {values.shape}, but its "
                         f"character promises {shape} on these nodes")
    return values


def _sample_field(name: str, per_layer: abc.Mapping[int, Field | None],
                  r: np.ndarray, node_layer: np.ndarray, theta: np.ndarray,
                  phi: np.ndarray) -> tuple[np.ndarray, Character]:
    """One field on the nodes, layer by layer, both sides of every jump.

    Direction-free when every layer's field is radial: then the stored
    components are functions of the radius alone and one column of
    values serves every direction.
    """
    present = [f for f in per_layer.values() if f is not None]
    char = present[0].character
    c = _stored_shape(char)
    radial = all(getattr(f, "is_radial", False) for f in present)
    nnode, ntheta, nphi = r.size, theta.size, phi.size
    if radial:
        out = np.full((nnode,) + c, np.nan)
    else:
        out = np.full((nnode, ntheta, nphi) + c, np.nan)
    for L, f in per_layer.items():
        if f is None:
            continue
        m = node_layer == L
        n = int(m.sum())
        if radial:
            v = f.evaluate(r[m], theta[0], phi[0])
            out[m] = _expect(v, (n,) + c, name)
        else:
            v = f.evaluate(r[m][:, None, None], theta[None, :, None],
                           phi[None, None, :])
            out[m] = _expect(v, (n, ntheta, nphi) + c, name)
    out.setflags(write=False)
    return out, char


def _sample_displacement(mapping: Mapping, r: np.ndarray, theta: np.ndarray,
                         phi: np.ndarray) -> np.ndarray:
    """R^T (m(X) - X) at every node: the shift's spherical components at X."""
    R = spherical_frame(theta[:, None], phi[None, :])    # (nt, np, 3, 3)
    X = r[:, None, None, None] * R[None, ..., :, 0]      # X = r e_r
    disp = getattr(mapping, "displacement", None)
    u = disp(X) if disp is not None else np.asarray(mapping(X), dtype=float) - X
    u = np.asarray(u, dtype=float)
    if u.shape != X.shape:
        raise ValueError(f"the mapping's displacement has shape {u.shape} for "
                         f"points of shape {X.shape}")
    # R[..., j, i] is Cartesian component j of frame vector i, so the
    # spherical components are u . e_i = sum_j R[j, i] u_j
    out = np.ascontiguousarray(np.einsum("tpji,ntpj->ntpi", R, u))
    out.setflags(write=False)
    return out


def sample(model: Model, grid: AngularGrid, *, fields: Iterable[str] | None = None,
           radial: RadialMesh | None = None, ngll: int = 5,
           drmax: float | None = None, missing: str = "refuse") -> Sample:
    """Evaluate a model's fields and its mapping's displacement on a grid.

    `grid` is the angular node set.  The radial one is `radial` when the
    caller has a `RadialMesh` over the model's skeleton, else
    `RadialMesh(model.geometry, ngll=ngll, drmax=drmax)` when `drmax` is
    given, else `RadialMesh(model.geometry, ngll=ngll, lmax=grid.lmax)`
    when the grid carries a band; a custom grid without a band needs one
    of the two said explicitly.

    `fields` is None for the names every layer holds, or a sequence of
    names.  Each is evaluated once per layer on the nodes of that
    layer's elements, so the two one-sided values at an interface come
    from the two layers that own them.  A layer lacking a name is
    refused with KeyError naming the layer and the name, unless
    `missing="nan"` asks for NaN on that layer's nodes.  The
    displacement is sampled unless the geometry is the identity.
    """
    if not isinstance(model, Model):
        raise TypeError(f"expected a Model, got {type(model).__name__}")
    if not isinstance(grid, AngularGrid):
        raise TypeError(f"grid must be an AngularGrid, got {type(grid).__name__}")
    if missing not in MISSING:
        raise ValueError(f"missing must be one of {MISSING}, got {missing!r}")
    mesh = _radial_mesh(model, grid, radial, ngll, drmax)
    names = _names(model, fields)
    r = mesh.r.ravel()
    node_layer = np.repeat(mesh.layer, mesh.ngll)
    layers = np.unique(mesh.layer)
    theta, phi = grid.colatitudes, grid.longitudes

    arrays, chars, dims = {}, {}, {}
    for name in names:
        per_layer = _layer_fields(model, name, layers, missing)
        if all(f is None for f in per_layer.values()):
            raise KeyError(f"no layer the mesh covers holds a field {name!r}")
        arrays[name], chars[name] = _sample_field(name, per_layer, r, node_layer,
                                                  theta, phi)
        spec = model.spec(name)
        dims[name] = None if spec is None else spec.dimensions
    geometry = model.geometry
    disp = (None if geometry.is_identity
            else _sample_displacement(geometry.mapping, r, theta, phi))
    return Sample(radial=mesh, angular=grid, fields=arrays, displacement=disp,
                  characters=chars, dimensions=dims, scales=model.scales,
                  layer_names=tuple(layer.name for layer in model.layers))
