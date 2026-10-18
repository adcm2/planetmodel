The mesh manifest
=================

The JSON that travels beside every mesh the 3D mesher writes, described
from the consumer's side.  It is written by ``planetmodel.mesh3d``; the
single definition of its shape is ``planetmodel/mesh3d/manifest.py``,
and ``validate_structure`` there checks a file against it.

A mesh file carries numbered attributes and nothing else.  Which number
is which layer, which boundary is which interface, where each sits,
whether the nodes have already been displaced by the geometry's
mapping, and which files beside the mesh hold which fields: none of
that survives into the mesh file, so it travels here.  The manifest
describes what the mesh is, never how it was built.  The one deliberate
exception is ``refinements[]``, which records the extra size fields the
mesh was sized under by identity alone (the sizes, and a point cloud's
count and digest, never the points), so a consumer can tell whether a
mesh on disk was built towards the features it now wants.

**Schema string**: ``planetmodel.mesh.manifest/5``.  Consumers check
it; it is bumped only for an incompatible change.  Keys may be added
within a schema: a reader ignores a key it does not know, and a key
added within the schema is optional to it (``refinements[]`` reads back
as an empty list, ``scales.temperature`` as a kelvin, ``model`` as
null, when absent).

**Numbers.**  Every length is in the geometry's own numbers, the ones
the mesh file holds.  The mesher neither scales nor normalises, so a
radius in the manifest is a radius in the mesh.  What the numbers mean
physically is the ``scales`` block's business, once a model's fields
were exported.

**Numbering.**  Layers carry ``attribute`` 1..N from the centre
outward; the geometry's layers come first, then any shells appended
outside it.  Interfaces carry ``attribute`` 1..M in the same order: the
geometry's interfaces from the centre, then the shells' outer
boundaries.  In the mesh file, attribute ``n`` of a cell is layer
``n``, and attribute ``n`` of a boundary element is interface ``n``.
``between_layers`` gives 0-based layer indices, with -1 for the
outside; a full mesh has one interface per layer and a hollow mesh
(inner radius above zero) one more, its inner boundary being interface
1 with ``between_layers = [-1, 0]``.

The blocks
----------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - block
     - keys
   * - ``schema``
     - the schema string
   * - ``mesh``
     - ``file`` (beside the manifest), ``format`` (``"msh"`` from the
       mesher, ``"mfem"`` from the exporter), ``nodes`` (``"reference"``,
       or ``"physical"`` for an MFEM mesh whose nodes the mapping has
       moved), ``read_options`` (the ``mfem::Mesh`` constructor arguments
       the field files' dof numbering was written under; empty for a
       ``.msh``), ``displacement`` (the name of the displacement field
       among ``fields``, or null)
   * - ``layers[]``
     - ``attribute``, ``name``, ``r_inner``, ``r_outer``, ``in_geometry``
       (false for a shell added by the mesh spec), ``fluid``
   * - ``interfaces[]``
     - ``attribute``, ``name``, ``radius``, ``between_layers``, ``kind``,
       ``values``
   * - ``refinements[]``
     - one record per extra size field the mesh was sized under: ``name``
       (its class, or a function's name), ``size`` and ``far_size`` (the
       floor and ceiling of the sizes it asked for), ``decay_width``,
       ``points`` (a cloud's count) and ``digest`` (the identity of its
       coordinates), the last three null for a bare callable
   * - ``fields[]``
     - one record per file beside the mesh: ``name``, ``file``,
       ``fe_space`` (MFEM's collection name, e.g. ``H1_3D_P2`` or
       ``L2_3D_P2``), ``vdim``, ``ordering`` (``byNODES`` or ``byVDIM``),
       ``rank``, ``weight``, ``voigt``, ``unit``, ``layers``,
       ``radial_degree``
   * - ``scales``
     - ``length``, ``mass``, ``time``, ``temperature``: what one stored
       unit is in SI; null until a model's fields are exported
   * - ``constants``
     - the model's constants in its units (``G``, and ``omega`` for a
       frozen model); empty until then
   * - ``model``
     - ``file`` (a model file beside the manifest, on the mesh's
       reference skeleton, layer ``n`` of the file being attribute
       ``n + 1`` of the mesh) and ``schema`` (the model file's schema
       string); null where no model file was written
   * - ``meta``
     - whatever the mesh spec's ``meta`` held; an offset mesh records
       its ``kind``, ``inclusion_radius`` and ``offset`` here

Names default to ``layer_<attribute>`` and ``interface_<attribute>``
when the geometry gave none; they are advisory, the attributes are the
contract.

What the model says
-------------------

Four records are read from the model, so only ``export_mfem`` writes
them and everything else writes null.  Null means "no model has said",
never "no": a manifest from the mesher, or from ``export_mfem_mesh``
without a model, carries null throughout.

- ``layers[].fluid``: whether every shear-bearing field of the model's
  layer vanishes throughout it (``materials.is_fluid``).  This is a
  statement about the exported model's description, the vanishing of
  shear in the fields written beside the mesh, not about any one
  timescale; a solver decides what fluidity means for its own problem.
  False for a shell; null where the layer's fields do not say.
- ``interfaces[].kind``: one of ``solid-solid``, ``fluid-solid``,
  ``fluid-fluid`` (the fluidity of the two model layers it separates),
  ``free`` (a model layer against the outside or a shell: the model's
  surface, or a hollow model's inner boundary), ``outer`` (the outer
  boundary of a mesh whose outermost layer is a shell).  Null between
  two shells, and wherever a needed fluidity is unknown.
- ``interfaces[].values``: ``{name: [below, above]}`` for each exported
  field that is a real scalar radial one, the value just inside each
  side at the interface radius; null on a side that holds no such
  value, and null as a whole where no name has any.  Exact for a radial
  model, and what the fluid-solid terms of a weak form want on the
  solid's boundary elements.
- ``fields[].radial_degree``: the polynomial degree in r that
  reproduces a scalar field within each layer holding it, so a consumer
  may choose a lower L2 order than the mesh's (a constant is exact at
  order 0 whatever the element).  Null for a field of rank above 0 (its
  Cartesian components vary with direction), and for a field that no
  single degree reproduces on some layer.

Fields
------

A ``fields[]`` record is what a consumer builds the finite-element
space from before it opens the file, and what the values in it are:

- ``file``, ``fe_space``, ``vdim``, ``ordering``: the GridFunction and
  its space.  Every field ``export_mfem`` writes is in an L2 space of
  the mesh, one set of dofs per element, so the two sides of an
  interface carry their own layer's value and nothing is averaged.  The
  displacement is in the mesh's own nodal (H1) space.
- ``rank``, ``weight``, ``voigt``: the field's character, its law under
  a mapping.  ``vdim`` is the number of stored components: 1 for a
  scalar, 3 for a vector, 6 for a Voigt rank-2 and 36 for a Voigt
  rank-4 field.  Components are Cartesian, in the order of the
  coordinates, with Voigt order ``(11, 22, 33, 23, 13, 12)`` and no
  engineering factors.
- ``unit``: a udunits-style string under the manifest's scales
  (``kg m-3``), ``1`` for a non-dimensional model, ``unknown`` for a
  name without declared dimensions.
- ``layers``: the attributes of the layers on which the values mean
  anything.  Elsewhere (a shell outside the model, a layer without the
  field) the file holds zeros; a GridFunction has no room for "not
  defined here".

Every value is referential: the model's own field at the reference
point, whichever delivery the mesh is written in.

The displacement and the two deliveries
---------------------------------------

The mesher writes the reference mesh, concentric spheres, and the
mapping is applied at the MFEM export.  ``mesh.nodes`` says which of
the two deliveries a file is:

- **physical** (``"nodes": "physical"``): the displacement ``m(X) - X``
  was added to the node coordinates before the mesh was written, and
  the exporter refused a mapping that folds any element.  No
  displacement field is recorded.
- **referential** (``"nodes": "reference"``): the mesh is spherical and
  ``m(X) - X`` is written beside it as a vector GridFunction in the
  mesh's own nodal space, named by ``mesh.displacement`` and recorded
  among ``fields``; the consumer forms the physical mesh in one call.
  Its unit is ``1`` until a model's scales are known and the length
  unit after.

Reading an MFEM delivery
------------------------

A ``.gf`` file is indexed by the dof numbering of the mesh it was
written with, and MFEM re-marks tetrahedra for refinement on load
unless told not to, which permutes that numbering.  The mesh must
therefore be constructed with the arguments in ``mesh.read_options``,
which are ``generate_edges = 1``, ``refine = 0``, ``fix_orientation =
false``: ``Mesh(path, 1, 0, false)`` in C++, ``mfem.Mesh(path, 1, 0,
False)`` in PyMFEM.

An example
----------

.. code-block:: json

   {
     "schema": "planetmodel.mesh.manifest/5",
     "mesh": {"file": "earth.mesh", "format": "mfem", "nodes": "reference",
              "read_options": {"generate_edges": 1, "refine": 0,
                               "fix_orientation": false},
              "displacement": "displacement"},
     "layers": [
       {"attribute": 1, "name": "core", "r_inner": 0.0, "r_outer": 0.546,
        "in_geometry": true, "fluid": true},
       {"attribute": 2, "name": "mantle", "r_inner": 0.546, "r_outer": 1.0,
        "in_geometry": true, "fluid": false},
       {"attribute": 3, "name": "buffer", "r_inner": 1.0, "r_outer": 1.5,
        "in_geometry": false, "fluid": false}
     ],
     "interfaces": [
       {"attribute": 1, "name": "cmb", "radius": 0.546, "between_layers": [0, 1],
        "kind": "fluid-solid", "values": {"rho": [1.7157, 0.7655]}},
       {"attribute": 2, "name": "surface", "radius": 1.0, "between_layers": [1, 2],
        "kind": "free", "values": {"rho": [0.4642, null]}},
       {"attribute": 3, "name": "interface_3", "radius": 1.5,
        "between_layers": [2, -1], "kind": "outer", "values": null}
     ],
     "refinements": [],
     "fields": [
       {"name": "displacement", "file": "earth.displacement.gf",
        "fe_space": "H1_3D_P2", "vdim": 3, "ordering": "byNODES", "rank": 1,
        "weight": 0, "voigt": false, "unit": "1", "layers": [1, 2, 3],
        "radial_degree": null},
       {"name": "rho", "file": "earth.rho.gf", "fe_space": "L2_3D_P2",
        "vdim": 1, "ordering": "byNODES", "rank": 0, "weight": 1,
        "voigt": false, "unit": "1", "layers": [1, 2], "radial_degree": 3}
     ],
     "scales": {"length": 6371000.0, "mass": 1.4262e+24, "time": 1647.5,
                "temperature": 1.0},
     "constants": {"G": 1.0},
     "model": {"file": "earth.model.nc", "schema": "planetmodel.model/1"},
     "meta": {}
   }
