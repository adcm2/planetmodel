planetmodel: spherically layered planetary models
=================================================

**planetmodel** is a Python library for spherically layered planetary
models — a planet described as concentric layers with physical
properties on each — and the meshes that hand such a model to a
numerical solver.

With this library you can build a model (PREM out of the box, simple
layered models, any mineos deck file, or a model type of your own), ask
it for its properties with the discontinuities treated honestly, deform
it with ellipticity or surface topography, and mesh it: a
spectral-element mesh along the radius, or a 2D or 3D finite-element
mesh for `MFEM <https://mfem.org>`_, refined where you need it.  A
sub-package, `planetmodel.randomfield`, draws Matérn random fields on
balls, annuli and layers.

A model is three things:

* **A skeleton**: the list of boundary radii, increasing, possibly
  starting above zero for a hollow shell.
* **A geometry**: the skeleton placed in the physical world by one
  continuous mapping from the reference (spherical) body to the
  physical one, plus names for the layers and interfaces.
* **Fields**: data on each layer, named by a vocabulary that says what
  each name means and what dimensions it carries.

Installation
============

.. code-block:: bash

   pip install planetmodel

The base install depends on numpy and scipy alone.  Meshing, MFEM
export, spherical-harmonic grid transforms, the model file and plotting
are extras:

.. code-block:: bash

   pip install "planetmodel[meshing,mfem,harmonics,netcdf,plot]"

File formats
============

The files the library writes for codes in other languages, described
for their readers:

* :doc:`formats/model_file`: a model in one netCDF file, its fields on
  angular grids at radial nodes, and its mapping.
* :doc:`formats/mesh_manifest`: the JSON beside every mesh, saying what
  the mesh's numbered attributes and the field files beside it are.

.. toctree::
   :maxdepth: 2
   :caption: File formats
   :hidden:

   formats/model_file
   formats/mesh_manifest

.. toctree::
   :maxdepth: 4
   :caption: API Reference
   :hidden:

   modules
