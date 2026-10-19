# Examples

`tutorials/` walks through the ideas of the library in order, one script
per idea. Each is a plain Python file with `# %%` cell markers, so it runs
as a script from any directory and cell by cell in an editor. Every
tutorial runs headless in a few seconds; figures and the files a
tutorial keeps go under `figures/`, which is gitignored.

| tutorial | what it shows |
|---|---|
| `01_skeleton_and_geometry.py` | a skeleton, changing it, a geometry with names, hollow geometries, scaling |
| `02_an_analytic_mapping.py` | a radial stretch from an analytic displacement, `F` and `J`, validity, kinks, a non-radial mapping |
| `03_a_radial_mesh.py` | the GLL reference element, a radial mesh over a skeleton, per-element nodes, the exact polynomial view, truncation by degree, a figure (uses the `plot` extra when present) |
| `04_a_mesh_for_mfem.py` | a mesh of a skeleton and its manifest, then topography through a mapping as a physical mesh, then a buffer shell with the topography tapered across it, each delivered to MFEM; the files are kept under `figures/tutorial_04_meshes/` for gmsh and glvis, or discarded with `--temp` (needs the `meshing` and `mfem` extras) |
| `05_fields.py` | fields as functions of position on one layer, radial fields and the layer functions they store, exact arithmetic, real and complex dtypes, tensors and frames, push-forward, a figure |
| `06_a_model_of_your_own.py` | a model from scratch: geometry, fields, specs, validation, conversion of units, changing the layering, a model type of your own, a figure |
| `07_prem.py` | PREM from its polynomials: layers, moduli, gravity, nodal values on a radial mesh, a figure |
| `08_simple_models.py` | homogeneous, layered and ellipsoidal models with analytic boundary shapes, a figure |
| `09_fields_for_mfem.py` | a model's fields exported beside its mesh and read back by MFEM (needs the `meshing` and `mfem` extras) |
| `10_random_fields.py` | Matern random fields of radius, on the layers of PREM as a density perturbation, and on a shell as a field with a map and a slice, a figure |
| `11_models_from_decks.py` | PREM's mineos deck as a model against the polynomial PREM, the shipped GIA deck format and a model type for it with a viscosity column, a figure |
| `12_a_laterally_varying_model.py` | PREM with a harmonic perturbation of the shear velocity and a log perturbation of a viscosity, written to a model file, read back and compared, and drawn as maps at one depth (needs the `netcdf` and `maps` extras), a figure |
| `13_pictures.py` | pictures of a perturbed PREM: slices and maps at one depth, a side taken at an interface, cross-sections as wedges and a disc, the body drawn flattened, profiles, spectra, a comparison of two models by degree and depth, and files for ParaView with first steps in it (needs the `maps` extra), figures |
| `14_published_models.py` | published models from the registry: S40RTS read as the model it is and drawn, put onto PREM and filtered, S362ANI+M given as velocities and put onto PREM through its own reference, the two compared, a regional model of the North Atlantic merged in, ANT-20 (Antarctica, on a polar stereographic map) merged in and drawn at 100 km with West Antarctica slow and East Antarctica fast, the result written to a model file, a model of the Moon (needs the network the first time, and the `netcdf` and `maps` extras), figures |
| `15_kernels.py` | a sensitivity kernel as a code writes it (the file `cpp/examples/write_kernel.cpp` writes, here written by the Python, or read from the C++ example's output given its path): maps at depths on a scale centred on zero and clipped at a percentile, a section, profiles and the one-dimensional kernel, the column integral, spectra, a kernel on an interface, and files for ParaView (needs the `netcdf` and `maps` extras), figures |

```
poetry run python examples/tutorials/01_skeleton_and_geometry.py
```

To run a tutorial cell by cell as a notebook, install the `notebook` extra
(`poetry install --extras notebook`), which provides the IPython kernel
an editor's interactive window uses.
