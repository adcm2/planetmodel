# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# The package lives in a src layout; point autodoc at it.
sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "planetmodel"
copyright = "2026, David Al-Attar"
author = "David Al-Attar"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # Generate docs from docstrings.
    "sphinx.ext.napoleon",  # Understand numpy-style docstrings.
    "sphinx.ext.viewcode",  # Link the documentation to the source.
]

templates_path = ["_templates"]
exclude_patterns = []

# The docstrings use single backticks for inline code (`Model`, `Skeleton`).
# Left to itself reStructuredText reads those as "title reference" and
# renders them as italics; this makes them render as code, which is what
# they mean everywhere in this project.
default_role = "code"

# The dataclasses describe their fields in an `Attributes:` docstring
# section *and* declare them as annotated fields.  Napoleon's default turns
# that section into standalone `.. attribute::` directives, which autodoc
# then documents a second time from the annotations.  Rendering them as
# info-field entries instead leaves each field defined exactly once.
napoleon_use_ivar = True

# The optional extras are not installed where the docs are built; their
# importers (gmsh at the top of planetmodel.mesh3d, the rest inside the
# functions that need them) document against these stand-ins.
autodoc_mock_imports = ["gmsh", "mfem", "ducc0", "matplotlib"]


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]


# -- Generated API reference -------------------------------------------------
# sphinx-apidoc runs from here rather than from .readthedocs.yaml, so that a
# local build and a Read the Docs build produce the same pages from the same
# settings.  The .rst files it writes are build output, not source, and are
# gitignored.

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SOURCE_DIR, "..", ".."))


def _run_apidoc(_app: object) -> None:
    """Regenerate the per-module .rst stubs before the build reads them."""
    from sphinx.ext.apidoc import main

    main(
        [
            "--force",
            "--templatedir",
            os.path.join(_REPO_ROOT, "docs", "apidoc_templates"),
            "--output-dir",
            _SOURCE_DIR,
            os.path.join(_REPO_ROOT, "src", "planetmodel"),
            # Private modules stay out of the API reference.  The class
            # [!_] keeps the pattern off the __init__.py files, which
            # apidoc needs to see to know a directory is a package.
            os.path.join(_REPO_ROOT, "src", "planetmodel", "*", "_[!_]*.py"),
            os.path.join(_REPO_ROOT, "src", "planetmodel", "_[!_]*.py"),
        ]
    )


def setup(app: object) -> None:
    app.connect("builder-inited", _run_apidoc)
