"""Sphinx configuration for pymu-eval documentation (Read the Docs compatible)."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure pymu_eval is importable during autodoc builds
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pymu_eval  # noqa: E402

# -- Project information -----------------------------------------------------
project = "pymu-eval"
copyright = "2026, pymu-eval contributors"
author = "pymu-eval contributors"
version = pymu_eval.__version__
release = pymu_eval.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*.md"]
root_doc = "index"

# -- Autodoc configuration ---------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {}

# -- Options for HTML output -------------------------------------------------
try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
except ImportError:
    html_theme = "alabaster"

html_static_path = ["_static"]
html_title = f"pymu-eval {release} Documentation"
