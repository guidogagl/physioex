"""Sphinx configuration for the PhysioEx documentation.

The docs are Markdown-first (MyST) and describe the ``physioex-dev`` API.
Autodoc imports the package at build time, so builds run in an environment
where ``physioex`` and its core deps (torch, braindecode) are installed
(the A30 venv or the CI job that does ``pip install -e ".[docs]"``).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "PhysioEx"
author = "Guido Gagliardi"
copyright = "2023, Guido Gagliardi"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",                 # MyST Markdown + notebook support (bundles myst_parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",     # Google-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",           # {tab-set}, cards, grids
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

# myst-nb auto-registers `.md` and `.ipynb`; `.rst` is built in. No override needed.
root_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/_generated",
    "pages/unavailable.md",   # empty placeholder, not part of the site
]

# -- MyST / notebooks --------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "html_image",
    "substitution",
]
# Render bare ```mermaid fences (used throughout the architecture pages)
# as the sphinxcontrib-mermaid directive, so no page bodies need editing.
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3

nb_execution_mode = "off"      # matches the previous mknotebooks `execute: false`

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
# Mock only the lazily-imported optional extras that CI does not install.
# NOT torch/braindecode (core), and NOT mne: braindecode (a core dep) pulls
# mne transitively and registers models via __init_subclass__, which breaks if
# mne is a mock ("ATCNet.__init_subclass__() takes no keyword arguments").
autodoc_mock_imports = ["transformers", "wandb", "tensorboard", "nvitop"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
}

# -- Mermaid -----------------------------------------------------------------

mermaid_version = "11.4.0"

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "PhysioEx"
html_logo = "assets/images/logo_bar.svg"
html_favicon = "assets/images/logo_bar.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/guidogagl/physioex",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Website",
            "url": "https://guidogagl.github.io/",
            "icon": "fa-solid fa-globe",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "use_edit_page_button": False,
}
html_context = {
    "default_mode": "light",
    "github_user": "guidogagl",
    "github_repo": "physioex",
    "github_version": "physioex-dev",
}
