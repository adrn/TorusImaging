import importlib.metadata

project = "torusimaging"
copyright = "2024, Adrian Price-Whelan"
author = "Adrian Price-Whelan"
version = release = importlib.metadata.version("torusimaging")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "myst_nb",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
html_theme_options = {"header_links_before_dropdown": 1}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": (
        "https://numpy.org/doc/stable/",
        (None, "http://data.astropy.org/intersphinx/numpy.inv"),
    ),
    "matplotlib": (
        "https://matplotlib.org/stable/",
        (None, "http://data.astropy.org/intersphinx/matplotlib.inv"),
    ),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "gala": ("https://gala.adrian.pw/en/latest/", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True

# myst-nb
nb_execution_mode = "off"
