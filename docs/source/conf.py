# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'xcal'
copyright = '2025, XCal development team'
author = 'XCal development team'

import xcal
release = xcal.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []

# Google-style docstrings only.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/cabouman/xcal',
    'use_repository_button': True,
}
html_title = 'xcal'
html_static_path = ['_static']
html_logo = '_static/logo.png'
