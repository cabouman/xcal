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
    'sphinxext.opengraph',
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
    'logo': {
        'image_light': '_static/logo.png',
        'image_dark': '_static/logo_dark.png',
    },
}
html_title = 'xcal'
html_static_path = ['_static']

# Open Graph / social link preview.  Pasting a documentation URL into a
# chat, a post, or a message shows the card in _static/og_card.png, made
# by dev_scripts/make_social_card.py.  ogp_site_url makes the card and
# page URLs absolute, which link-preview crawlers require.
ogp_site_url = 'https://xcal.readthedocs.io/en/latest/'
ogp_image = 'https://xcal.readthedocs.io/en/latest/_static/og_card.png'
ogp_image_alt = 'xCal: model-based X-ray CT spectral calibration'
ogp_type = 'website'
ogp_enable_meta_description = False
ogp_description_length = 0
ogp_social_cards = {'enable': False}    # use og_card.png, not per-page cards
