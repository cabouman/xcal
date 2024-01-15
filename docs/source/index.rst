.. xspec documentation master file, created by
   sphinx-quickstart on Thu Dec  1 22:51:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xspec's documentation!
=================================
**xspec** is a Python package for automatically estimating the X-ray CT parameters that determine the X-ray energy spectrum including the source voltage, anode take-off angle, filter material and thickness, and scintillator and thickness. The package takes as input views of serval known material targets at different energies.


Features
--------
* Supports X-ray system parameters estimation using numerical optimization from normalized radiographs of serval known material targets such as Titanium (Ti), Vanadium (V), and Aluminum (Al) at different energies.
* Supports spectral energy response calculation with estimated system paramters.
* Supports the computation of ideal normalized radiographs using any customized forward projection method.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Background

   overview
   theory
   credits

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: User Guide

   install
   api
   demo

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Developer Guide

   docs