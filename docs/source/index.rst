.. xspec documentation master file, created by
   sphinx-quickstart on Thu Dec  1 22:51:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xspec's documentation!
=================================
This software provides a comprehensive solution for X-ray system response estimation, leveraging calibration data
with known materials and dimensions. Key features and benefits include:

- **Parametric-Based Method**: Utilizes a parametric approach to accurately estimate X-ray system responses.

- **Flexibility Across X-Ray Systems**: Designed to be adaptable, the software can be employed with a wide range of
X-ray systems, accommodating different parameters and constraints.

- **Built on PyTorch**: Incorporates PyTorch for automatic differentiation, facilitating the use of standard
optimization algorithms for more efficient and accurate estimations.

- **Ease of Use**: While focusing on technical robustness, xspec also aims to provide a user-friendly
interface and comprehensive documentation, making it accessible to a broad audience.

Target Audience
----------------

The software is intended for:

- Medical imaging and radiology professionals seeking precise calibration and response estimation of X-ray systems.
- Industrial users who rely on X-ray technology for material analysis and quality control.
- Researchers and developers in the field of computational imaging looking for a versatile and efficient tool for X-ray system analysis.


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
   tutorials
   api
   demo

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Developer Guide

   docs