Getting Started/Tutorial
========================

Xspec aims to accurately determine the spectral response of X-ray systems by solving the inverse problem using calibration measurements.

The following part of this section describes the process for addressing the inverse problem related to x-ray spectral estimation, detailing how each step maps to a component of XSPEC.

Forward Modeling
----------------

Beer's Law
~~~~~~~~~~

Beer's Law, also known as Beer-Lambert Law, is a key principle in spectroscopy and radiography, describing the attenuation of light (or X-ray) through a medium. The law is mathematically represented as:

.. math::

    I = I_0 e^{-\mu x}

where:

- :math:`I` is the intensity of the X-ray after passing through the material,
- :math:`I_0` is the initial intensity of the X-ray before it enters the material,
- :math:`\mu` is the linear attenuation coefficient of the material, which depends on the energy of the X-ray and the material's properties,
- :math:`x` is the thickness of the material.