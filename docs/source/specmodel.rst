Total X-ray Spectral Response Configuration
===========================================

This tutorial outlines the process for configuring the total X-ray spectral response, :math:`S(E)`, in an imaging
system. The response is the product of the source spectrum, :math:`S^{sr}(E)`, the filter response, :math:`S^{fl}(E)
`, and the scintillator response, :math:`S^{sc}(E)`.

XSPEC offers object-oriented models for sources, filters, and scintillators utilizing the ``Base_Spec_Model`` class.
For instructions on crafting your own model, please see :doc:`Developing xspec Spectral Models <custmspec>`.


Prerequisites
-------------

Ensure you have the following before starting:

- Python 3.x
- Numpy
- torch
- Spekpy (Reflection source spectra generation, can be replaced by your own source spectra generator)

Source Configuration
--------------------

Begin by defining the source parameters, including the list of voltages and the anode angle.

.. code-block:: python

    import numpy as np
    import spekpy as sp

    voltage_list = [80.0, 130.0, 180.0]
    simkV_list = np.linspace(30, 200, 18, endpoint=True).astype('int')
    max_simkV = max(simkV_list)
    reference_anode_angle = 11

Generating Source Spectra
~~~~~~~~~~~~~~~~~~~~~~~~~

Generate the source spectra for each simulation voltage using Spekpy. Adjust the detected number of photons based on
the detector pixel size.

.. code-block:: python

    dsize = 0.01  # mm, detector pixel size
    energies = np.linspace(1, max_simkV, max_simkV)

    src_spec_list = []
    for simkV in simkV_list:
        s = sp.Spek(kvp=simkV + 1, th=reference_anode_angle, dk=1, mas=1, char=True)
        k, phi_k = s.get_spectrum(edges=True)
        phi_k = phi_k * ((dsize / 10) ** 2)

        src_spec = np.zeros((max_simkV))
        src_spec[:simkV] = phi_k[::2]
        src_spec_list.append(src_spec)

Setting Up X-ray Sources
~~~~~~~~~~~~~~~~~~~~~~~~

With the source spectra generated, configure the X-ray sources by defining their voltages and other parameters.
The process involves two principal steps:

1. Configure source model with Reflection_Source by pass two continuous parameters, voltages and takeoff_angle, which
should be a tuple (init value, min, max). Note that min and max set to None means that we do not need to optimize
this paramter.

2. Reflection source are an interpolation-based model, use set_src_spec_list to set up a list of simulation source
spectra with corresponding source voltages.

Notice that, reflection source always have a single takeoff angle but many different source voltage, by setting
single_takeoff_angle to True can make sure all takeoff angle would are the same for different instance of
Reflection_Source.

.. code-block:: python

    from xspec.models import Reflection_Source  # Replace with the actual import statement

    sources = [Reflection_Source(voltage=(voltage, None, None), takeoff_angle=(25, 5, 45), single_takeoff_angle=True)
               for voltage in voltage_list]

    for src_i, source in enumerate(sources):
        source.set_src_spec_list(src_spec_list, simkV_list, reference_anode_angle)

Filter and Scintillator Configuration
--------------------------------------

Specify the filters and scintillators by defining their materials, densities, and thicknesses.

.. code-block:: python

    from xspec import Material
    from xspec.models import Filter, Scintillator

    # Example configurations for filters and scintillators
    psb_fltr_mat = [Material(formula='Al', density=2.702), Material(formula='Cu', density=8.92)]
    filter_1 = Filter(psb_fltr_mat, thickness=(5, 0, 10))

    scint_params_list = [
        {'formula': 'CsI', 'density': 4.51},
        # Add additional materials as required
    ]
    psb_scint_mat = [Material(formula=scint_p['formula'], density=scint_p['density']) for scint_p in scint_params_list]
    scintillator_1 = Scintillator(materials=psb_scint_mat, thickness=(0.25, 0.01, 0.5))

Combining Components for Total Spectral Response
------------------------------------------------

The total spectral response, :math:`S(E)`, combines the source spectrum, filter response, and scintillator response.

.. math::

   S(E) = S^{sr}(E) \cdot S^{fl}(E) \cdot S^{sc}(E).

Implement this by integrating the configured sources, filters, and scintillators to a list.

.. code-block:: python

    total_spec_model_1 = [sources[0], filter_1, scintillator_1]

Conclusion
----------

This tutorial provides a comprehensive guide for configuring the total X-ray spectral response, including the setup
of X-ray sources, generation of source spectra, and configuration of filters and scintillators.
