Forward Matrix Calculation with Custom Forward Projector
=========================================================

This tutorial outlines the process of calculating a forward matrix (:math:`N_{\text{views}}
\times N_{\text{rows}} \times N_{\text{cols}} \times N_E`), utilizing a collection
of masks that delineate homogeneous objects within a Region of Interest (ROI). It also covers how to work with a
list of materials, each defined by known Linear Attenuation Coefficients (LAC) :math:`\mu(E)`, and demonstrates the
customization of a forward projector. This projector is tailored for performing forward projections of the given
masks, thereby determining the line path traversing a homogeneous object in in units of mm.


Prerequisites
-------------
Before you start, ensure you have the following prerequisites installed:

- Python 3.x
- Numpy
- SVMBIR (for forward projection, you can use your own forward projector)

You should also have a basic understanding of medical imaging and forward projection concepts.

Custom Forward Projector
------------------------

The custom forward projector is encapsulated within the ``fw_projector`` class. This class is designed to initialize with specific parameters that outline the imaging configuration. It features a forward function dedicated to executing the projection of a specified mask.

In the ensuing example, we encapsulate the ``svmbir.project`` function within our ``fw_projector`` class. The process involves two principal steps:

1. Initializing the class with essential geometric parameters through the ``__init__()`` method.
2. Crafting a ``forward(self, mask)`` method that computes and returns the projection (:math:`N_{\text{views}}
\times N_{\text{rows}} \times N_{\text{cols}}`) of a 3D mask.

.. code-block:: python

    import numpy as np
    import svmbir

    class fw_projector:
        """A class for forward projection using SVMBIR."""

        def __init__(self, angles, num_channels, delta_pixel=1):
            """
            Initializes the forward projector with specified geometric parameters.

            Parameters:
                angles (array): Array of projection angles.
                num_channels (int): Number of detector channels.
                delta_pixel (float, optional): Size of a pixel, defaults to 1.
            """
            self.angles = angles
            self.num_channels = num_channels
            self.delta_pixel = delta_pixel

        def forward(self, mask):
            """
            Computes the projection of a given mask.

            Parameters:
                mask (numpy.ndarray): 3D mask of the object to be projected.

            Returns:
                numpy.ndarray: The computed projection of the mask.
            """
            projections = svmbir.project(mask, self.angles, self.num_channels) * self.delta_pixel
            return projections


.. code-block:: python

	angles = np.linspace(-np.pi/2, np.pi, 40, endpoint=False)
	projector = fw_projector(angles, num_channels=1024, delta_pixel=0.01)


Obtain LAC :math:`\mu(E)`
-------------------------
Obtain LAC :math:`\mu(E)` for each scanned homogenous object with ``get_lin_att_c_vs_E``:

.. code-block:: python

    import numpy as np
    from xspec.chem_consts import get_lin_att_c_vs_E

    # Scanned cylinders
    materials = ['V', 'Al', 'Ti', 'Mg']
    mat_density = [density['%s' % formula] for formula in materials]
    energies = np.linspace(1, 160, 160) # Define energies bins from 1 kV to 160 kV with step size 1 kV.
    lac_vs_E_list = []

    for i in range(len(materials)):
        formula = materials[i]
        den = mat_density[i]
        lac_vs_E_list.append(get_lin_att_c_vs_E(den, formula, energies))


Calculating the Forward Matrix
------------------------------
Calculate the forward matrix(:math:`N_{\text{views}} \times N_{\text{rows}} \times N_{\text{cols}} \times N_E`) using
the list of masks, LAC, and projector using ``calc_forward_matrix``:

.. code-block:: python

    from xspec import calc_forward_matrix
    spec_F = calc_forward_matrix(mask_list, lac_vs_E_list, projector)


Conclusion
----------
This tutorial provided a basic overview of how to calculate a forward matrix using a custom forward projector, masks
representing homogeneous objects, and known material LAC.
