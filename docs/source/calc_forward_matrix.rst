Forward Matrix Calculation with Custom Forward Projector
=========================================================

This tutorial demonstrates how to calculate a forward matrix using a list of masks that represent homogeneous objects in a sample, along with a list of materials with known Linear Attenuation Coefficients (LAC). It includes the customization of a forward projector for forward projection of given masks.

Prerequisites
-------------
Before you start, ensure you have the following prerequisites installed:

- Python 3.x
- Numpy
- SVMBIR (for forward projection)

You should also have a basic understanding of medical imaging and forward projection concepts.

Custom Forward Projector
------------------------
The custom forward projector is defined in the ``fw_projector`` class. This class is initialized with parameters that define the imaging setup and includes a forward function for projecting a given mask.

.. code-block:: python

    class fw_projector:
        def __init__(self, angles, num_channels, delta_pixel=1, geometry='parallel'):
            self.angles = angles
            self.num_channels = num_channels
            self.delta_pixel = delta_pixel
            self.geometry = geometry

        def forward(self, mask):
            projections = svmbir.project(mask, self.angles, self.num_channels) * self.delta_pixel
            return projections

Calculating the Forward Matrix
------------------------------
To calculate the forward matrix, you need a list of masks representing homogeneous objects and a list of material properties, including the LAC for each material at different energies.

1. Define the material properties and the LAC for each material:

.. code-block:: python

    lac_vs_E_list = []
    for i in range(len(mask_list)):
        formula = materials[i]
        den = mat_density[i]
        lac_vs_E_list.append(get_lin_att_c_vs_E(den, formula, energies))

2. Initialize the custom forward projector with the desired parameters:

.. code-block:: python

    projector = fw_projector(angles, num_channels=nchanl, delta_pixel=rsize)

3. Calculate the forward matrix using the defined masks and material properties:

.. code-block:: python

    spec_F = calc_forward_matrix(mask_list, lac_vs_E_list, projector)

Conclusion
----------
This tutorial provided a basic overview of how to calculate a forward matrix using a custom forward projector, masks representing homogeneous objects, and known material properties. This process is crucial in medical imaging, especially in computed tomography (CT) for accurately modeling the interaction of X-rays with different materials.
