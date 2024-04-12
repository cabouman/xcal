========================
Getting Started/Tutorial
========================

Xspec aims to accurately determine the spectral response of X-ray systems by solving the inverse problem using calibration measurements.

.. figure:: /figs/physic_model.png
   :align: left

   **Physics model of CT scanning.** A homogeneous cylindrical object is scanned. The total spectral response can be decomposed into the product of the X-ray source spectrum, filter response, and scintillator response, respectively.

Before reading the tutorial please prepare required information listed below

- **Normalized Radiograph**: Users are required to provide a normalized radiograph for analysis.
- **System Components' Information**: Basic knowledge of the source, filter, and scintillator components used in the setup is necessary. This includes:

  - Source Types: Options include Transmission, Reflection, or Synchrotron, among others.
  - Possible Filter Materials
  - Possible Scintillator Materials
  - The specific component used in each measurement.

- **Homogeneous Samples with known composition and dimensions** (Optional): Providing homogeneous samples, with their composition and dimensions known, is essential. If the user does not know the dimension of the sample, functions are provided to calibrate the dimension information from a 3D reconstruction.

0. Jupyter Notebooks Tutorials
==============================
The remainder of this tutorial outlines the steps involved in XSPEC's spectral estimation with multiple jupyter notebook tutorials.

The
`[xspec] <https://github.com/cabouman/xspec>`__
contains a number of example Python notebooks in the
`[examples] <https://github.com/cabouman/xspec/tree/main/examples>`__
folder.

To run jupyter notebook:

1. If you install anaconda, in terminal, run

::

	jupyter notebook


2. Access Kernel Menu: In the notebook toolbar, click on the "Kernel" menu to see a list of options.

3. Change Kernel: Select "Change kernel" from the dropdown. A list of available kernels will appear, including Python versions and any other kernels you have installed (like R, Julia, etc.).

4. Select Desired Kernel (xspec): Click on the kernel you wish to switch to. The notebook will refresh, and the new kernel will be activated.



Example Dependencies
--------------------
Some examples use additional dependencies, which are listed in `requirements.txt <https://github.com/cabouman/xspec/blob/main/demo/requirements.txt>`_.

::

   pip install -r demo/requirements.txt  # Installs other example requirements


1. Total X-ray System Response
==============================

The total X-ray system response, :math:`S(E)`, describes the sensitivity across different energy bins, remaining constant when scanning various samples. This sensitivity is determined by the energy-wise product of three components: the source spectrum :math:`S^{sr}(E)`, the filter response :math:`S^{fl}(E)`, and the scintillator response :math:`S^{sc}(E)`:

.. math::

   S(E) = S^{sr}(E) \cdot S^{fl}(E) \cdot S^{sc}(E).

We discretize the continuous functions :math:`S(E)` by dividing them into :math:`N_j` non-overlapping energy bins :math:`[E_j, E_{j+1}]`. Thus, we define the discretized normalized spectral energy response value at the :math:`j^{th}` energy bin as

.. math::
  x_j = \frac{\int_{E_j}^{E_{j+1}} S(E) \, dE}{\int_{0}^{E_{max}} S(E) \, dE}

The normalized X-ray System Response :math:`\mathbf{x}` is one of essential knowledge we want to determine for further X-ray analysis.

Examples: Spectral Models
-------------------------
The following Python notebooks teach you
	- How to utilize provided spectral classes to configure X-ray system components (source, filter, scintillator)?
	- How to customize your own spectral class to adapt to different scenarios?

.. toctree::
   :maxdepth: 1

   examples/notebook/configure_spectral_models
   examples/notebook/user_spectral_models

2. Forward Modeling
===================

In order to determine the spectral response of X-ray systems, we must link The normalized X-ray System Response :math:`x`, to the measurable data, :math:`y`. This connection is established through a model of the measurement process:

.. math::

   \mathbf{y} = A \cdot \mathbf{x}.

Forward modeling is a crucial step that allows us to predict how changes in :math:`\mathbf{x}` affect our observed data, :math:`\mathbf{y}`. By comprehending this relationship, we can refine our measurements and enhance the accuracy of our spectral estimations.

How to get A?
-------------

Start from a single data point :math:`y`, which might pass through multiple homogenous samples.

Assuming that each LAC value of sample :math:`m`, :math:`\mu_m (E)`, is constant within each bin :math:`E \in [E_j, E_{j+1}]`, the total transmission of one scanning at the :math:`j^{th}` energy bin is given by

.. math::
  A_{j}=\exp \left\{-\sum_{m \in \Phi} \mu_m\left(E_j\right) L_{m}\right\}.

where:

- :math:`\Phi` represents the material set, containing multiple homogenous materials;
- :math:`L_{m}` as the path length of the projection through the :math:`m^{th}` material rod.
- :math:`\mu_m(E)` as the LAC of the material :math:`m` at energy :math:`E`.


For 3 dimensional case, calculate the forward matrix(:math:`N_{\text{views}} \times N_{\text{rows}} \times N_{\text{cols}} \times N_E`) using
the list of masks, LAC, and projector using :func:`xspec.calc_forward_matrix`:

Examples: Forward Matrix
------------------------
The following Python notebooks teach you
	- How to obtain a list of masks from reconstructions?
	- How to configure your own forward projector (parallel beam, conebeam, fanbeam, etc.) to calculate the forward matrix?

.. toctree::
   :maxdepth: 1

   examples/notebook/obj_detection
   examples/notebook/user_forward_projector

3. Estimate System Parameters by Solving Inverse Problem
========================================================

We define the MAP cost function for the multi-polychromatic dataset. This is accomplished by summing over all :math:`k`, as shown below:

.. math::
  L(\Theta) = \sum_{k=1}^{K} l\left(\theta_{a_k}^{sr}, \left\{\theta_{p}^{fl} \mid p \in B_k\right\}, \theta^{sc} \right) = \sum_{k=1}^{K}  \frac{1}{2}\left\|\boldsymbol{y}^{(k)}- \boldsymbol{A} \boldsymbol{x}^{(k)} \right\|_{\Lambda^{(k)}}^2,

where

- :math:`\Theta` denotes the aggregate set of parameters across all datasets, with :math:`K` representing the total number of single-polychromatic datasets.
- The parameter set :math:`\Theta` is composed of the source parameters :math:`\left\{\theta_{a}^{sr} \mid a = 1, \ldots, N_a\right\}`, the filter parameters :math:`\left\{\theta_{b}^{fl} \mid b = 1, \ldots, N_b\right\}`, and the scintillator parameter :math:`\theta^{sc}`.
- :math:`\Lambda^{(k)}` can be identity matrix or diagonal matrix with :math:`\Lambda_{i, i}^{(k)}=\frac{1}{y_i}`.

The optimal parameter set :math:`\Theta^*` is determined by minimizing :math:`L(\Theta)` using gradient descent:

.. math::
  \Theta^* = \arg \min_{\Theta \in \mathcal{U}} L(\Theta),

where :math:`\mathcal{U}` represents the constrained solution space.


Examples: Spectral Estimation
-----------------------------
The following Python notebooks teach you
	- How to generate a simulated dataset?
	- How to do spectral estimation with prepared spectral models and forward matrices?


.. toctree::
   :maxdepth: 1

   examples/notebook/simulated_data_3voltages
   examples/notebook/spectral_estimation_3voltages
