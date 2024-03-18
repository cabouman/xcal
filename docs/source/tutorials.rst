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

The remainder of this section outlines the steps involved in spectral estimation, and shows how each concept maps to a component of XSPEC.

Total X-ray Spectral Response
=============================

The total X-ray spectral response, :math:`S(E)`, describes the sensitivity across different energy bins, remaining constant when scanning various samples. This sensitivity is determined by the energy-wise product of three components: the source spectrum :math:`S^{sr}(E)`, the filter response :math:`S^{fl}(E)`, and the scintillator response :math:`S^{sc}(E)`:

.. math::

   S(E) = S^{sr}(E) \cdot S^{fl}(E) \cdot S^{sc}(E).

We discretize the continuous functions :math:`S(E)` by dividing them into :math:`N_j` non-overlapping energy bins :math:`[E_j, E_{j+1}]`. Thus, we define the discretized normalized spectral energy response value at the :math:`j^{th}` energy bin as

.. math::
  x_j = \frac{\int_{E_j}^{E_{j+1}} S(E) \, dE}{\int_{0}^{E_{max}} S(E) \, dE}

The normalized X-ray spectral response :math:`\mathbf{x}` is one of essential knowledge we want to determine for further X-ray analysis.

1. Configure Source Model
-------------------------
XSPEC provides the :func:`xspec.models.Reflection_Source` and :func:`xspec.models.Transmission_Source` classes in order to provide a interpolation function over multiple source spectra simulations. User can write their own source classes by subclassing :func:`xspec.models.Base_Spec_Model`. Details can be found in `Developing your own xspec Spectral Models <custmspec.html>`_.

Here is an example to configure the provided reflection source class. With the source spectra provided, configure the X-ray source by defining two continuous variable, voltages and takeoff angle. The process involves two principal steps:

1. Configure the source model with `Reflection_Source` by passing two continuous parameters, voltages, and takeoff angle, which should be a tuple (init value, min, max). Note that setting min and max to `None` indicates that we do not need to optimize this parameter.

2. Reflection sources are an interpolation-based model. Use `set_src_spec_list` to set up a list of simulation source spectra with corresponding source voltages.

.. note::

   In practice, the reflection source always has a fixed takeoff angle but the source voltage is adjustable. By setting `single_takeoff_angle` to True, you can ensure all takeoff angles are the same for different instances of `Reflection_Source`.


.. code-block:: python

	import numpy as np
	from xspec.models import Reflection_Source
	import matplotlib.pyplot as plt
	import spekpy as sp
	import torch

	# Use Spekpy to generate source spectra with source voltages from 30 to 200 kV and takeoff angle = 11.
	min_simkV = 30
	max_simkV = 200
	dsize = 0.01 # mm
	simkV_list = np.linspace(min_simkV, max_simkV, 18, endpoint=True).astype('int')
	reference_anode_angle = 11
	energies = np.linspace(1, max_simkV, max_simkV)

	src_spec_list = []
	for simkV in simkV_list:
		s = sp.Spek(kvp=simkV + 1, th=reference_anode_angle, dk=1, mas=1, char=True)
		k, phi_k = s.get_spectrum(edges=True)  # Get arrays of energy & fluence spectrum
		phi_k = phi_k * ((dsize / 10) ** 2)

		src_spec = np.zeros((max_simkV))
		src_spec[:simkV] = phi_k[::2]
		src_spec_list.append(src_spec)

	# Initial reflection source model.
	# source voltage is initialized as 80 kV with a range [30, 200] kV.
	# takeoff angle is initialized as 25 degree with a range [5, 45].
	source = Reflection_Source(voltage=(80, 30, 200), takeoff_angle=(25, 5, 45), single_takeoff_angle=True)
	# set source spectral list for interpolation over source voltage.
	source.set_src_spec_list(src_spec_list, simkV_list, reference_anode_angle)

	# Plot the source spectrum with given initial value.
	with torch.no_grad():
		plt.plot(energies, source(energies))

.. note::

   When configuring the reflection source model, ensure to provide the following inputs:

   - **`simkV_list`**: Array of source voltages, defined using `np.linspace(30, 200, 18, endpoint=True)`, to specify the voltage range and intervals.
   - **`reference_anode_angle`**: The takeoff angle for all spectra, set as a fixed value (e.g., 11 degrees).
   - **`src_spec_list`**: A list that will contain the spectral data for each source voltage. This list is populated through a loop that generates spectra for each voltage in `simkV_list` using Spekpy.
   - **Reflection_Source Configuration**:
     - **`voltage`**: A tuple indicating the initial source voltage (e.g., 80 kV) and its allowable range ([30, 200] kV).
     - **`takeoff_angle`**: A tuple for the initial takeoff angle (e.g., 25 degrees) and its range ([5, 45] degrees).
     - **`single_takeoff_angle`**: A boolean value (`True`) to maintain the same takeoff angle across different instances of `Reflection_Source`.



2. Configure Filter Model
-------------------------
The filter response is fundamentally influenced by the filter material composition and thickness. X-ray filters, made of materials like aluminum (Al) or copper (Cu), absorb low-energy photons from the X-ray beam. The filter response is represented as:

.. math::

   S^{fl}(E) = \prod_{p=1}^{N^{fl}} s^{fl}\left(E; M_p^{fl}, T_p^{fl}\right) = \mathrm{e}^{-\sum_p \mu(E, M_p^{fl}) T_p^{fl}},

where :math:`\mu(E, M_p^{fl})` is the Linear Attenuation Coefficient (LAC) of the :math:`p^{th}` filter made of material :math:`M_p^{fl}` at energy :math:`E`, and :math:`T_p^{fl}` denotes its thickness.


We provide the :func:`xspec.models.Filter` class in order to provide a analytical filter model for gradient descent. Here is an example to configure a single filter :math:`s^{fl}\left(E; M_p^{fl}, T_p^{fl}\right)`.

.. code-block:: python

    from xspec import Material
    from xspec.models import Filter

    # Example configurations for a filter
    # Material takes chemical composition formula and density g/cm^3
    psb_fltr_mat = [Material(formula='Al', density=2.702), Material(formula='Cu', density=8.92)]
    filter = Filter(psb_fltr_mat, thickness=(5, 0, 10))

    # Plot the filter response with the first possible material and initial thickness.
    with torch.no_grad():
        plt.plot(energies, filter(energies))

.. note::
   When configuring the filter model, ensure to provide the following inputs:

   - **Possible Materials**
   - **Thickness Range**

3. Configure Scintillator Model
-------------------------------
A scintillator converts absorbed X-ray photon energies into visible light photons. The response of various scintillators, often modeled using MCNP simulations, can be represented as:

.. math::

   S^{sc}\left(E ; M^{sc}, T^{sc}\right) = \frac{\mu^{en}(E;  M^{sc})}{\mu(E;  M^{sc})}\left(1 - e^{-\mu(E;  M^{sc}) T^{sc}}\right) E,

where :math:`\mu^{en}(E;  M^{sc})` is the linear energy-absorption coefficient of the scintillator made of :math:`M^{sc}` and :math:`\mu` represents the LAC of the scintillator made of :math:`M^{sc}`.

We provide the :func:`xspec.models.Scintillator` class in order to provide a analytical scintillator model for gradient descent. Here is an example to configure a scintillator :math:`S^{sc}\left(E ; M^{sc}, T^{sc}\right)`.

.. code-block:: python

	from xspec import Material
	from xspec.models import Scintillator

	# Example configurations for scintillators
	# Material takes chemical composition formula and density g/cm^3
	scint_params_list = [
		{'formula': 'CsI', 'density': 4.51},
        {'formula': 'Lu3Al5O12', 'density': 6.73},
        {'formula': 'CdWO4', 'density': 7.9},
		# Add additional materials as required
	]
	psb_scint_mat = [Material(formula=scint_p['formula'], density=scint_p['density']) for scint_p in scint_params_list]
	scintillator = Scintillator(materials=psb_scint_mat, thickness=(0.25, 0.01, 0.5))

	# Plot the scintillator response with the first possible material and initial thickness.
	with torch.no_grad():
		plt.plot(energies, scintillator(energies))


.. note::
   When configuring a scintillator model, ensure to provide the following inputs:

   - **Possible Materials**
   - **Thickness Range**


Forward Modeling
================

In order to determine the spectral response of X-ray systems, we must link The normalized X-ray spectral response :math:`x`, to the measurable data, :math:`y`. This connection is established through a model of the measurement process:

.. math::

   \mathbf{y} = A \cdot \mathbf{x}.

Forward modeling is a crucial step that allows us to predict how changes in :math:`\mathbf{x}` affect our observed data, :math:`\mathbf{y}`. By comprehending this relationship, we can refine our measurements and enhance the accuracy of our spectral estimations.

How to get A?
-------------

Start from a single data point :math:`y`, which might pass through multiple homogenous samples.

Assuming that each LAC value of sample :math:`m`, :math:`\mu_m (E)`, is constant within each bin :math:`E \in [E_j, E_{j+1}]`, the total attenuation of one scanning at the :math:`j^{th}` energy bin is given by

.. math::
  A_{j}=\exp \left\{-\sum_{m \in \Phi} \mu_m\left(E_j\right) L_{m}\right\}.

where:

- :math:`\Phi` represents the material set, containing multiple homogenous materials;
- :math:`L_{m}` as the path length of the projection through the :math:`m^{th}` material rod.
- :math:`\mu_m(E)` as the LAC of the material :math:`m` at energy :math:`E`.


Extend to 3 dimensional case, calculate the forward matrix(:math:`N_{\text{views}} \times N_{\text{rows}} \times N_{\text{cols}} \times N_E`) using
the list of masks, LAC, and projector using :func:`xspec.calc_forward_matrix`:

.. code-block:: python

    from xspec import calc_forward_matrix
    spec_F = calc_forward_matrix(mask_list, lac_vs_E_list, projector)

.. note::
   Ensure to prepare for the following inputs:

   - **Composition of the sample**: Material set :math:`\Phi`
   - **Dimension of the samples**: List of masks corresponding to samples to calculate :math:`L_{m}`.
   - **CT Forward Projector**: Parallel beam or cone beam forward projector to calculate length path with each sample mask. User can develop own forward projector wrapper as  `Forward Matrix Calculation with Custom Forward Projector <calc_forward_matrix.html>`_.


Estimate System Parameters by Solving Inverse Problem
=====================================================

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



This section guides you through the process of spectral estimation using datasets scanned with three different source voltages, utilizing the :func:`xspec.Estimate` module.


Initializing the Estimator
--------------------------

Initialize the `Estimate` object with the energy bins for the spectral data.

.. code-block:: python

    import os
    from xspec import Estimate

    Estimator = Estimate(energies)

Load Data for Estimation
------------------------

Add your normalized radiographs :math:`[y_1, y_2, y_3]`, forward matrices :math:`[A_1, A_2, A_3]`, and spectral models :math:`x_1, x_2, x_3` for each source voltage to the estimator.

- :math:`y_k` should have dimension :math:`Nviews, Nrows, Ncols`
- :math:`A_k` should have dimension :math:`Nviews, Nrows, Ncols, Nenergies`.
- :math:`x_k` should have dimension :math:`Nenergies`.

Assume you have all normalized radiographs and corresponding forward matrices already configured 3 different sources, 1 filter and 1 scintillator, we can load data and spectral models to the estimator.

.. code-block:: python

    normalized_rads = [Your normalized radiographs here]
    forward_matrices = [Your forward matrices here]
    spec_models = [
    [source_1, filter, scintillator],
    [source_2, filter, scintillator],
    [source_3, filter, scintillator],
    ]

    for nrad, forward_matrix, concatenate_models in zip(normalized_rads, forward_matrices, spec_models):
        Estimator.add_data(nrad, forward_matrix, concatenate_models, weight=None)

Fitting the Model
-----------------

Fit the model with the specified learning rate, maximum iterations, stop threshold, optimizer type, and loss type. Optionally, specify the logpath and number of processes.

.. code-block:: python

    learning_rate = 0.02
    max_iterations = 5000
    stop_threshold = 1e-5
    optimizer_type = 'NNAT_LBFGS'
    loss_type = 'transmission'

    Estimator.fit(learning_rate=learning_rate,
                  max_iterations=max_iterations,
                  stop_threshold=stop_threshold,
                  optimizer_type=optimizer_type,
                  loss_type=loss_type,
                  logpath=None,
                  num_processes=1)

Retrieving the Results
----------------------

After fitting the model, retrieve the estimated spectral models and parameters.

.. code-block:: python

    res_spec_models = Estimator.get_spec_models()
    res_params = Estimator.get_params()

    # Process or analyze the retrieved models and parameters as needed
