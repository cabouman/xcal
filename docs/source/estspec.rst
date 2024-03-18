Spectral Estimation Using xspec.Estimate
========================================

This tutorial guides you through the process of spectral estimation using datasets scanned with three different source voltages, utilizing the `xspec.Estimate` module. We'll cover setting up the estimation environment, adding data for estimation, fitting the model, and retrieving the estimated spectral models and parameters.

Prerequisites
-------------

Before starting, ensure you have:

- Python environment set up
- Installed the necessary libraries, including xspec and its dependencies
- Access to datasets scanned with three source voltages

Setting Up
----------

Start by defining the initial conditions for the estimation process, including the learning rate, maximum iterations, stop threshold, and optimizer type.

.. code-block:: python

    import os
    from xspec import Estimate

    learning_rate = 0.02
    max_iterations = 5000
    stop_threshold = 1e-5
    optimizer_type = 'NNAT_LBFGS'

Preparing the Output Directory
-------------------------------

Prepare the output directory where the log files will be saved during the estimation process.

.. code-block:: python

    savefile_name = 'case_mv_%s_lr%.0e' % (optimizer_type, learning_rate)
    os.makedirs('./output_3_source_voltages/log/', exist_ok=True)

Initializing the Estimator
--------------------------

Initialize the `Estimate` object with the energy bins for the spectral data.

.. code-block:: python

    energies = [Your energy bins here]  # Define your energy bins
    Estimator = Estimate(energies)

Adding Data for Estimation
--------------------------

Add your normalized radiographs, forward matrices, and spectral models for each source voltage to the estimator.

.. code-block:: python

    normalized_rads = [Your normalized radiographs here]  # Define your normalized radiographs
    forward_matrices = [Your forward matrices here]  # Define your forward matrices
    spec_models = [Your spectral models here]  # Define your spectral models

    for nrad, forward_matrix, concatenate_models in zip(normalized_rads, forward_matrices, spec_models):
        Estimator.add_data(nrad, forward_matrix, concatenate_models, weight=None)

Fitting the Model
-----------------

Fit the model with the specified learning rate, maximum iterations, stop threshold, optimizer type, and loss type. Optionally, specify the logpath and number of processes.

.. code-block:: python

    Estimator.fit(learning_rate=learning_rate,
                  max_iterations=max_iterations,
                  stop_threshold=stop_threshold,
                  optimizer_type=optimizer_type,
                  loss_type='transmission',
                  logpath='./output_3_source_voltages/log/',
                  num_processes=1)

Retrieving the Results
----------------------

After fitting the model, retrieve the estimated spectral models and parameters.

.. code-block:: python

    res_spec_models = Estimator.get_spec_models()
    res_params = Estimator.get_params()

    # Process or analyze the retrieved models and parameters as needed

Conclusion
----------

This tutorial has walked you through the process of spectral estimation using `xspec.Estimate`, from setting up the estimator to retrieving the estimated results. This approach allows for accurate spectral estimation across different source voltages, enhancing the analysis of scanned datasets.
