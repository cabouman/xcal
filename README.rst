xspec
=====

Python code for X-ray spectral estimation.

Download xspec
--------------

.. code-block::

	git clone --recursive https://github.com/cabouman/xspec.git


Installing xspec
----------------
    a. Option 1: Clean install from dev_scripts

        *******You can skip all other steps if you do a clean install.******

        To do a clean install, use the command:

        .. code-block::

            cd dev_scripts
            yes | source install_all.sh
            cd ..

    b. Option 2: Manual install

        1. *Create conda environment:*
            Create a new conda environment named ``jax_labs`` using the following commands:

            .. code-block::

                conda env create -f environment.yml
                source activate xspec
                python -m ipykernel install --user --name xspec --display-name "xspec"


        2. *Install package:*

            .. code-block::
                pip install .
Build documentation
-------------------

After the package is installed, you can build the documentation.
In your terminal window,

a.Go to folder docs/
.. code-block::

	cd docs/

b.Install required dependencies
.. code-block::

	pip install -r requirements.txt

c.Build documentation
.. code-block::

	make clean html

d.Open documentation in build/html/index.html. You will see API references on that webpage.


Run Demo
--------

a.Go to folder demo/

.. code-block::

	cd demo/

b.Install required dependencies

.. code-block::

    pip install -r requirements.txt

c.Download simulated dataset

.. code-block::

    curl
    tar

d. Run demo 1: 3 Datasets scanned with 3 different source voltages and same filter and scintillator.

.. code-block::

    python demo_spec_est_3_voltages.py --dataset_path ../sim_data/sim_mv_dataset_rsn_1.hdf5

e. Run demo 2: 3 Datasets scanned with 3 different filters and same source voltage and scintillator.

.. code-block::

	python demo_spec_est_3_filters.py --dataset_path ../sim_data/sim_1v3f1s_dataset.hdf5