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

				conda remove env --name xspec --all
				conda create --name xspec python=3.10
				conda install ipykernel
				conda activate xspec
				python -m ipykernel install --user --name xspec --display-name xspec

        2. *Install package:*

            .. code-block::

                pip install .


	3. *Build documentation*
	    After the package is installed, you can build the documentation.
	    In your terminal window,

		a.Install required dependencies
		.. code-block::
            conda install pandoc
			cd docs/
			pip install -r requirements.txt


		b.Build documentation
		.. code-block::
		
			make clean html
            cd ..

		c.Open documentation in docs/build/html/index.html. You will see API references on that webpage.

	4. *install demo requirement*

            .. code-block::

                pip install -r demo/requirements.txt

Run Demo
--------

a. Go to folder demo/

.. code-block::

	cd demo/



b. Run demo 1: 3 Datasets scanned with 3 different source voltages and the same filter and scintillator.

.. code-block::

    python demo_spec_est_3_voltages.py


