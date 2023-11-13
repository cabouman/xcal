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


Build documentation
-------------------

a.Go to folder demo/
.. code-block::

	cd demo/

b.Install required dependencies
.. code-block::

    pip install -r requirements.txt

c.Download simulated dataset
.. code-block::
    curl

d. Run demo
.. code-block::
    python
