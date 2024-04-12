Examples/Advanced Tutorials
===========================

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

Organized by Application
------------------------

.. toctree::
   :maxdepth: 2

Spectral Models
^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   examples/notebook/configure_spectral_models

CT Foward Projector
^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   examples/notebook/user_forward_projector

Simulated datasets
^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   examples/notebook/simulated_data_3voltages

Spectral Estimation Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   examples/notebook/spectral_estimation_3voltages
