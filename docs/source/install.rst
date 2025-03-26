============
Installation
============

The ``xspec`` package is currently only available to download and install from source available from `XSPEC <https://github.com/cabouman/xspec>`_.

Step 1: Clone repository
------------------------

.. code-block:: bash

   git clone git@github.com:cabouman/xspec.git
   cd xspec

Step 2: Install xspec
---------------------

Two options are listed below for installing xspec.
Option 1 only requires that a bash script be run, but it is less flexible.
Option 2 explains how to perform manual installation.

Option 1: Clean install from dev_scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To do a clean install, use the command:

.. code-block:: bash

   cd dev_scripts
   source ./clean_install_all.sh
   cd ..

Option 2: Manual install
^^^^^^^^^^^^^^^^^^^^^^^^

1. **Create conda environment:**
   Create a new conda environment named ``xspec`` using the following commands:

   .. code-block:: bash

      conda remove env --name xspec --all
      conda create --name xspec python=3.10
      conda activate xspec
      conda install ipykernel
      python -m ipykernel install --user --name xspec --display-name xspec

2. **Install package:**

   .. code-block:: bash

      pip install -r requirements.txt
      pip install .

