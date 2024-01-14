================================
Build documentation with Sphinx
================================

Build HTML locally
------------------

After the package is installed, the documentation can be built by following these steps in the terminal:

1. Install required dependencies:

.. code-block:: bash

   cd docs/
   conda install pandoc
   pip install -r requirements.txt

2. Build documentation:

.. code-block:: bash

   make clean html
   cd ..

3. View the documentation:

   The built HTML documentation can be found at ``docs/build/html/index.html``. Opening this file in a web browser will display the API references and other documentation.
