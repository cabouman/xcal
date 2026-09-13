============
Installation
============

xcal is installed from source:

.. code-block:: bash

   git clone git@github.com:cabouman/xcal.git
   cd xcal
   pip install .

This installs the Python dependencies (numpy, scipy, torch, h5py,
pyyaml, chemparse, matplotlib) automatically.

Two dependencies are separate:

* **mbirtorch** reads the scanner data and provides the tomography
  models.  It is not on PyPI; install it from
  `its repository <https://github.com/cabouman/mbirtorch>`_.
* **Spekpy** generates reflection tube source spectra.  It is needed
  only for :class:`~xcal.ReflectionSource`:
  ``pip install spekpy``.

To verify the installation, run the test suite:

.. code-block:: bash

   pip install pytest
   pytest tests/

The tests take a few seconds.  For a complete runnable example with a
known ground truth, run:

.. code-block:: bash

   python demo/demo_1_multi_voltage.py

It simulates three scans of a calibration target set at different voltages,
calibrates, and compares the estimated spectrum to the truth.  It
takes about two minutes on a laptop CPU.
