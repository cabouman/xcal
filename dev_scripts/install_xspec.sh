#!/bin/bash
# This script just installs xspec along with requirements of xspec, demos, and documentation..
# However, it does not remove the existing installation of xspec.

cd ..
pip install -r requirements.txt
pip install .
pip install -r docs/requirements.txt
cd dev_scripts