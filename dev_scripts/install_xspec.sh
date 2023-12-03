#!/bin/bash
# This script just installs xspec along with requirements of xspec.

cd ..
pip install -r requirements.txt
pip install -r docs/requirements.txt
pip install -r demo/requirements.txt
pip install .
cd dev_scripts
