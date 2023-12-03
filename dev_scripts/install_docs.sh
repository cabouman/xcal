#!/bin/bash
# This script installs the documentation.
# You can view documentation pages from xspec/docs/build/index.html .

# Build documentation
conda install pandoc
cd ../docs
pip install -r requirements.txt
make clean html
cd ../dev_scripts
