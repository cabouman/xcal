#!/bin/bash
# This script installs the documentation.
# You can view documentation pages from xspec/docs/build/index.html .

# Build documentation
cd ../docs
make clean html
cd ../dev_scripts
