#!/bin/bash
# This script just installs xspec along with requirements of xspec.

cd ..
pip install -r requirements.txt
pip install -r docs/requirements.txt
pip install -r demo/requirements.txt
pip install .

# Get the operating system name
os=$(uname -s)
# Check if the OS is Linux
if [ "$os" = "Linux" ]; then
    echo "This is a Linux system."
    conda install numpy
else
    echo "This is not a Linux system."
fi
cd dev_scripts
