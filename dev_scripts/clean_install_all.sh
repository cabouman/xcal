#!/bin/bash
# This script destroys the conda environment named "xcal" and reinstalls it.
# It then installs xcal package.
# It also installs the documentation.

# Clean out old installation
source clean_xcal.sh

# Destroy conda environement named xcal and reinstall it
yes | source install_conda_environment.sh
conda activate xcal

# Install xcal
yes | source install_xcal.sh

# Build documentation
# yes | source install_docs.sh
