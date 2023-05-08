#!/bin/bash
# This script destroys the conda environment named "xspec" and uninstalls xspec.
# It then creates an "xspec" environment and reinstalls xspec along with the documentation and demo requirements.

# Clean out old installation
source clean_xspec.sh

# Destroy conda environement named xspec and reinstall it
source install_conda_environment.sh

# Install xspec
source install_xspec.sh

# Build documentation
source install_docs.sh