#!/bin/bash
# This script destroys the conda environment named "xspec" and reinstall it.

# Create and activate new conda environment
cd ..
conda deactivate
conda remove env --name xspec --all
conda create --name xspec python=3.8
conda activate xspec
cd dev_scripts