#!/bin/bash
# This script destroys the conda environment named "xspec" and reinstall it.

# Create and activate new conda environment
cd ..
conda deactivate
conda remove env --name xspec --all
conda env create -f environment.yml
source activate xspec
python -m ipykernel install --user --name xspec --display-name "xspec"
cd dev_scripts