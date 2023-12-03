#!/bin/bash
# This script destroys the conda environment named "xspec" and reinstall it.

# First check if the target environment is active and deactivate if so
NAME=xspec

ENV_STRING=$((conda env list) | grep $NAME)
if [[ $ENV_STRING == *$NAME* ]]; then
    conda deactivate
fi
cd ..

conda remove env --name $NAME --all
conda env create -f environment.yml
conda activate $NAME
python -m ipykernel install --user --name $NAME --display-name $NAME
cd dev_scripts
