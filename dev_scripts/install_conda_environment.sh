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
conda create --name $NAME python=3.10
conda activate $NAME
conda install ipykernel
conda install pandoc
python -m ipykernel install --user --name $NAME --display-name $NAME
cd dev_scripts
