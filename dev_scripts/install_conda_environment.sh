#!/bin/bash
# This script destroys the conda environment named "xcal" and reinstall it.

# First check if the target environment is active and deactivate if so
NAME=xcal

ENV_STRING=$((conda env list) | grep $NAME)
if [[ $ENV_STRING == *$NAME* ]]; then
    conda deactivate
fi
cd ..

yes | conda remove env --name $NAME --all
yes | conda create --name $NAME python=3.10
conda activate $NAME
yes | conda install ipykernel
yes | conda install conda-forge::pandoc
python -m ipykernel install --user --name $NAME --display-name $NAME
cd dev_scripts
