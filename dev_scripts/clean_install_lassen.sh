#!/bin/bash
# This script purges XSPEC

cd ..
/bin/rm -r build
/bin/rm -r dist
/bin/rm -r xspec.egg-info

#!/bin/bash
# This script destroys the conda environment named "xspec" and reinstall it.

# First check if the target environment is active and deactivate if so
NAME=xspec

ENV_STRING=$((conda env list) | grep $NAME)
if [[ $ENV_STRING == *$NAME* ]]; then
    conda deactivate
fi

# Load cuda/11.8
module load cuda/11.8.0

conda remove env --name $NAME --all
conda create --name $NAME python=3.9
conda activate $NAME

# register LLNL SSL certificates
conda config --env --set ssl_verify /etc/pki/tls/cert.pem
 
# register LC's local conda channel for Open-CE
condachannel="/collab/usr/global/tools/opence/blueos_3_ppc64le_ib_p9/opence-1.9.1/condabuild-py3.9-cuda11.8"
conda config --env --prepend channels "file://$condachannel"

# install some packages
conda install -y pytorch=2.0.1=cuda11.8_py39_1

conda install ipykernel
conda install pandoc 
conda install opencv
conda install scikit-image
conda install matplotlib
conda install h5py

python -m ipykernel install --user --name $NAME --display-name $NAME

pip install -r requirements.txt
# pip install -r docs/requirements.txt

pip install -r demo/requirements.txt
pip install .


