#!/bin/bash
# This script purges XSPEC

cd ..
/bin/rm -r build
/bin/rm -r dist
/bin/rm -r xspec.egg-info

# This script install xspec to an exist environment.

NAME=opence-1.9.1
conda activate $NAME

# Below are some scripts following opence-1.9.1 https://lc.llnl.gov/confluence/pages/viewpage.action?pageId=785286611.
# register LLNL SSL certificates
# conda config --env --set ssl_verify /etc/pki/tls/cert.pem
 
# register LC's local conda channel for Open-CE
# condachannel="/collab/usr/global/tools/opence/blueos_3_ppc64le_ib_p9/opence-1.9.1/condabuild-py3.9-cuda11.8"
# conda config --env --prepend channels "file://$condachannel"

# install pytorch
# conda install -y pytorch=2.0.1=cuda11.8_py39_1

# Conda install required packages
conda install ipykernel
conda install pandoc 
conda install opencv
conda install scikit-image
conda install matplotlib
conda install h5py

# Register conda environment in jupyter notebook.
python -m ipykernel install --user --name $NAME --display-name $NAME

pip install -r requirements.txt
# pip install -r docs/requirements.txt

pip install -r demo/requirements.txt
pip install .


