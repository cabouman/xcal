#!/bin/bash
# Install xcal from scratch into a fresh conda environment, then
# build the documentation.  mbirtorch is a separate install: it is
# not on PyPI, so install it from its repository before using xcal
# with real scanner data.

NAME="xcal"
PYTHON_VERSION="3.11"

# Remove any previous builds.
cd ..
/bin/rm -r docs/build &> /dev/null
/bin/rm -r dist &> /dev/null
/bin/rm -r "$NAME.egg-info" &> /dev/null
/bin/rm -r build &> /dev/null
cd dev_scripts

# Deactivate all conda environments.
while [ ${#CONDA_DEFAULT_ENV} -gt 0 ]; do
  echo "Deactivating $CONDA_DEFAULT_ENV"
  conda deactivate
done
echo "No conda environment active"

# Remove the environment if it exists.
output=$(yes | conda remove --name $NAME --all 2>&1)
if echo "$output" | grep -q "DirectoryNotACondaEnvironmentError:"; then
  conda activate $NAME
  CUR_ENV_PATH=$CONDA_PREFIX
  conda deactivate
  rm -rf $CUR_ENV_PATH
fi

# Create and activate a new environment.
yes | conda create -n $NAME python="$PYTHON_VERSION"
conda activate $NAME

# Editable installs.  A non-editable install of the same package
# would replace the editable one, freezing the env at
# install-time code.
pip install -e ..
pip install -e "..[test]"
pip install -e "..[docs]"

source build_docs.sh

red=`tput setaf 1`
reset=`tput sgr0`
echo " "
echo "Use"
echo "${red}   conda activate xcal   ${reset}"
echo "to activate the conda environment."
echo " "
echo "mbirtorch is not on PyPI.  Install it separately from"
echo "   https://github.com/cabouman/mbirtorch"
echo "to run xcal on scanner data."
echo " "
