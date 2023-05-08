#!/bin/bash
# This script purges XSPEC

cd ..
/bin/rm -r build
/bin/rm -r dist
/bin/rm -r xspec.egg-info

pip uninstall xspec
cd dev_scripts