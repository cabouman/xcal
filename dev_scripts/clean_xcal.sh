#!/bin/bash
# This script purges XCAL

cd ..
/bin/rm -r build
/bin/rm -r dist
/bin/rm -r xcal.egg-info

pip uninstall xcal
cd dev_scripts
