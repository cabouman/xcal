#!/bin/bash
# Purge the documentation and rebuild it.

cd ../docs
/bin/rm -r build &> /dev/null

make clean html

echo ""
echo "*** The html documentation is at xcal/docs/build/html/index.html ***"
echo ""

cd ../dev_scripts
