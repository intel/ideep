#!/bin/bash

# Install script for Anaconda environments on macOS and linux.
# This script is not supposed to be called directly, but should be run by:
#
# $ cd <path to ideep, e.g. ~/ideep>
# $ conda build conda
#
#
# If you're debugging this, it may be useful to use the env that conda build is
# using:
# $ cd <anaconda_root>/conda-bld/ideep_<timestamp>
# $ source activate _h_env_... # some long path with lots of placeholders
#
# Also, failed builds will accumulate those ideep_<timestamp> directories. You
# can remove them after a succesfull build with
# $ conda build purge
#
git submodule update --init
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
cd ../python
python setup.py install
