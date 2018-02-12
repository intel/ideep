#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib/
export OMP_NUM_THREADS=1
./test 
