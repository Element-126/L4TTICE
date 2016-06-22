#!/bin/sh

cmd="nvcc $* -std=c++11 -I./usr/include -I./usr/cub -L./usr/lib -lhdf5_cpp -lhdf5 main.cu -o lattice"
echo $cmd
eval $cmd
