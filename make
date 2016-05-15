#!/bin/sh

nvcc $1 -std=c++11 -I./usr/include -L./usr/lib -lhdf5_cpp -lhdf5 main.cu -o lattice
