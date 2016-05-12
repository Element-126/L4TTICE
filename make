#!/bin/sh

nvcc -std=c++11 -I./usr/include -L./usr/lib -lhdf5 main.cu -o lattice
