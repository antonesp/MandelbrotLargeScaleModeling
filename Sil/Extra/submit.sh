#!/bin/sh
#BSUB -J Mandelbrot
#BSUB -q hpcintro
#BSUB -n 8
#BSUB -R "span[ptile=4]"

module load matplotlib/3.10.8-numpy-2.4.1-python-3.14.2
module load mpi4py/4.1.1-python-3.14.2-openmpi-5.0.9

python3 Non-blocking.py