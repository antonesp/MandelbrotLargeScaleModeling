In this folder you will find the code used to generate results for Group 1's Mandelbrot project. In order to run any script load the following modules.
module load matplotlib/3.10.6-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

Then execute any program with

mpirun -np 2 python3 Dynamic/Blocking.py 10 10000x10000
mpirun -np 2 python3 Static/Script.py 10 10000x10000