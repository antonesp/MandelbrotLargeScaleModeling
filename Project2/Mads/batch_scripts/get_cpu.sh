#!/bin/sh 
#BSUB -J GET_CPU
#BSUB -q hpcintro
#BSUB -W 00:25
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=30GB]"

#BSUB -o Mads/Output/GET_CPU_%J.out 
#BSUB -e Mads/Errors/GET_CPU_%J.err

module load matplotlib/3.10.6-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

cd /zhome/e1/d/168534/Desktop/MandelbrotLargeScaleModeling/Project2

lscpu | grep "Model name"
lscpu | grep -E "Model name|Socket|Core|NUMA"
mpirun -np 12 --report-bindings python3 -m Mads.parallel_imp 1000 50
# mpirun -np 12 -map-by ppr:6:package python3 -m Mads.parallel_imp 1000 50