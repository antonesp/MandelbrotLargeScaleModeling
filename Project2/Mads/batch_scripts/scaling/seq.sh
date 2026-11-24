#!/bin/sh 
#BSUB -J sequential_scaling
#BSUB -q hpcintro
#BSUB -W 00:20
#BSUB -n 1
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=100GB]"

#BSUB -o Mads/Output/seq_scale_%J.out 
#BSUB -e Mads/Errors/seq_scale_%J.err

module load matplotlib/3.10.6-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

cd /zhome/e1/d/168534/Desktop/MandelbrotLargeScaleModeling/Project2

python3 -m poisson_solver_timed 1000 25