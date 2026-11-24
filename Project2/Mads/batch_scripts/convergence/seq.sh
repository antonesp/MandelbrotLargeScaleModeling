#!/bin/sh 
#BSUB -J seq_conv
#BSUB -q hpcintro
#BSUB -W 00:30
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"

#BSUB -o Mads/Output/seq_conv_%J.out 
#BSUB -e Mads/Errors/seq_conv_%J.err

module load matplotlib/3.10.6-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

cd /zhome/e1/d/168534/Desktop/MandelbrotLargeScaleModeling/Project2
python3 -m Mads.src.seq_conv