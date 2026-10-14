#BSUB -J testbuffer
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -q hpcintro
#BSUB -W 00:10
#BSUB -R "rusage[mem=1GB]"
#BSUB -M 1GB
#BSUB -R "select[model == XeonE5_2650v4]"

module load python/3.14.2
module load matplotlib/3.10.8-numpy-2.4.1-python-3.14.2
module load mpi4py/4.1.1-python-3.14.2-openmpi-5.0.9
module load numpy


mpirun -np 2 python3 testbuffersize.py