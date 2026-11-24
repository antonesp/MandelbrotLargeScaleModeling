#!/bin/sh
#BSUB -J square_memory_test
#BSUB -q hpcintro
#BSUB -W 00:15
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "select[model == XeonE5_2650v4]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -o Anton/Output/memory_test_%J.out
#BSUB -e Anton/Errors/memory_test_%J.err

module load python/3.14.2
module load matplotlib/3.10.8-numpy-2.4.1-python-3.14.2
module load mpi4py/4.1.1-python-3.14.2-openmpi-5.0.9
module load numpy

cd /zhome/e3/5/167986/Desktop/MandelbrotLargeScaleModeling/Project2

N_INT=1000
N_ITERS=50

echo "========================================"
echo "Memory-bandwidth test"
echo "Allocated cores : 24"
echo "MPI ranks       : 12"
echo "Grid            : ${N_INT}^3"
echo "Iterations      : ${N_ITERS}"
echo "========================================"

echo
echo "CPU topology:"
lscpu | grep -E "Model name|Socket|Core|NUMA"

echo
echo "MPI run:"
mpirun -np 12 \
    --map-by core \
    --bind-to core \
    --rank-by fill \
    --report-bindings \
    python3 -m Anton.par_imp_square \
        --n-int "$N_INT" \
        --maxiter "$N_ITERS" \
        --tolerence -1