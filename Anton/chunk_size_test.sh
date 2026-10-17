#!/bin/bash

#BSUB -J ChunkSize[1-10]
#BSUB -o outputs/chunk_size/chunk_%J_%I.out
#BSUB -e outputs/chunk_size/chunk_%J_%I.err

# Reserve one complete 24-core node
#BSUB -n 24
#BSUB -R "span[hosts=1]"

#BSUB -q hpcintro
#BSUB -W 00:15
#BSUB -M 4GB
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model == XeonE5_2650v4]"

module load matplotlib/3.10.8-numpy-2.4.1-python-3.14.2
module load mpi4py/4.1.1-python-3.14.2-openmpi-5.0.9


ncores=24
size=10000

chunk_sizes=(
    1
    2
    4
    8
    16
    32
    64
    128
    256
    400
)

chunk=${chunk_sizes[$((LSB_JOBINDEX - 1))]}


echo "========================================"
echo "MPI ranks:   $ncores"
echo "Image size:  ${size}x${size}"
echo "Chunk size:  $chunk"
echo "LSF job:     $LSB_JOBID"
echo "Array index: $LSB_JOBINDEX"
echo "========================================"


mpirun -np "$ncores" \
    --map-by core \
    --bind-to core \
    --rank-by fill \
    --report-bindings \
    python3 MandelbrotSendRecv.py "$chunk" "${size}x${size}"