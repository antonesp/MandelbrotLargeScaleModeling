#!/bin/bash

#BSUB -J DynamicMandelBrot_small[1-44]
#BSUB -o outputs/scale_compute/small_%J_%I.out
#BSUB -e outputs/scale_compute/small_%J_%I.err

# Reserve one complete 24-core node
#BSUB -n 24
#BSUB -R "span[hosts=1]"

#BSUB -q hpcintro
#BSUB -W 02:00
#BSUB -M 8GB
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[model == XeonE5_2650v4]"


module load matplotlib/3.10.8-numpy-2.4.1-python-3.14.2
module load mpi4py/4.1.1-python-3.14.2-openmpi-5.0.9


cores=(6 12 18 24)

sizes=(
    10000
    11000
    12000
    13000
    14000
    15000
    16000
    17000
    18000
    19000
    20000
)


# Convert single LSF array index into
# core-count index and problem-size index
idx=$((LSB_JOBINDEX - 1))

core_idx=$((idx / 11))
size_idx=$((idx % 11))

ncores=${cores[$core_idx]}
size=${sizes[$size_idx]}


echo "========================================"
echo "MPI ranks:   $ncores"
echo "Image size:  ${size}x${size}"
echo "LSF job:     $LSB_JOBID"
echo "Array index: $LSB_JOBINDEX"
echo "========================================"


mpirun -np "$ncores" \
    --map-by core \
    --bind-to core \
    --rank-by fill \
    --report-bindings \
    python3 MandelbrotSendRecv.py 10 "${size}x${size}"