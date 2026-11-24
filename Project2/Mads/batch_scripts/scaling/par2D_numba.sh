#!/bin/sh 
#BSUB -J par_scale_saus_numba[1-6]
#BSUB -q hpcintro
#BSUB -W 00:15
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=10GB]"

#BSUB -o Mads/Output/par_scale_saus_numba_%J_%I.out 
#BSUB -e Mads/Errors/par_scale_saus_numba_%J_%I.err

module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
module load matplotlib/3.10.6-numpy-2.3.1-python-3.12.11
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

N_cores=(4 8 12 16 20 24)
# N_iters=(100 160 170 230 290 350)
N_iters=(20 20 20 20 20 20)
N_int=1000

INDEX=$(($LSB_JOBINDEX - 1))
CURRENT_N_CORES=${N_cores[$INDEX]}
CURRENT_N_ITERS=${N_iters[$INDEX]}

cd /zhome/e1/d/168534/Desktop/MandelbrotLargeScaleModeling/Project2

# Write to a separate file per job to avoid concurrent write collisions
OUTFILE="Mads/Results/par_scale_saus_${CURRENT_N_CORES}.csv"

echo "Running with cores = $CURRENT_N_CORES for $CURRENT_N_ITERS iterations..."
DX=$(awk -v n=$N_int 'BEGIN { printf "%.16f", 2.0 / (n + 1) }')

# Pass -1 for max_err so it won't stop early due to convergence
OUTPUT=$(mpirun -np $CURRENT_N_CORES python3 -m Mads.par_saus_numba $N_int $CURRENT_N_ITERS -1)

# Parse timing
TIME=$(echo "$OUTPUT" | grep -m 1 "Compute time:" | awk '{print $3}')
GRIDs=$(echo "$OUTPUT" | grep -m 1 "Compute time:" | awk '{print $5}' | tr -d '(,')
OVERHEAD=$(echo "$OUTPUT" | grep -m 1 "Overhead time:" | awk '{print $3}')
COMM_TIME=$(echo "$OUTPUT" | grep -m 1 "Communication time:" | awk '{print $3}')

# Parse convergence
IT=$(echo "$OUTPUT" | grep -m 1 "Convergence (it =" | awk '{print $4}' | tr -d '),')
STEADY_ERROR=$(echo "$OUTPUT" | grep -m 1 "Steady-state error:" | awk '{print $3}')
ERROR=$(echo "$OUTPUT" | grep -m 1 "True final error:" | awk '{print $4}')

# Parse performance
GRID_KB=$(echo "$OUTPUT" | grep -m 1 "Performance (Grid:" | awk '{print $3}' | tr -d ',')
MLUP=$(echo "$OUTPUT" | grep -m 1 "Throughput:" | awk '{print $2}')
MEMBAND=$(echo "$OUTPUT" | grep -m 1 "Memory bandwidth:" | awk '{print $3}')
MFLOPS=$(echo "$OUTPUT" | grep -m 1 "Compute rate:" | awk '{print $3}')

# Columns: cores,N_int,dx,true_error,steady_error,it,compute_time,overhead_time,comm_time,iter_per_s,grid_kb,mlup,memband_mbs,mflops
echo "$CURRENT_N_CORES,$N_int,$DX,$ERROR,$STEADY_ERROR,$IT,$TIME,$OVERHEAD,$COMM_TIME,$GRIDs,$GRID_KB,$MLUP,$MEMBAND,$MFLOPS" > "$OUTFILE"