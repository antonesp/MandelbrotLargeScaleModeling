#!/bin/bash
#BSUB -J parallel_scaling_square[1-6]
#BSUB -q hpcintro
#BSUB -W 00:15
#BSUB -n 48
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=10GB]"

#BSUB -o Anton/Output/par_scale_square_large_%J_%I.out
#BSUB -e Anton/Errors/par_scale_square_large_%J_%I.err

module load python/3.14.2
module load matplotlib/3.10.8-numpy-2.4.1-python-3.14.2
module load mpi4py/4.1.1-python-3.14.2-openmpi-5.0.9
module load numpy

# Match the sheet scaling experiment
N_cores=(28 32 36 40 44 48)
N_iters=(252 312 336 372 390 456)
N_int=1000

INDEX=$((LSB_JOBINDEX - 1))
CURRENT_N_CORES=${N_cores[$INDEX]}
CURRENT_N_ITERS=${N_iters[$INDEX]}

cd /zhome/e3/5/167986/Desktop/MandelbrotLargeScaleModeling/Project2 || exit 1

# Write to a separate file per job to avoid concurrent write collisions
OUTFILE="Anton/Results/par_scale_square_${CURRENT_N_CORES}.csv"

echo "Running square decomposition with cores = $CURRENT_N_CORES for $CURRENT_N_ITERS iterations..."

DX=$(awk -v n=$N_int 'BEGIN { printf "%.16f", 2.0 / (n + 1) }')

# Negative tolerance prevents convergence from stopping the fixed-iteration benchmark early
OUTPUT=$(mpirun --report-bindings -np "$CURRENT_N_CORES" python3 -m Anton.par_imp_square \
    --n-int "$N_int" \
    --maxiter "$CURRENT_N_ITERS" \
    --tolerence -1)

# Query LSF for the peak memory observed so far for this array element.
# LSB_UNIT_FOR_JOBS_DISPLAY=M requests memory values in MB.
JOB_SPEC="${LSB_JOBID}[${LSB_JOBINDEX}]"
MAX_MEM=$(
    LSB_UNIT_FOR_JOBS_DISPLAY=M \
    bjobs -noheader -o max_mem "$JOB_SPEC" 2>/dev/null \
    | grep -oE '[0-9]+([.][0-9]+)?' \
    | head -1
)

# Keep the CSV numeric even if LSF fails to return a value.
if [ -z "$MAX_MEM" ]; then
    MAX_MEM="nan"
fi

echo "Max memory observed by LSF: ${MAX_MEM} MB"

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
COMM_POINTS=$(echo "$OUTPUT" | grep -m 1 "Communication points/iter:" | awk '{print $3}')

# Columns:
# cores,N_int,dx,true_error,steady_error,it,compute_time,overhead_time,
# comm_time,iter_per_s,grid_kb,mlup,memband_mbs,mflops,max_mem_mb,comm_points_per_iter
echo "$CURRENT_N_CORES,$N_int,$DX,$ERROR,$STEADY_ERROR,$IT,$TIME,$OVERHEAD,$COMM_TIME,$GRIDs,$GRID_KB,$MLUP,$MEMBAND,$MFLOPS,$MAX_MEM,$COMM_POINTS" > "$OUTFILE"

echo "Saved results to $OUTFILE"
