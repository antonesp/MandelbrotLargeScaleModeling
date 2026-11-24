#!/bin/bash
#BSUB -J parallel_scaling_square_socket[1-4]
#BSUB -q hpcintro
#BSUB -W 00:15
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "select[model == XeonE5_2650v4]"

# Separate log names from the previous scaling experiment
#BSUB -o Anton/Output/square_bound_test_%J_%I.out
#BSUB -e Anton/Errors/square_bound_test_%J_%I.err


module load matplotlib/3.10.8-numpy-2.4.1-python-3.14.2
module load mpi4py/4.1.1-python-3.14.2-openmpi-5.0.9

cd /zhome/e3/5/167986/Desktop/MandelbrotLargeScaleModeling/Project2 || exit 1

# Reserve the full 24-core node, but use at most one 12-core socket.
N_cores=(9 10 11 12)

# Use fewer iterations for the low-core-count cases so the job array
# remains practical within the 25-minute wall-clock limit.
N_iters=(80 85 85 85)

N_int=1000

INDEX=$((LSB_JOBINDEX - 1))
CURRENT_N_CORES=${N_cores[$INDEX]}
CURRENT_N_ITERS=${N_iters[$INDEX]}

# New filenames so the existing par_scale_square_*.csv files are untouched.
OUTFILE="Anton/Results/square_bound_test${CURRENT_N_CORES}.csv"

echo "============================================================"
echo "Single-socket square-decomposition scaling test"
echo "Allocated cores : 24"
echo "MPI ranks       : ${CURRENT_N_CORES}"
echo "Grid            : ${N_int}^3"
echo "Iterations      : ${CURRENT_N_ITERS}"
echo "Result file     : ${OUTFILE}"
echo "============================================================"

echo
echo "CPU topology:"
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA"

# Build a CPU list containing one hardware thread from each physical core
# on socket 0. This keeps all MPI ranks on one socket.
SOCKET_CPUS=$(
    lscpu -p=CPU,CORE,SOCKET |
    awk -F, '
        $1 !~ /^#/ && $3 == 0 && !seen[$2]++ {
            if (out != "") out = out ","
            out = out $1
        }
        END { print out }
    '
)

echo
echo "Socket-0 CPU set: ${SOCKET_CPUS}"
echo

DX=$(awk -v n=$N_int 'BEGIN { printf "%.16f", 2.0 / (n + 1) }')

# Negative tolerance prevents convergence from stopping the fixed-iteration
# benchmark early.
OUTPUT=$(mpirun -np "$CURRENT_N_CORES" \
    --map-by "pe-list=${SOCKET_CPUS}:ordered" \
    --bind-to core \
    --report-bindings \
    python3 -m Anton.par_imp_square \
        --n-int "$N_int" \
        --maxiter "$CURRENT_N_ITERS" \
        --tolerence -1)

# Preserve the program output in the LSF log.
echo "$OUTPUT"

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

# Same 14-column schema as the previous scaling files.
echo "$CURRENT_N_CORES,$N_int,$DX,$ERROR,$STEADY_ERROR,$IT,$TIME,$OVERHEAD,$COMM_TIME,$GRIDs,$GRID_KB,$MLUP,$MEMBAND,$MFLOPS" > "$OUTFILE"

echo
echo "Saved results to $OUTFILE"
