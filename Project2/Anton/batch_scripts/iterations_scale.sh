#!/bin/bash

#BSUB -J PoissonIterationPerformance
#BSUB -o outputs/PoissonIterationPerformance_%J.out
#BSUB -e outputs/PoissonIterationPerformance_%J.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -q hpcintro
#BSUB -W 00:10
#BSUB -M 2GB
#BSUB -R "select[model == XeonE5_2650v4]"

module load python/3.14.2
module load matplotlib/3.10.8-numpy-2.4.1-python-3.14.2
module load mpi4py/4.1.1-python-3.14.2-openmpi-5.0.9
module load numpy

# Fixed problem parameters
N_INT=200
N_RANKS=8

# Vary the maximum number of iterations
MAX_ITERS=(10 25 50 100 250 500 1000 2500 5000 10000)

OUTPUT="Anton/iterations_per_second.csv"

echo "max_iterations,iterations_per_second" > "$OUTPUT"

for MAX_IT in "${MAX_ITERS[@]}"
do
    echo "Running maxiter=$MAX_IT"

    RATE=$(mpirun -np "$N_RANKS" python3 -m Anton.parallel_imp \
        --n-int "$N_INT" \
        --maxiter "$MAX_IT" \
        --tolerence 0 \
        | awk '/Iterations per second/ {print $4}')

    echo "$MAX_IT,$RATE" >> "$OUTPUT"

    echo "maxiter=$MAX_IT, iterations/sec=$RATE"
done

echo "Results saved to $OUTPUT"