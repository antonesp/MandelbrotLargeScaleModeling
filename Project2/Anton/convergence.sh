#!/bin/bash

#BSUB -J PoissonConvergence
#BSUB -o outputs/PoissonConvergence_%J.out
#BSUB -e outputs/PoissonConvergence_%J.err
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

N_VALUES=(10 14 19 24 28 33 38 43 47 52 57 62 66 71 76 81 85 90 95 100)
ERRORS=()

for N in "${N_VALUES[@]}"
do
    echo "Running N_int=$N"

    ERROR=$(mpirun -np 8 python3 -m Anton.parallel_imp \
        --n-int "$N")

    ERRORS+=("$ERROR")

    echo "N_int=$N, error=$ERROR"
done

OUTPUT="Anton/convergence_results.csv"

echo "N_int,error" > "$OUTPUT"

for i in "${!N_VALUES[@]}"
do
    echo "${N_VALUES[$i]},${ERRORS[$i]}" >> "$OUTPUT"
done

echo "Saved results to $OUTPUT"