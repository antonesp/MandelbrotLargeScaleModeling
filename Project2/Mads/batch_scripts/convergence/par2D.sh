#!/bin/sh 
#BSUB -J conv_par2D
#BSUB -q hpcintro
#BSUB -W 00:30
#BSUB -n 24
#BSUB -R "span[ptile=24]"
#BSUB -R "rusage[mem=1GB]"

#BSUB -o Mads/Output/convergence/par2D_%J_%I.out 
#BSUB -e Mads/Errors/convergence/par2D_%J_%I.err

module load matplotlib/3.10.6-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8

N_int=(
    20 22 24 26 28 30 32 34 36 38 
    40 42 44 46 48 50 52 54 56 58 
    60 62 64 66 68 70 72 74 76 78 
    80 82 84 86 88 90 92 94 96 98)

# INDEX=$(($LSB_JOBINDEX - 1))
# CURRENT_N_INT=${N_int[$INDEX]}

cd /zhome/e1/d/168534/Desktop/MandelbrotLargeScaleModeling/Project2

OUTFILE="Mads/Results/Convergence/par2D.csv"
echo "N_int,dx,error" > $OUTFILE

# Loop through the bash array one by one
for CURRENT_N_INT in "${N_int[@]}"; do
    echo "Running N_int = $CURRENT_N_INT..."
    
    DX=$(awk -v n=$CURRENT_N_INT 'BEGIN { printf "%.16f", 2.0 / (n + 1) }')
    
    # Run mpirun and capture output
    OUTPUT=$(mpirun -np 12 python3 -m Mads.src.par2D $CURRENT_N_INT)
    
    # Extract the error
    ERROR=$(printf '%s\n' "$OUTPUT" | awk '/True final error:/ {print $NF; exit}')

    # Safely append to the single file because the jobs are running one at a time
    echo "$CURRENT_N_INT,$DX,$ERROR" >> $OUTFILE
done

# python3 Mads/post_process.py