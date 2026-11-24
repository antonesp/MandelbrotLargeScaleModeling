#!/bin/bash
JOBID=$1

if [ -z "$JOBID" ]; then
    echo "Usage: bash update_mem.sh <JOBID>"
    exit 1
fi

N_cores=(4 8 12 16 20 24)

for i in {1..6}; do
    OUTLOG="Mads/Output/scaling/par1D_${JOBID}_${i}.out"
    CORES=${N_cores[$((i - 1))]}
    CSV="Mads/Results/scaling/par1D_${CORES}.csv"

    if [ -f "$OUTLOG" ] && [ -f "$CSV" ]; then
        # Extract the numeric value from "Max Memory : 46264 MB"
        MAX_MEM=$(awk '/Max Memory :/ {print $4}' "$OUTLOG")

        # Update column 16 (MAX_MEM) in the CSV file
        awk -F',' -v OFS=',' -v mem="$MAX_MEM" '{$16 = mem; print}' "$CSV" > "${CSV}.tmp" && mv "${CSV}.tmp" "$CSV"
        
        echo "Cores $CORES (Job ${JOBID}[${i}]): Updated MAX_MEM = ${MAX_MEM} MB"
    else
        echo "Missing $OUTLOG or $CSV"
    fi
done

python3 Mads/src/combine_files.py