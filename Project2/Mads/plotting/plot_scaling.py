import numpy as np
import matplotlib.pyplot as plt
import os

# Define the folder containing your CSV files
directory = "Mads/Results/" 
par_file = os.path.join(directory, "par_scale_15m.csv")
seq_file = os.path.join(directory, "seq_scale.csv")

# Load the data
par_data = np.genfromtxt(par_file, delimiter=',', skip_header=1)
seq_data = np.genfromtxt(seq_file, delimiter=',', skip_header=1)

# If seq_data accidentally loads as a 2D array, force 1D
if seq_data.ndim > 1:
    seq_data = seq_data[0]

# Extract parallel cores and times
par_cores = par_data[:, 0]
par_time = par_data[:, 6]

# Extract sequential time 
seq_time = seq_data[6]

# Combine sequential (1 core) and parallel data into single arrays
cores = np.insert(par_cores, 0, 1) 
times = np.insert(par_time, 0, seq_time) 
# Calculate Metrics
speedup = times / seq_time
print(cores, speedup)
efficiency = speedup / cores  # Efficiency is Speedup divided by Number of Cores

# --- Plotting ---
# Create a figure with 1 row and 2 columns of plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
N_int_val = int(seq_data[1])

# Plot 1: Strong Scaling (Speedup)
ax1.plot(cores, speedup, marker='o', linewidth=2, label='Measured Speedup', color='tab:blue')
ax1.plot(cores, cores, linestyle='--', linewidth=2, label='Ideal Scaling', color='black')
ax1.set_xlabel('Number of Cores', fontsize=12)
ax1.set_ylabel('Speedup', fontsize=12)
ax1.set_title(f'Strong Scaling (N_int = {N_int_val})', fontsize=14)
ax1.set_xticks(cores)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(fontsize=12)

# Plot 2: Parallel Efficiency
ax2.plot(cores, efficiency, marker='s', linewidth=2, label='Measured Efficiency', color='tab:orange')
ax2.axhline(y=1.0, linestyle='--', linewidth=2, label='Ideal Efficiency (100%)', color='black')
ax2.set_xlabel('Number of Cores', fontsize=12)
ax2.set_ylabel('Efficiency', fontsize=12)
ax2.set_title('Parallel Efficiency', fontsize=14)
ax2.set_xticks(cores)
ax2.set_ylim(0, 1.1)  # Efficiency is usually 0 to 1, leaving a little headroom
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(directory, "Figures/scaling_and_efficiency.png"), dpi=300)
plt.show()
