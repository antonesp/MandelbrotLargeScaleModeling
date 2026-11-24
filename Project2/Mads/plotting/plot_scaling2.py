import numpy as np
import matplotlib.pyplot as plt
import os

# Define the folder containing your CSV files
directory = "Mads/Results/"
par_file = os.path.join(directory, "par_scale_sausage.csv")
sheet_file = os.path.join(directory, "par_scale_sheet.csv")
seq_file = os.path.join(directory, "seq_scale.csv")

# Load the data
par_data = np.genfromtxt(par_file, delimiter=',', skip_header=1)
sheet_data = np.genfromtxt(sheet_file, delimiter=',', skip_header=1)
seq_data = np.genfromtxt(seq_file, delimiter=',', skip_header=1)

# Force 1D for seq_data and 2D for parallel datasets
if seq_data.ndim > 1:
    seq_data = seq_data[0]
if par_data.ndim == 1:
    par_data = par_data[np.newaxis, :]
if sheet_data.ndim == 1:
    sheet_data = sheet_data[np.newaxis, :]

# Sequential baseline (assumes seq_scale.csv uses the 14-column schema)
seq_mlup = seq_data[11]
seq_mem_bw = seq_data[12]
N_int_val = int(seq_data[1])

# Combine sequential (1 core) with standard parallel data
cores = np.insert(par_data[:, 0], 0, seq_data[0])
mlups = np.insert(par_data[:, 11], 0, seq_mlup)          # MLUP/s
mem_bw = np.insert(par_data[:, 12], 0, seq_mem_bw)       # MemBand [MB/s]
speedup = mlups / seq_mlup
efficiency = speedup / cores

# Combine sequential (1 core) with sheet parallel data
sheet_cores = np.insert(sheet_data[:, 0], 0, seq_data[0])
sheet_mlups = np.insert(sheet_data[:, 11], 0, seq_mlup)
sheet_mem_bw = np.insert(sheet_data[:, 12], 0, seq_mem_bw)
sheet_speedup = sheet_mlups / seq_mlup
sheet_efficiency = sheet_speedup / sheet_cores

# Unified x-axis ticks in case the two files have different core counts
all_cores = np.unique(np.concatenate([cores, sheet_cores]))

# --- Plotting ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Strong Scaling (Speedup)
ax1.plot(cores, speedup, marker='o', linewidth=2, label='Parallel (Sausage)', color='tab:green')
ax1.plot(sheet_cores, sheet_speedup, marker='D', linewidth=2, label='Parallel (Sheet)', color='tab:red')
ax1.plot(all_cores, all_cores, linestyle='--', linewidth=2, label='Ideal Scaling', color='black')
ax1.set_xlabel('Number of Cores', fontsize=12)
ax1.set_ylabel('Speedup (MLUP/s)', fontsize=12)
ax1.set_title(f'Strong Scaling (N_int = {N_int_val})', fontsize=13)
ax1.set_xticks(all_cores)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(fontsize=10)

# Plot 2: Parallel Efficiency
ax2.plot(cores, efficiency, marker='s', linewidth=2, label='Parallel (Sausage)', color='tab:green')
ax2.plot(sheet_cores, sheet_efficiency, marker='D', linewidth=2, label='Parallel (Sheet)', color='tab:red')
ax2.axhline(y=1.0, linestyle='--', linewidth=2, label='Ideal Efficiency (100%)', color='black')
ax2.set_xlabel('Number of Cores', fontsize=12)
ax2.set_ylabel('Efficiency', fontsize=12)
ax2.set_title('Parallel Efficiency', fontsize=13)
ax2.set_xticks(all_cores)
ax2.set_ylim(0, 1.1)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(fontsize=10)

# Plot 3: Measured Memory Bandwidth
ax3.plot(cores, mem_bw, marker='^', linewidth=2, label='Parallel (Sausage)', color='tab:green')
ax3.plot(sheet_cores, sheet_mem_bw, marker='D', linewidth=2, label='Parallel (Sheet)', color='tab:red')
ax3.set_xlabel('Number of Cores', fontsize=12)
ax3.set_ylabel('Memory Bandwidth [MB/s]', fontsize=12)
ax3.set_title('Measured Memory Bandwidth', fontsize=13)
ax3.set_xticks(all_cores)
ax3.grid(True, linestyle=':', alpha=0.7)
ax3.legend(fontsize=10)

plt.tight_layout()
os.makedirs(os.path.join(directory, "Figures"), exist_ok=True)
plt.savefig(os.path.join(directory, "Figures/scaling_and_bandwidth.png"), dpi=300)
plt.show()