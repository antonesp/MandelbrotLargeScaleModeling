import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
square_file = "Anton/Results/par_scale_square.csv"

mads_directory = "Mads/Results"
sausage_file = os.path.join(mads_directory, "par_scale_sausage.csv")
sheet_file = os.path.join(mads_directory, "par_scale_sheet.csv")

# Common sequential baseline
seq_file = os.path.join(mads_directory, "seq_scale.csv")

figure_directory = "Anton/Results/Figures"
os.makedirs(figure_directory, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_parallel(filename):
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    # Sort by number of cores
    return data[np.argsort(data[:, 0])]


def load_sequential(filename):
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)

    if data.ndim > 1:
        data = data[0]

    return data


def scaling_metrics(data, seq_data):
    seq_cores = seq_data[0]
    seq_mlup = seq_data[11]
    seq_mem_bw = seq_data[12]

    cores = np.insert(data[:, 0], 0, seq_cores)
    mlups = np.insert(data[:, 11], 0, seq_mlup)
    mem_bw = np.insert(data[:, 12], 0, seq_mem_bw)

    speedup = mlups / seq_mlup
    efficiency = speedup / cores

    return cores, mlups, mem_bw, speedup, efficiency


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
square_data = load_parallel(square_file)
sausage_data = load_parallel(sausage_file)
sheet_data = load_parallel(sheet_file)
seq_data = load_sequential(seq_file)

N_int_val = int(seq_data[1])

square_cores, square_mlups, square_mem_bw, square_speedup, square_eff = \
    scaling_metrics(square_data, seq_data)

sausage_cores, sausage_mlups, sausage_mem_bw, sausage_speedup, sausage_eff = \
    scaling_metrics(sausage_data, seq_data)

sheet_cores, sheet_mlups, sheet_mem_bw, sheet_speedup, sheet_eff = \
    scaling_metrics(sheet_data, seq_data)

all_cores = np.unique(
    np.concatenate([
        square_cores,
        sausage_cores,
        sheet_cores,
    ])
)


# ---------------------------------------------------------------------
# Figure 1: Strong scaling, efficiency, memory bandwidth
# ---------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

# Strong scaling
ax1.plot(
    square_cores,
    square_speedup,
    marker="o",
    linewidth=2,
    label="Parallel (Square)",
)
ax1.plot(
    sausage_cores,
    sausage_speedup,
    marker="s",
    linewidth=2,
    label="Parallel (Sausage)",
)
ax1.plot(
    sheet_cores,
    sheet_speedup,
    marker="D",
    linewidth=2,
    label="Parallel (Sheet)",
)
ax1.plot(
    all_cores,
    all_cores,
    linestyle="--",
    linewidth=2,
    label="Ideal Scaling",
)

ax1.set_xlabel("Number of Cores", fontsize=12)
ax1.set_ylabel("Speedup (MLUP/s)", fontsize=12)
ax1.set_title(f"Strong Scaling (N_int = {N_int_val})", fontsize=13)
ax1.set_xticks(all_cores)
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.legend(fontsize=9)


# Parallel efficiency
ax2.plot(
    square_cores,
    square_eff,
    marker="o",
    linewidth=2,
    label="Parallel (Square)",
)
ax2.plot(
    sausage_cores,
    sausage_eff,
    marker="s",
    linewidth=2,
    label="Parallel (Sausage)",
)
ax2.plot(
    sheet_cores,
    sheet_eff,
    marker="D",
    linewidth=2,
    label="Parallel (Sheet)",
)
ax2.axhline(
    y=1.0,
    linestyle="--",
    linewidth=2,
    label="Ideal Efficiency (100%)",
)

ax2.set_xlabel("Number of Cores", fontsize=12)
ax2.set_ylabel("Efficiency", fontsize=12)
ax2.set_title("Parallel Efficiency", fontsize=13)
ax2.set_xticks(all_cores)
ax2.set_ylim(0, 1.1)
ax2.grid(True, linestyle=":", alpha=0.7)
ax2.legend(fontsize=9)


# Memory bandwidth
ax3.plot(
    square_cores,
    square_mem_bw,
    marker="o",
    linewidth=2,
    label="Parallel (Square)",
)
ax3.plot(
    sausage_cores,
    sausage_mem_bw,
    marker="s",
    linewidth=2,
    label="Parallel (Sausage)",
)
ax3.plot(
    sheet_cores,
    sheet_mem_bw,
    marker="D",
    linewidth=2,
    label="Parallel (Sheet)",
)
ax3.plot(
    [all_cores.min(), 12, 12, all_cores.max()],
    [76800, 76800, 76800 * 2, 76800 * 2],
    linestyle="--",
    linewidth=2,
    label="Memory Bandwidth Reference",
)

ax3.set_xlabel("Number of Cores", fontsize=12)
ax3.set_ylabel("Memory Bandwidth [MB/s]", fontsize=12)
ax3.set_title("Measured Memory Bandwidth", fontsize=13)
ax3.set_xticks(all_cores)
ax3.grid(True, linestyle=":", alpha=0.7)
ax3.legend(fontsize=9)

plt.tight_layout()

scaling_figure = os.path.join(
    figure_directory,
    "scaling_square_sausage_sheet.png",
)

plt.savefig(
    scaling_figure,
    dpi=300,
    bbox_inches="tight",
)
plt.close()




# ---------------------------------------------------------------------
# Figure 2: Compute rate
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    square_cores,
    square_mlups,
    marker="o",
    linewidth=2,
    label="Parallel (Square)",
)
plt.plot(
    sausage_cores,
    sausage_mlups,
    marker="s",
    linewidth=2,
    label="Parallel (Sausage)",
)
plt.plot(
    sheet_cores,
    sheet_mlups,
    marker="D",
    linewidth=2,
    label="Parallel (Sheet)",
)

plt.xlabel("Number of Cores", fontsize=12)
plt.ylabel("Compute Rate [MLUP/s]", fontsize=12)
plt.title("Compute Rate", fontsize=13)
plt.xticks(all_cores)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

compute_rate_figure = os.path.join(
    figure_directory,
    "compute_rate_square_sausage_sheet.png",
)

plt.savefig(
    compute_rate_figure,
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ---------------------------------------------------------------------
# Figure 3: Communication overhead
# ---------------------------------------------------------------------
square_comm_pct = 100 * square_data[:, 8] / square_data[:, 6]
sausage_comm_pct = 100 * sausage_data[:, 8] / sausage_data[:, 6]
sheet_comm_pct = 100 * sheet_data[:, 8] / sheet_data[:, 6]

plt.figure(figsize=(8, 5))

plt.plot(
    square_data[:, 0],
    square_comm_pct,
    marker="o",
    linewidth=2,
    label="Parallel (Square)",
)
plt.plot(
    sausage_data[:, 0],
    sausage_comm_pct,
    marker="s",
    linewidth=2,
    label="Parallel (Sausage)",
)
plt.plot(
    sheet_data[:, 0],
    sheet_comm_pct,
    marker="D",
    linewidth=2,
    label="Parallel (Sheet)",
)

plt.xlabel("Number of Cores", fontsize=12)
plt.ylabel("Communication time [% of compute time]", fontsize=12)
plt.title("Communication Overhead", fontsize=13)
plt.xticks(all_cores)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

communication_figure = os.path.join(
    figure_directory,
    "communication_square_sausage_sheet.png",
)

plt.savefig(
    communication_figure,
    dpi=300,
    bbox_inches="tight",
)
plt.close()


print(f"Saved: {scaling_figure}")
print(f"Saved: {compute_rate_figure}")
print(f"Saved: {communication_figure}")
