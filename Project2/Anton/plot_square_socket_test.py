import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
RESULTS_DIR = Path("Anton/Results")
FIGURE_DIR = RESULTS_DIR / "Figures" / "square_bound"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CORES = [9, 10, 11, 12]

# Estimated source-level memory traffic for the current NumPy iteration:
#   stencil update            ~192 B/LUP
#   max(abs(phi-phi_prev))     ~48 B/LUP
#   total                     ~240 B/LUP
BYTES_PER_LUP = 240.0

# Intel Xeon E5-2650 v4 theoretical memory bandwidth per socket.
SOCKET_BW_GBS = 76.8

COLUMNS = [
    "cores",
    "N_int",
    "dx",
    "true_error",
    "steady_error",
    "it",
    "compute_time",
    "overhead_time",
    "comm_time",
    "iter_per_s",
    "grid_kb",
    "mlup",
    "memband_mbs",
    "mflops",
]


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_result(path):
    """
    Load one single-run CSV.

    The socket-test batch jobs write one data row without a header.
    This function also tolerates a header if one is added later.
    """
    with path.open("r", newline="") as f:
        rows = [
            [value.strip() for value in row]
            for row in csv.reader(f)
            if any(value.strip() for value in row)
        ]

    if not rows:
        raise ValueError(f"No data found in {path}")

    # Skip a header row if present.
    if rows[0][0].lower() == "cores":
        rows = rows[1:]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one result row in {path}, found {len(rows)}"
        )

    if len(rows[0]) != len(COLUMNS):
        raise ValueError(
            f"{path} has {len(rows[0])} columns; expected {len(COLUMNS)}"
        )

    return np.asarray(rows[0], dtype=float)


rows = []
for core_count in CORES:
    filename = RESULTS_DIR / f"square_bound_test{core_count}.csv"

    if not filename.exists():
        raise FileNotFoundError(
            f"Missing result file: {filename}\n"
            f"Expected results for cores {CORES}."
        )

    rows.append(load_result(filename))

data = np.vstack(rows)
data = data[np.argsort(data[:, 0])]


# ---------------------------------------------------------------------
# Extract metrics
# ---------------------------------------------------------------------
cores = data[:, 0].astype(int)
iterations = data[:, 5]
compute_time = data[:, 6]
overhead_time = data[:, 7]
comm_time = data[:, 8]
iter_per_s = data[:, 9]
mlup = data[:, 11]

# Old value saved by the solver: MLUP/s * 64 B/LUP.
old_bw_gbs = data[:, 12] / 1000.0

# New implementation-level traffic estimate.
#
# MLUP/s * B/LUP = MB/s
# divide by 1000 -> GB/s
estimated_bw_gbs = mlup * BYTES_PER_LUP / 1000.0

comm_pct = 100.0 * comm_time / compute_time
overhead_pct = 100.0 * overhead_time / (compute_time + overhead_time)

# Relative scaling within this experiment, normalized to the 9-core run.
relative_speedup = mlup / mlup[0]
ideal_relative_speedup = cores / cores[0]


# ---------------------------------------------------------------------
# Print a compact numerical summary
# ---------------------------------------------------------------------
print()
print("Single-socket memory-scaling results")
print("=" * 82)
print(
    f"{'Cores':>5} "
    f"{'MLUP/s':>10} "
    f"{'240 B/LUP GB/s':>16} "
    f"{'64 B/LUP GB/s':>15} "
    f"{'Comm %':>10} "
    f"{'Iter/s':>10}"
)
print("-" * 82)

for i in range(len(cores)):
    print(
        f"{cores[i]:5d} "
        f"{mlup[i]:10.2f} "
        f"{estimated_bw_gbs[i]:16.2f} "
        f"{old_bw_gbs[i]:15.2f} "
        f"{comm_pct[i]:10.2f} "
        f"{iter_per_s[i]:10.2f}"
    )

print("=" * 82)
print(f"Estimated traffic model: {BYTES_PER_LUP:.0f} B/LUP")
print(f"Single-socket theoretical bandwidth: {SOCKET_BW_GBS:.1f} GB/s")
print()


# ---------------------------------------------------------------------
# Figure 1: overview
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
ax.plot(cores, mlup, marker="o")
ax.set_xlabel("MPI ranks / physical cores")
ax.set_ylabel("MLUP/s")
ax.set_title("Stencil throughput")
ax.set_xticks(cores)
ax.grid(True, linestyle=":")

ax = axes[0, 1]
ax.plot(cores, estimated_bw_gbs, marker="o", label="Estimated: 240 B/LUP")
ax.plot(cores, old_bw_gbs, marker="s", label="Old model: 64 B/LUP")
ax.axhline(
    SOCKET_BW_GBS,
    linestyle="--",
    label="E5-2650 v4 theoretical socket bandwidth",
)
ax.set_xlabel("MPI ranks / physical cores")
ax.set_ylabel("Estimated memory traffic [GB/s]")
ax.set_title("Memory-bandwidth estimate")
ax.set_xticks(cores)
ax.grid(True, linestyle=":")
ax.legend()

ax = axes[1, 0]
ax.plot(cores, comm_pct, marker="o")
ax.set_xlabel("MPI ranks / physical cores")
ax.set_ylabel("Communication time [% of compute time]")
ax.set_title("Communication overhead")
ax.set_xticks(cores)
ax.grid(True, linestyle=":")

ax = axes[1, 1]
ax.plot(cores, relative_speedup, marker="o", label="Measured")
ax.plot(cores, ideal_relative_speedup, linestyle="--", label="Ideal from 9-core baseline")
ax.set_xlabel("MPI ranks / physical cores")
ax.set_ylabel("Relative speedup")
ax.set_title("Scaling from 9 to 12 cores")
ax.set_xticks(cores)
ax.grid(True, linestyle=":")
ax.legend()

fig.suptitle("Single-socket scaling: square decomposition")
fig.tight_layout()
overview_file = FIGURE_DIR / "socket_scaling_overview.png"
fig.savefig(overview_file, dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2: memory bandwidth only
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    cores,
    estimated_bw_gbs,
    marker="o",
    linewidth=2,
    label=f"Estimated implementation traffic ({BYTES_PER_LUP:.0f} B/LUP)",
)
ax.axhline(
    SOCKET_BW_GBS,
    linestyle="--",
    linewidth=2,
    label=f"Theoretical socket maximum ({SOCKET_BW_GBS:.1f} GB/s)",
)

ax.set_xlabel("MPI ranks / physical cores")
ax.set_ylabel("Estimated memory traffic [GB/s]")
ax.set_title("Single-socket estimated memory bandwidth")
ax.set_xticks(cores)
ax.grid(True, linestyle=":")
ax.legend()

fig.tight_layout()
bandwidth_file = FIGURE_DIR / "socket_memory_bandwidth.png"
fig.savefig(bandwidth_file, dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------
# Figure 3: throughput only
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(cores, mlup, marker="o", linewidth=2)
ax.set_xlabel("MPI ranks / physical cores")
ax.set_ylabel("MLUP/s")
ax.set_title("Single-socket Poisson throughput")
ax.set_xticks(cores)
ax.grid(True, linestyle=":")

fig.tight_layout()
throughput_file = FIGURE_DIR / "socket_throughput.png"
fig.savefig(throughput_file, dpi=200, bbox_inches="tight")
plt.close(fig)


print("Saved figures:")
print(f"  {overview_file}")
print(f"  {bandwidth_file}")
print(f"  {throughput_file}")
