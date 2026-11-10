from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

folder = Path("outputs/chunk_size")
figures_folder = Path("figures")
figures_folder.mkdir(exist_ok=True)


def get(pattern, text, dtype=float):
    return dtype(re.search(pattern, text).group(1))


# -------------------------
# Load data
# -------------------------

results = []

for file in folder.glob("*.out"):
    text = file.read_text()

    results.append({
        "chunk_size": get(r"Chunk size:\s+(\d+)", text, int),
        "mpi_ranks": get(r"MPI ranks:\s+(\d+)", text, int),
        "image_size": get(r"Image size:\s+(\d+)x\d+", text, int),
        "wall_time": get(r"Wall time\s+([\d.eE+-]+)", text),
        "max_work_time": get(r"Max Work time\s+([\d.eE+-]+)", text),
        "min_work_time": get(r"Min Work time\s+([\d.eE+-]+)", text),
        "max_comm_share": get(r"Maximum comm share\s+([\d.eE+-]+)", text),
        "max_comm_time": get(r"Maximum comm time\s+([\d.eE+-]+)", text),
    })

df = pd.DataFrame(results).sort_values(["mpi_ranks", "image_size"])

# print(df[["mpi_ranks", "image_size", "wall_time"]].to_string(index=False))

# Rank 0 only distributes work
df["working_ranks"] = df["mpi_ranks"] - 1

df["relative_work_imbalance"] = (
    (df["max_work_time"] - df["min_work_time"])
    / df["max_work_time"]
)

df["pixels_per_second_per_worker"] = (
    df["image_size"]**2 * 241.284
    / (df["wall_time"] * df["working_ranks"])
)

# Wall-time increase relative to the smallest image for each worker count.
df["pixel_count"] = df["image_size"]**2
df = df.sort_values(["working_ranks", "pixel_count"])
df["time_increase_factor"] = (
    df["wall_time"]
    / df.groupby("working_ranks")["wall_time"].transform("first")
)

# Parallelization efficiency relative to the 6-rank run:
# E_p = T_6 / (p * T_p)
baseline_time = (
    df.loc[df["mpi_ranks"] == 6]
    .set_index("image_size")["wall_time"]
)
df["parallelization_efficiency"] = (
    df["image_size"].map(baseline_time)*5
    / ((df["mpi_ranks"]-1) * df["wall_time"])
)

# Plot wall time and relative work imbalance side by side.
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

for (mpi_ranks, image_size), group in df.groupby(["mpi_ranks", "image_size"]):
    group = group.sort_values("chunk_size")
    label = f"{mpi_ranks} ranks, {image_size}px"
    axes[0].plot(group["chunk_size"], group["wall_time"], marker="o", label=label)
    axes[1].plot(
        group["chunk_size"],
        group["relative_work_imbalance"],
        marker="o",
        label=label,
    )

axes[0].set_title("Wall time")
axes[0].set_xlabel("Chunk size")
axes[0].set_ylabel("Wall time (s)")
axes[1].set_title("Relative work imbalance")
axes[1].set_xlabel("Chunk size")
axes[1].set_ylabel("Relative work imbalance")

for axis in axes:
    axis.grid(True, alpha=0.3)

# axes[0].set_yscale("log", base=2)
axes[0].set_xscale("log", base = 2)
axes[1].set_xscale("log", base = 2)

axes[1].legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc="upper left")
fig.savefig(figures_folder / "dynamic_chunk_size.png", dpi=300)
plt.show()


