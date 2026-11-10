from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Update this path if your .out files are in a subfolder like "Output/RR"
folder = Path("Output/Block")
figures_folder = Path("figures/Block")
figures_folder.mkdir(parents=True, exist_ok=True)


def get(pattern, text, dtype=float):
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Pattern not found: {pattern}")
    return dtype(match.group(1))


# -------------------------
# Load data
# -------------------------

results = []

for file in folder.glob("*.out"):
    text = file.read_text()

    # Extract worker calculation times
    calc_matches = re.findall(
        r"Rank\s+(\d+)\s+spent\s+([\d.eE+-]+)\s+s\s+calculating", text
    )

    # Skip incomplete or non-matching .out files
    if not calc_matches:
        continue

    mpi_ranks = get(r"MPI ranks:\s+(\d+)", text, int)
    wall_time = get(r"Time spent running the program\s+([\d.eE+-]+)\s+s", text)
    rank0_time = get(r"Rank 0 spent\s+([\d.eE+-]+)\s+s", text)

    # Build calc array for all ranks (0 .. mpi_ranks-1)
    calc_dict = {int(r): float(t) for r, t in calc_matches}
    if 0 not in calc_dict:
        calc_dict[0] = rank0_time
    calc_array = np.array([calc_dict.get(r, 0.0) for r in range(mpi_ranks)])

    # Extract worker send times (matches e.g. "Rank X spent Y s with sending" or "sending")
    send_matches = re.findall(
        r"Rank\s+(\d+)\s+spent\s+([\d.eE+-]+)\s+s\s+(?:with\s+)?sending", text
    )
    send_dict = {int(r): float(t) for r, t in send_matches}
    send_dict[0] = 0.0  # Rank 0 has send = 0.0
    send_array = np.array([send_dict.get(r, 0.0) for r in range(mpi_ranks)])

    calc_times_only = [float(t) for _, t in calc_matches]
    comm_time = wall_time - rank0_time
    comm_share = comm_time / wall_time

    results.append({
        "mpi_ranks": mpi_ranks,
        "image_size": get(r"Image size:\s+(\d+)x\d+", text, int),
        "wall_time": wall_time,
        "max_work_time": max(calc_times_only),
        "min_work_time": min(calc_times_only),
        "max_comm_time": comm_time,
        "max_comm_share": comm_share,
        "calc_times": calc_array,
        "send_times": send_array,
    })

df = pd.DataFrame(results).sort_values(["mpi_ranks", "image_size"])

subset = df.loc[(df["mpi_ranks"] == 12) & (df["image_size"] == 10000), "wall_time"]
if not subset.empty:
    print("Wall time for rank 12, image size 10000:", subset.iloc[0])

# Full MPI ranks are working ranks
df["working_ranks"] = df["mpi_ranks"]

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
# E_p = (6 * T_6) / (p * T_p)
baseline_time = (
    df.loc[df["mpi_ranks"] == 6]
    .set_index("image_size")["wall_time"]
)
df["parallelization_efficiency"] = (
    df["image_size"].map(baseline_time) * 6
    / (df["mpi_ranks"] * df["wall_time"])
)


# -------------------------
# Wall time
# -------------------------

for mpi_ranks, group in df.groupby("mpi_ranks"):
    if mpi_ranks not in {6, 12, 18, 24, 48}:
        continue

    plt.plot(
        group["image_size"],
        group["wall_time"],
        marker="o",
        label=f"{mpi_ranks} MPI ranks",
    )

plt.xlabel("Image width")
plt.ylabel("Wall time [s]")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_folder / "wall_time_vs_image_size.png", dpi=300)
plt.close()


# -------------------------
# Strong scaling speedup
# -------------------------

wall_time = df.pivot(
    index="mpi_ranks",
    columns="image_size",
    values="wall_time",
)

baseline_ranks = wall_time.index.min()
speedup = wall_time.loc[baseline_ranks] / wall_time
available_sizes = [s for s in [10000, 15000, 20000] if s in speedup.columns]
speedup = speedup[available_sizes]

for image_size in speedup.columns:
    plt.plot(
        speedup.index,
        speedup[image_size],
        marker="o",
        label=f"{image_size} × {image_size}",
    )

ideal_speedup = speedup.index / baseline_ranks

plt.plot(
    speedup.index,
    ideal_speedup,
    "k--",
    label="Ideal scaling",
)

plt.xlabel("MPI ranks")
plt.ylabel("Speedup")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_folder / "strong_scaling_speedup.png", dpi=300)
plt.close()


# -------------------------
# Parallelization efficiency
# -------------------------

for image_size, group in df[df["image_size"].isin([14000, 15000, 20000])].groupby("image_size"):
    plt.plot(
        group["mpi_ranks"],
        group["parallelization_efficiency"],
        marker="o",
        label=f"{image_size} × {image_size}",
    )

plt.xlabel("MPI ranks")
plt.ylabel("Parallelization efficiency")
plt.ylim(0.8, 1.5)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_folder / "parallelization_efficiency.png", dpi=300)
plt.close()


# -------------------------
# Average parallelization efficiency by rank
# -------------------------

average_efficiency = (
    df.groupby("mpi_ranks")["parallelization_efficiency"]
    .mean()
)

plt.plot(
    average_efficiency.index,
    average_efficiency,
    marker="o",
)
plt.xlabel("Working ranks")
plt.ylabel("Average parallelization efficiency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    figures_folder / "average_parallelization_efficiency.png",
    dpi=300,
)
plt.close()


# -------------------------
# Heatmap helper
# -------------------------

def heatmap(
    column,
    colorbar_label,
    filename=None,
    show_values=False,
    ax=None,
    x_column="image_size",
    x_label="Image width",
):
    data = df.pivot(
        index="working_ranks",
        columns=x_column,
        values=column,
    )

    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots()

    image = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
    )

    if show_values:
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                value = data.iloc[row, col]
                if pd.notna(value):
                    ax.text(
                        col,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

    ax.set_xticks(
        range(len(data.columns)),
        [
            f"{value:,.0f}" if x_column == "pixel_count" else value
            for value in data.columns
        ],
        rotation=45,
    )

    ax.set_yticks(
        range(len(data.index)),
        data.index,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Working ranks")
    ax.figure.colorbar(image, ax=ax, label=colorbar_label)

    if own_figure:
        ax.figure.tight_layout()
        ax.figure.savefig(figures_folder / filename, dpi=300)
        plt.close(ax.figure)


# -------------------------
# Heatmaps
# -------------------------

heatmap(
    "max_comm_share",
    "Maximum communication share",
    "maximum_communication_share.png",
)

heatmap(
    "relative_work_imbalance",
    "Relative work imbalance",
    "relative_work_imbalance.png",
)

heatmap(
    "time_increase_factor",
    "Wall-time increase factor",
    "time_increase_factor_vs_image_size.png",
    show_values=True,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

heatmap(
    "pixels_per_second_per_worker",
    "FLOPS / worker",
    show_values=False,
    ax=axes[0],
)

heatmap(
    "parallelization_efficiency",
    "Parallelization efficiency",
    show_values=True,
    ax=axes[1],
)

fig.tight_layout()
fig.savefig(figures_folder / "performance_heatmaps.png", dpi=300)
plt.close(fig)


# -------------------------
# Work time & Send overhead per core (Ranks 6, 24, 42, 48)
# -------------------------

core_counts = [6, 24, 42, 48]
target_image_size = 10000

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
axes = axes.flatten()

for i, cores in enumerate(core_counts):
    ax = axes[i]
    row = df[
        (df["mpi_ranks"] == cores)
        & (df["image_size"] == target_image_size)
    ]

    if row.empty:
        ax.set_title(f"{cores} MPI Ranks (No data)", fontsize=10)
        continue

    calc = row.iloc[0]["calc_times"]
    send = row.iloc[0]["send_times"]

    # Calculate delta overhead (send timestamp - calc timestamp)
    # Rank 0 has send = 0.0, so overhead is set to 0.0
    send_overhead = np.maximum(0.0, send - calc)
    send_overhead[0] = 0.0

    ranks = np.arange(cores)

    # Stacked bars: Calculation at the bottom, Send overhead on top
    ax.bar(
        ranks,
        calc,
        label="Computation Time ($T_{calc}$)",
        color="#2b5c8f",
        width=0.75,
        alpha=0.9,
    )
    ax.bar(
        ranks,
        send_overhead,
        bottom=calc,
        label="Send Overhead ($T_{send} - T_{calc}$)",
        color="#e26d5c",
        width=0.75,
        alpha=0.9,
    )

    # ax.set_title(f"{cores} MPI Ranks", fontsize=10)
    ax.set_xlabel("MPI Rank", fontsize=9)
    ax.set_ylabel("Time [s]", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    # ax.legend(fontsize=8)

    # Adjust tick intervals based on rank count
    if cores > 24:
        ax.set_xticks(np.arange(0, cores, 4))
    else:
        ax.set_xticks(np.arange(0, cores, 2))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=11, frameon=True)
# fig.tight_layout()
fig.savefig(figures_folder / f"work_time_per_rank_{target_image_size}.png", dpi=300, bbox_inches='tight')
plt.close(fig)