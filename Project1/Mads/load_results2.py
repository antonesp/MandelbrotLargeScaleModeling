from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

# Update this path if your .out files are in a subfolder like "Output/NonBlock"
folder = Path("Output/NonBlock")
figures_folder = Path("figures/NonBlock")
figures_folder.mkdir(exist_ok=True)


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

    # Extract rank ID and time for all ranks
    work_matches = re.findall(
        r"Rank\s+(\d+)\s+spent\s+([\d.eE+-]+)\s+s\s+(?:calculating|with sending)",
        text,
    )

    if not work_matches:
        continue

    # Sort by rank index (0, 1, ..., p-1)
    work_matches_sorted = sorted(work_matches, key=lambda x: int(x[0]))
    work_times = [float(t) for _, t in work_matches_sorted]
    wall_time = get(r"Time spent running the program\s+([\d.eE+-]+)\s+s", text)

    results.append({
        "mpi_ranks": get(r"MPI ranks:\s+(\d+)", text, int),
        "image_size": get(r"Image size:\s+(\d+)x\d+", text, int),
        "wall_time": wall_time,
        "max_work_time": max(work_times),
        "min_work_time": min(work_times),
        "work_times": work_times,
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
plt.ylim(0.6, 1.2)
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
# Heatmaps (No communication plot)
# -------------------------

heatmap(
    "relative_work_imbalance",
    "Maximum work imbalance",
    "maximum_work_imbalance.png",
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
# Work time per core (Ranks 6, 24, 42, 48)
# -------------------------
import numpy as np
core_counts = [6, 24, 42, 48]
# Select target image size (defaults to largest available, e.g. 20000)
# target_image_size = df["image_size"].min()
target_image_size = 17000


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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

    calc = row.iloc[0]["work_times"]
    ranks = np.arange(cores)

    ax.bar(
        ranks,
        calc,
        label="Work Time ($T_{work}$)",
        color="#2b5c8f",
        width=0.75,
        alpha=0.9,
    )

    ax.set_title(f"{cores} MPI Ranks", fontsize=10)
    ax.set_xlabel("MPI Rank", fontsize=9)
    ax.set_ylabel("Time [s]", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    # ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(figures_folder / f"work_time_per_rank_{target_image_size}.png", dpi=300)
plt.close(fig)