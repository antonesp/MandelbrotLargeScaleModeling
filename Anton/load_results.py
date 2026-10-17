from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

folder = Path("outputs/scale_compute")
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

wall_times = df[df["image_size"].isin([10000, 20000])].pivot(
    index="working_ranks",
    columns="image_size",
    values="wall_time",
)
mean_wall_time_increase = (wall_times[20000] / wall_times[10000]).mean()
print(
    "Mean wall-time increase factor at image width 20000 compared to 10000: "
    f"{mean_wall_time_increase:.2f}"
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


# -------------------------
# Wall time
# -------------------------

for mpi_ranks, group in df.groupby("mpi_ranks"):
    if mpi_ranks not in {6, 12, 24, 48}:
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
speedup = speedup[[10000, 15000, 20000]]

for image_size in speedup.columns:
    plt.plot(
        speedup.index,
        speedup[image_size],
        marker="o",
        label=f"{image_size} × {image_size}",
    )

ideal_speedup = (
    (speedup.index - 1)
    / (baseline_ranks - 1)
)

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
        group["mpi_ranks"]-1,
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

plt.scatter(
    average_efficiency.index - 1,
    average_efficiency,
)
plt.axhline(1, color="black", linestyle="--", label="Perfect parallelization")
plt.xlabel("Working ranks")
plt.ylabel("Average parallelization efficiency")
plt.grid(alpha=0.3)
plt.ylim(0.6, 1.2)
plt.legend()
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
            for column in range(data.shape[1]):
                value = data.iloc[row, column]
                if pd.notna(value):
                    ax.text(
                        column,
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
    "Maximum work imbalance",
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



