import csv
import re
from pathlib import Path

RESULTS_DIR = Path("Anton/Results")
OUTPUT_FILE = RESULTS_DIR / "par_scale_square.csv"

HEADER = [
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


def combine_results():
    # Match only per-core square result files, e.g.
    # par_scale_square_4.csv, par_scale_square_8.csv, ...
    source_files = sorted(
        (
            f for f in RESULTS_DIR.glob("par_scale_square_*.csv")
            if re.fullmatch(r"par_scale_square_\d+\.csv", f.name)
        ),
        key=lambda f: int(re.search(r"(\d+)", f.stem).group(1)),
    )

    if not source_files:
        print(f"No matching files found in {RESULTS_DIR}")
        return

    rows = []

    for path in source_files:
        with path.open("r", newline="") as f:
            for row in csv.reader(f):
                cleaned = [val.strip() for val in row]

                if any(cleaned):
                    rows.append(cleaned)

    # Sort numerically by core count in the first column
    rows.sort(key=lambda r: int(r[0]))

    # Write combined CSV first
    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    # Delete source files only after the combined file has been written
    for path in source_files:
        path.unlink()

    print(
        f"Merged {len(source_files)} files into "
        f"{OUTPUT_FILE} and removed the originals."
    )


if __name__ == "__main__":
    combine_results()
