import csv
import re
from pathlib import Path

RESULTS_DIR = Path("Mads/Results/scaling")
OUTPUT_FILE = RESULTS_DIR / "par1D.csv"
# OUTPUT_FILE = RESULTS_DIR / "par_scale_sausage.csv"
# OUTPUT_FILE = RESULTS_DIR / "par_scale_sheet.csv"

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
    "total_time",
    "iter_per_s",
    "grid_kb",
    "mlup",
    "memband_mbs",
    "mflops",
    "max_mem_mb",
    "comm_pts"
]


def combine_results():
    # Match only per-core files (e.g., par_scale_4.csv) and ignore the combined file
    source_files = sorted(
        (
            f for f in RESULTS_DIR.glob("par1D_*.csv")
            if re.fullmatch(r"par1D_\d+\.csv", f.name)
            # f for f in RESULTS_DIR.glob("par_scale_sheet_*.csv")
            # if re.fullmatch(r"par_scale_sheet_\d+\.csv", f.name)
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
                if any(cleaned):  # Skip blank lines
                    rows.append(cleaned)

    # Sort numerically by core count (first column)
    rows.sort(key=lambda r: int(r[0]))

    # Write combined CSV first
    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    # Only delete source files after the combined file is safely written
    for path in source_files:
        path.unlink()

    print(f"Merged {len(source_files)} files into {OUTPUT_FILE} and removed the originals.")


if __name__ == "__main__":
    combine_results()