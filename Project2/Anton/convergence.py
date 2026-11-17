from mpi4py import MPI
import argparse
import numpy as np
from matplotlib import pyplot as plt
from Anton.parallel_imp import run_Poisson_square

N_ints = np.linspace(10, 100, 10, dtype=int)

Errors = []

for n in N_ints:
    print(n)
    Errors.append(run_Poisson_square(n, max_err = 1e-10))

plt.loglog(N_ints, Errors, marker="o")
plt.loglog(N_ints, 50*(1/N_ints)**2, linestyle="--", label=r"$x^2$")
plt.xlabel("Grid size")
plt.ylabel("Error")
plt.title("Error vs. Grid Size")
plt.grid(True)
plt.legend()
plt.savefig("Anton/errors_grid_size.png", dpi=300, bbox_inches="tight")
plt.close()


