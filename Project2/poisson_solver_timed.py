import sys
import time
import numpy as np
from poisson_helper import force_term, true_sol, stencil_update

t0 = time.time()
argv = sys.argv[1:]

N_int = 50
max_it = 1_000_000
max_err = 1e-14
if argv:
    N_int = int(argv.pop(0))
if argv:
    max_it = int(argv.pop(0))
if argv:
    max_err = float(argv.pop(0))

print(f"{N_int=}, {max_it=}, {max_err=}")
N_bound = 2
N_tot = N_int + N_bound

x, dx = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)
y, dy = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)
z, dz = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)

X, Y, Z = np.meshgrid(x, y, z)

F = force_term(X, Y, Z)
u = true_sol(X, Y, Z)
phi = np.zeros(shape=(N_tot, N_tot, N_tot))

ERR = np.inf
it = 0
phi_prev = phi.copy()

t1 = time.time()
while (ERR > max_err) and (it < max_it):
    phi[1:-1, 1:-1, 1:-1] = stencil_update(phi_prev, F, dx)
    ERR = np.max(np.abs(phi - phi_prev))
    it += 1
    phi_prev, phi = phi, phi_prev

t2 = time.time()

overhead_time = t1 - t0
compute_time = t2 - t1
comm_time = 0.0
iter_per_s = it / compute_time
grid_kb = (N_tot**3 * 8) / 1e3
Mlup = N_int**3 * it / (compute_time * 1e6)
MBs = Mlup * 64
MFlops = N_int**3 * it * 9 / (compute_time * 1e6)
true_error = np.max(np.abs(phi_prev - u))

header = (
    "cores,N_int,dx,true_error,steady_error,it,"
    "compute_time,overhead_time,comm_time,iter_per_s,"
    "grid_kb,mlup,memband_mbs,mflops"
)

row = np.column_stack((
    1, N_int, dx, true_error, ERR, it,
    compute_time, overhead_time, comm_time, iter_per_s,
    grid_kb, Mlup, MBs, MFlops
))

np.savetxt(
    "Mads/Results/seq_scale.csv",
    row,
    delimiter=",",
    header=header,
    comments="",
    fmt=[
        "%d", "%d", "%.6f", "%.6e", "%.6e", "%d",
        "%.5f", "%.5f", "%.5f", "%.5f",
        "%.1f", "%.2f", "%.2f", "%.2f"
    ],
)

print(f"True final error: {true_error}")
print(f"Steady-state final error: {ERR} obtained at {it=}")
print(f"The program ran in: {compute_time:.5f}s with {iter_per_s:.5f} iterations per s")
print(f"Performance: {Mlup:.2f} Mlup/s")
print(f"Measured memory bandwidth: {MBs:.2f} MB/s")
print(f"Compute rate: {MFlops:.2f} MFLOP/s")