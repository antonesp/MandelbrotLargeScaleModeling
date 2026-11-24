import sys
import time
import numpy as np
from numba import jit, stencil
from mpi4py import MPI
from poisson_helper import force_term, stencil_update, true_sol, stencil_update_better

# Since it loops through axis backward to step any row, we have to advance the z- and y-axis
def send_col(col_idx):
    col = MPI.DOUBLE.Create_subarray(
        sizes=(Nx, Ny, Nz),
        subsizes=(1, Ny, Nz),
        starts=(col_idx, 0, 0)
    )
    col.Commit()
    return col

def send_row(row_idx):
    row = MPI.DOUBLE.Create_subarray(
        sizes=(Nx, Ny, Nz),
        subsizes=(Nx, 1, Nz),
        starts=(0, row_idx, 0)
    )
    row.Commit()
    return row

# Setup
t1 = time.time()
comm = MPI.COMM_WORLD.Clone()
size = comm.Get_size()

argv = sys.argv[1:]

# Create cartesian communicator split in the x,y plane with full z
dim = 2
dims = MPI.Compute_dims(comm.Get_size(), dim)
cart_comm = comm.Create_cart(dims, [False, False])

# Get the placement in our subdivided grid
coords = cart_comm.Get_coords(comm.Get_rank())
rank = cart_comm.Get_cart_rank(coords)

# Define the solver params, global interior nodes and boundary nodes
N_INT = 50
max_it = 1_000_000
max_err = 1e-14
if argv:
    N_INT = int(argv.pop(0))    
if argv:
    max_it = int(argv.pop(0))    
if argv:
    max_err = float(argv.pop(0))    

if rank == 0:
    print(f"{N_INT=}, {max_it=}, {max_err=}")

N_BOUND = 2
N_TOT = N_INT + N_BOUND
(no_x, no_y), _, _ = cart_comm.topo

# Define global domain size
lower = -1                            # Lower domain limit
upper = 1                             # Upper domain limit
DX = (upper - lower) / (N_TOT - 1)    # Global step size

# Find the local division
N_intx = N_INT // no_x + (1 if coords[0] < (N_INT % no_x) else 0)
N_inty = N_INT // no_y + (1 if coords[1] < (N_INT % no_y) else 0)

Nx, Ny, Nz = N_intx + N_BOUND, N_inty + N_BOUND, N_TOT

# Define our domain
DOMAIN = np.linspace(-1, 1, num=N_TOT, endpoint=True)   # Defines the x, y, z 

x_lwr = coords[0] * (N_INT // no_x) + min(coords[0], N_INT % no_x)
x_upp = min(x_lwr + N_intx, N_INT) + N_BOUND
y_lwr = coords[1] * (N_INT // no_y) + min(coords[1], N_INT % no_y)
y_upp = min(y_lwr + N_inty, N_INT) + N_BOUND

domain_x = DOMAIN[x_lwr:x_upp]
domain_y = DOMAIN[y_lwr:y_upp]
domain_z = DOMAIN

X, Y, Z = np.meshgrid(domain_x, domain_y, domain_z, indexing='ij')

F = force_term(X, Y, Z)
phi = np.zeros(shape=(N_intx+N_BOUND, N_inty+N_BOUND, N_TOT))

# Find our local ranks neighbors
xm, xp = cart_comm.Shift(0, 1)
ym, yp = cart_comm.Shift(1, 1)

neighbor_place = [0, 2, 1, 3]
neighbors = [xm, xp, ym, yp]
neighbor_place_filt = []
neighbors_filt = []
for npl, n in zip(neighbor_place, neighbors):
    if n != MPI.PROC_NULL:
        neighbor_place_filt.append(npl)
        neighbors_filt.append(n)


def make_subarray(shape, subsizes, starts):
    sub = MPI.DOUBLE.Create_subarray(sizes=shape, subsizes=subsizes, starts=starts)
    sub.Commit()
    return sub

shape = (Nx, Ny, Nz)
halo_specs = [
    (xm, (1, Ny, Nz), (1, 0, 0),      (0, 0, 0)),       # left
    (xp, (1, Ny, Nz), (Nx - 2, 0, 0), (Nx - 1, 0, 0)),  # right
    (ym, (Nx, 1, Nz), (0, 1, 0),      (0, 0, 0)),       # top
    (yp, (Nx, 1, Nz), (0, Ny - 2, 0), (0, Ny - 1, 0)),  # bottom
]

halos = [
    (nbr, make_subarray(shape, sub, s_start), make_subarray(shape, sub, r_start))
    for nbr, sub, s_start, r_start in halo_specs
    if nbr != MPI.PROC_NULL
]

# Define initial errors and loop
local_error = np.array([np.inf], dtype=np.float64)
global_error = np.array([np.inf], dtype=np.float64)
it = 0
phi_prev = phi.copy()
band_time = 0
comm_time = 0

# @stencil
@jit(parallel=True, nopython=True)
def jit_stencil(phi, F, DX):
    phi_new = (
        phi[:-2, 1:-1, 1:-1] + phi[2:, 1:-1, 1:-1] +
        phi[1:-1, :-2, 1:-1] + phi[1:-1, 2:, 1:-1] +
        phi[1:-1, 1:-1, :-2] + phi[1:-1, 1:-1, 2:] +
        DX**2 * F[1:-1, 1:-1, 1:-1]
    ) / 6
    err = np.max(np.abs(phi_new - phi[1:-1, 1:-1, 1:-1]))
    return phi_new, err

t2 = time.time()
while (it < max_it) and (global_error[0] > max_err):
    reqs = []

    t1_comm = time.time()
    for nbr, send_t, recv_t in halos:
        reqs.append(cart_comm.Isend([phi_prev, 1, send_t], dest=nbr))
        reqs.append(cart_comm.Irecv([phi_prev, 1, recv_t], source=nbr))
    MPI.Request.Waitall(reqs)
    t2_comm = time.time()
    comm_time += t2_comm - t1_comm

    t1_band = time.time()
    phi[1:-1, 1:-1, 1:-1], local_error[0] = jit_stencil(phi_prev, F, DX)
    # local_error[0] = np.max(np.abs(phi - phi_prev))
    t2_band = time.time()
    band_time += t2_band - t1_band
    
    t3_comm = time.time()
    comm.Allreduce(sendbuf=local_error, recvbuf=global_error, op=MPI.MAX)
    comm_time += time.time() - t3_comm
    it += 1
    phi_prev, phi = phi, phi_prev
t3 = time.time()

# Final error tracking
global_true_error = np.array([np.inf], dtype=np.float64)
local_true_error = np.max(np.abs(phi_prev - true_sol(X, Y, Z)))

t4_comm = time.time()
comm.Allreduce(sendbuf=local_true_error, recvbuf=global_true_error, op=MPI.MAX)
comm_time += time.time() - t4_comm

if rank == 0:
    time_spent = t3 - t2
    Mlup = N_INT**3 * it / (band_time * 1e6)
    MBs = Mlup * 80
    MFlops = N_INT**3 * it * 9 / (time_spent * 1e6)

    overhead = t2 - t1
    compute_time = t3 - t2
    total_time = t3 - t1
    overhead_pct = 100 * overhead / total_time
    comm_pct = 100 * comm_time / compute_time
    iter_per_sec = it / compute_time
    grid_kb = (N_TOT**3 * 8) / 1e3

    print(
        f"\n{' Simulation Summary ':=^56}\n"
        f"Timing\n"
        f"  {'Compute time:':<24} {compute_time:>10.5f} s  ({iter_per_sec:,.2f} it/s)\n"
        f"  {'Overhead time:':<24} {overhead:>10.5f} s  ({overhead_pct:.2f}% of total)\n"
        f"  {'Communication time:':<24} {comm_time:>10.5f} s  ({comm_pct:.2f}% of compute time)\n"
        f"\n"
        f"Convergence (it = {it:,})\n"
        f"  {'Steady-state error:':<24} {global_error[0]:>10.3e}\n"
        f"  {'True final error:':<24} {global_true_error[0]:>10.3e}\n"
        f"\n"
        f"Performance (Grid: {grid_kb:,.1f} KB)\n"
        f"  {'Throughput:':<24} {Mlup:>10.2f} MLUP/s\n"
        f"  {'Memory bandwidth:':<24} {MBs:>10.2f} MB/s\n"
        f"  {'Compute rate:':<24} {MFlops:>10.2f} MFLOP/s\n"
        f"{'=' * 56}\n"
    )

# Clean-up for all the messages
for _, send_t, recv_t in halos:
    send_t.Free()
    recv_t.Free()