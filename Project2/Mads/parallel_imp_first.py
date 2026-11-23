import sys
import time
import numpy as np
from mpi4py import MPI
from poisson_helper import force_term, stencil_update, true_sol

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


# Create messages
msg_size = 0
for npl, n in zip(neighbor_place_filt, neighbors_filt):
    if npl == 0:
        msg_recv_left  = send_col(0)      # Ghost cell
        msg_send_left  = send_col(1)      # Interior cell
        msg_size += 8 * Ny * Nz
    elif npl == 1:
        msg_recv_top   = send_row(0)      # Ghost cell
        msg_send_top   = send_row(1)      # Interior cell
        msg_size += 8 * Nx * Nz
    elif npl == 2:
        msg_send_right = send_col(Nx - 2) # Interior cell
        msg_recv_right = send_col(Nx - 1) # Ghost cell
        msg_size += 8 * Ny * Nz
    elif npl == 3:
        msg_send_bot   = send_row(Ny - 2) # Interior cell
        msg_recv_bot   = send_row(Ny - 1) # Ghost cell
        msg_size += 8 * Nx * Nz


# Define initial errors and loop
local_error = np.array([np.inf], dtype=np.float64)
global_error = np.array([np.inf], dtype=np.float64)
it = 0
phi_prev = phi.copy()
band_time = 0
comm_time = 0

t2 = time.time()
while (it < max_it) and (global_error[0] > max_err):
    reqs = []
    t1_comm = time.time()
    for npl, n in zip(neighbor_place_filt, neighbors_filt):
        if npl == 0:
            reqs.append(comm.Isend([phi_prev, 1, msg_send_left], dest=n))
            reqs.append(comm.Irecv([phi_prev, 1, msg_recv_left], source=n))
        elif npl == 1:
            reqs.append(comm.Isend([phi_prev, 1, msg_send_top], dest=n))
            reqs.append(comm.Irecv([phi_prev, 1, msg_recv_top], source=n))
        elif npl == 2:
            reqs.append(comm.Isend([phi_prev, 1, msg_send_right], dest=n))
            reqs.append(comm.Irecv([phi_prev, 1, msg_recv_right], source=n))
        elif npl == 3:
            reqs.append(comm.Isend([phi_prev, 1, msg_send_bot], dest=n))
            reqs.append(comm.Irecv([phi_prev, 1, msg_recv_bot], source=n))
    MPI.Request.Waitall(reqs)

    t2_comm = time.time()
    comm_time += t2_comm - t1_comm

    t1_band = time.time()
    phi[1:-1, 1:-1, 1:-1] = stencil_update(phi_prev, F, DX)
    t2_band = time.time()
    band_time += t2_band - t1_band

    if it % 50 == 0:
        local_error[0] = np.max(np.abs(phi - phi_prev))
        t3_comm = time.time()
        comm.Allreduce(sendbuf=local_error, recvbuf=global_error, op=MPI.MAX)
        comm_time += time.time() - t3_comm
    it += 1
    phi_prev, phi = phi, phi_prev
t3 = time.time()

# Measuring Mlup/s
if rank == 0: 
    time_spent = t3 - t2
    Mlup = N_INT**3 * it / (time_spent * 1e6)
    MBs = Mlup * 64
    MFlops = N_INT**3 * it * 9 / (time_spent * 1e6)

# Final error tracking
global_true_error = np.array([np.inf], dtype=np.float64)
local_true_error = np.max(np.abs(phi_prev - true_sol(X, Y, Z)))

t4_comm = time.time()
comm.Allreduce(sendbuf=local_true_error, recvbuf=global_true_error, op=MPI.MAX)
comm_time += time.time() - t4_comm
# if rank==0:
#     print(f"Time spent on overhead {t2 - t1:.5f} s using up {100*(t2-t1)/(t3-t1):.2f}% of the program")
#     print(f"The program ran in: {t3 - t2}s with {it/(t3 - t2):.5f} iterations per s")
#     print(f"Time spent communicating {comm_time:.5f}s")
#     print()
#     print(f"Steady-state final error: {global_error[0]} obtained at {it=}")
#     print(f"True final error: {global_true_error[0]}")
#     print()
#     print(f"Performance: {Mlup:.2f} Mlup/s")
#     print(f"Measured memory bandwidth: {MBs:.2f} MB/s")
#     print(f"Performance: {MFlops:.2f} MFLOPS for a grid of size {(N_TOT**3 * 8) / 1e3} KB")

if rank == 0:
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
# for npl, n in zip(neighbor_place_filt, neighbors_filt):
#     if npl == 0:
#         msg_recv_left.Free() 
#         msg_send_left.Free() 
#     elif npl == 1:
#         msg_recv_top.Free()
#         msg_send_top.Free()
#     elif npl == 2:
#         msg_send_right.Free()
#         msg_recv_right.Free()
#     elif npl == 3:
#         msg_send_bot.Free()
#         msg_recv_bot.Free()
for _, send_t, recv_t in halos:
    send_t.Free()
    recv_t.Free()