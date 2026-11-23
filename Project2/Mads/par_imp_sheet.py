import sys
import time
import numpy as np
from mpi4py import MPI
from poisson_helper import force_term, stencil_update, true_sol

t1 = time.time()
comm = MPI.COMM_WORLD.Clone()
size = comm.Get_size()
rank = comm.Get_rank()

argv = sys.argv[1:]

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

N_BOUND = 2
N_TOT = N_INT + N_BOUND

# Define global domain size
lower = -1                            # Lower domain limit
upper = 1                             # Upper domain limit
DX = (upper - lower) / (N_TOT - 1)    # Global step size

# Find the local division
N_intz = N_INT // size + (1 if rank < (N_INT % size) else 0)

Nx, Ny, Nz = N_TOT, N_TOT, N_intz + N_BOUND

# Define our domain
DOMAIN = np.linspace(-1, 1, num=N_TOT, endpoint=True)   # Defines the x, y, z 

z_lwr = rank * (N_INT // size) + min(rank, N_INT % size)
z_upp = min(z_lwr + N_intz, N_INT) + N_BOUND

domain_x = DOMAIN
domain_y = DOMAIN
domain_z = DOMAIN[z_lwr:z_upp]

X, Y, Z = np.meshgrid(domain_x, domain_y, domain_z, indexing='ij')

F = force_term(X, Y, Z)
phi = np.zeros(shape=(N_TOT, N_TOT, N_intz+N_BOUND))

# print(f"{rank=} has solution space of shape {phi.shape}")

# Find our local ranks neighbors
neighbors = []
if rank == 0:
    neighbors.append(rank + 1)
elif rank == size - 1:
    neighbors.append(rank - 1)
else:
    neighbors.append(rank-1)
    neighbors.append(rank+1)

# Since it loops through axis backward to step any row, we have to advance the z- and y-axis
def send_face(face_idx):
    face = MPI.DOUBLE.Create_subarray(
        sizes=(Nx, Ny, Nz),
        subsizes=(Nx, Ny, 1),
        starts=(0, 0, face_idx)
    )
    face.Commit()
    return face

# Create Send and Receive types for X faces
if rank == 0:
    msg_send_back = send_face(Nz-2)
    msg_recv_back = send_face(Nz-1)
elif rank == size-1:
    msg_send_front = send_face(1)
    msg_recv_front = send_face(0)
else:
    msg_send_front = send_face(1)
    msg_recv_front = send_face(0)
    msg_send_back = send_face(Nz-2)
    msg_recv_back = send_face(Nz-1)


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
    if rank == 0:
        reqs.append(comm.Isend([phi_prev, 1, msg_send_back], dest=rank+1))
        reqs.append(comm.Irecv([phi_prev, 1, msg_recv_back], source=rank+1))
    elif rank == size-1:
        reqs.append(comm.Isend([phi_prev, 1, msg_send_front], dest=rank-1))
        reqs.append(comm.Irecv([phi_prev, 1, msg_recv_front], source=rank-1))
    else:
        reqs.append(comm.Isend([phi_prev, 1, msg_send_back], dest=rank+1))
        reqs.append(comm.Irecv([phi_prev, 1, msg_recv_back], source=rank+1))
        reqs.append(comm.Isend([phi_prev, 1, msg_send_front], dest=rank-1))
        reqs.append(comm.Irecv([phi_prev, 1, msg_recv_front], source=rank-1))

    MPI.Request.Waitall(reqs)
    t2_comm = time.time()
    comm_time += t2_comm - t1_comm


    t1_band = time.time()
    phi[1:-1, 1:-1, 1:-1] = stencil_update(phi_prev, F, DX)
    t2_band = time.time()
    band_time += t2_band - t1_band

    local_error[0] = np.max(np.abs(phi - phi_prev))
    comm.Allreduce(sendbuf=local_error, recvbuf=global_error, op=MPI.MAX)
    it += 1
    phi_prev, phi = phi, phi_prev
t3 = time.time()

# Final error tracking
global_true_error = np.array([np.inf], dtype=np.float64)
local_true_error = np.max(np.abs(phi_prev - true_sol(X, Y, Z)))
    
comm.Allreduce(sendbuf=local_true_error, recvbuf=global_true_error, op=MPI.MAX)

if rank == 0:
    time_spent = t3 - t2
    Mlup = N_INT**3 * it / (time_spent * 1e6)
    MBs = Mlup * 64
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
if rank == 0:
    msg_send_back.Free()
    msg_recv_back.Free()
elif rank == size - 1:
    msg_send_front.Free()
    msg_recv_front.Free()
else:
    msg_send_back.Free()
    msg_recv_back.Free()
    msg_send_front.Free()
    msg_recv_front.Free()