# This is where we do the splitting strategies.
import numpy as np
from poisson_helper import force_term, stencil_update, true_sol
from mpi4py import MPI


def split_dimension(N, P, coord):
    q, r = divmod(N, P)

    n_local = q + (coord < r)
    start = coord * q + min(coord, r)

    return start, n_local


def send_recive(phi_prev, neighbors, send_idx, recv_idx, cart):
    requests = []
    send_buffers = [None] * len(neighbors)
    recv_buffers = [None] * len(neighbors)
    for i, neighbor in enumerate(neighbors):

        if neighbor == MPI.PROC_NULL:
            continue

        recv_buffers[i] = np.empty_like(phi_prev[send_idx[i]])

        requests.append(
            cart.Irecv(
                recv_buffers[i],
                source=neighbor,
                tag=i // 2
            )
        )

        sendbuf = phi_prev[send_idx[i]].copy()
        send_buffers[i] = sendbuf

        requests.append(
            cart.Isend(
                sendbuf,
                dest=neighbor,
                tag=i // 2
            )
        ) 

    MPI.Request.Waitall(requests)

    for i, neighbor in enumerate(neighbors):
        if neighbor == MPI.PROC_NULL:
            continue

        phi_prev[recv_idx[i]] = recv_buffers[i]
    
    return phi_prev


def optimal_dims(cores, shape):
    Nx, Ny, Nz = shape

    best_dims = None
    best_area = np.inf

    for Px in range(1, cores + 1):
        if cores % Px != 0:
            continue

        remaining = cores // Px

        for Py in range(1, remaining + 1):
            if remaining % Py != 0:
                continue

            Pz = remaining // Py

            area = (
                (Px - 1) * Ny * Nz
                + (Py - 1) * Nx * Nz
                + (Pz - 1) * Nx * Ny
            )

            if area < best_area:
                best_area = area
                best_dims = (Px, Py, Pz)

    return best_dims