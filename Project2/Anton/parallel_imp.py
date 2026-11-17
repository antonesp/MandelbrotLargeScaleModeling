import argparse
import numpy as np
from poisson_helper import force_term, stencil_update, true_sol
from mpi4py import MPI
from Anton.more_help_funcs import send_recive, split_dimension
from time import time
# My implimentation of a parallel poisson solver.


def run_Poisson_square(N_int, max_it = 1_000_000, max_err = 1e-6, report = False):
        
    
    comm = MPI.COMM_WORLD.Clone()

    # We start by assuming a simple square decomposition of the problem

    # Also assume rank 4 for the simple basic test.
    rank = comm.Get_rank()
    cores = comm.Get_size()
    dims = MPI.Compute_dims(cores, [0, 0, 0])
    cart = comm.Create_cart(
        dims=dims,
        periods=(False, False, False),
        reorder=False,
    )

    rank = cart.Get_rank()
    coords = cart.Get_coords(rank)


    N_bound = 2
    N_tot = N_int + N_bound


    # Split the grid up, this is the number of steps in each dimensions assinged to the rank.
    # This only works when it can be decomposed nicely.

    Px, Py, Pz = dims

    # Get the dimensions of the smaller squares.


    x, dx = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)
    y, dy = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)
    z, dz = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)
    # The interior points
    x_int = x[1:-1]
    y_int = y[1:-1]
    z_int = z[1:-1]

    # Dimensions in rank space.
    rx, ry, rz = coords

    ix0, nx = split_dimension(N_int, Px, rx)
    iy0, ny = split_dimension(N_int, Py, ry)
    iz0, nz = split_dimension(N_int, Pz, rz)

    subproblem_dimensions = cart.gather((nx, ny, nz), root=0)
    if rank == 0 and report:
        print(f"Number of cores: {cores}")
        print(f"Number of interior points: {N_int}")
        print(f"Subproblem dimensions: {subproblem_dimensions}")


    # We can then calculate the local grid
    x_local = x_int[ix0:ix0 + nx]
    y_local = y_int[iy0:iy0 + ny]
    z_local = z_int[iz0:iz0 + nz]

    X, Y, Z = np.meshgrid(x_local, y_local, z_local, indexing = "ij")

    phi= np.zeros((nx + 2, ny + 2, nz + 2))

    F = force_term(X, Y, Z)
    F = np.pad(
        F,
        pad_width=1,
        mode="constant",
        constant_values=0
    )
    u = true_sol(X, Y, Z)

    xm, xp = cart.Shift(0, 1)  # x-direction
    ym, yp = cart.Shift(1, 1)  # y-direction
    zm, zp = cart.Shift(2, 1)  # z-direction


    neighbors = [xm, xp, ym, yp, zm, zp]

    send_idx = [
        (1,              slice(1, -1), slice(1, -1)),  # xm
        (-2,             slice(1, -1), slice(1, -1)),  # xp
        (slice(1, -1), 1,              slice(1, -1)),  # ym
        (slice(1, -1), -2,             slice(1, -1)),  # yp
        (slice(1, -1), slice(1, -1), 1),               # zm
        (slice(1, -1), slice(1, -1), -2),              # zp
    ]

    recv_idx = [
        (0,              slice(1, -1), slice(1, -1)),  # xm
        (-1,             slice(1, -1), slice(1, -1)),  # xp
        (slice(1, -1), 0,              slice(1, -1)),  # ym
        (slice(1, -1), -1,             slice(1, -1)),  # yp
        (slice(1, -1), slice(1, -1), 0),               # zm
        (slice(1, -1), slice(1, -1), -1),              # zp
    ]



    ERR = np.inf
    it = 0

    phi_prev = phi.copy()



    while (ERR > max_err) and (it < max_it):
        # We need to get the square sorounding the square we are solving
        phi[1:-1, 1:-1, 1:-1] = stencil_update(phi_prev, F, dx)
        # phi = enforce_bounds(phi)
        err_local = np.max(np.abs(phi - phi_prev))

        it += 1
        phi_prev, phi = phi, phi_prev

        # This is the communication step where we send boundaries

        phi_prev = send_recive(phi_prev, neighbors, send_idx, recv_idx, cart)

        ERR = cart.allreduce(err_local, op=MPI.MAX)

        if (it % 100 == 0) and rank == 0 and report:
            print(f"{ERR=} at {it=}")


    err_inf_local = np.max(
    np.abs(phi_prev[1:-1, 1:-1, 1:-1] - u)
    )

    err_inf = cart.allreduce(
        err_inf_local,
        op=MPI.MAX
    )

    if rank == 0 and report:
        print(f"{ERR=} at {it=}")
        print(phi_prev)
        print(f"True final error: {err_inf}")
    return err_inf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-int",
        type=int,
        default=100,
        help="Number of interior grid points per dimension"
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000000,
        help="Maximum number of iterations"
    )

    parser.add_argument(
        "--tolerence",
        type=float,
        default=1e-14,
        help="Error tolerance"
    )

    args = parser.parse_args()

    N_int = args.n_int
    maxiter = args.maxiter
    tolerence = args.tolerence

    t1 = time()

    error = run_Poisson_square(
        N_int,
        max_it=maxiter,
        max_err=tolerence
    )

    t2 = time()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(error)
        print("Iterations per second", maxiter / (t2 - t1))