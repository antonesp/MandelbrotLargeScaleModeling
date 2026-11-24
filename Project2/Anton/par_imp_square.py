import argparse
import numpy as np
from poisson_helper import force_term, stencil_update, true_sol
from mpi4py import MPI
from Anton.more_help_funcs import send_recive, split_dimension
from time import time


def run_Poisson_square(N_int, max_it=1_000_000, max_err=1e-6, report=False):
    # Match the sheet implementation's timing convention:
    # t1 -> start of setup, t2 -> start of iterative solve, t3 -> end of solve.
    t1 = time()

    comm = MPI.COMM_WORLD.Clone()
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

    Px, Py, Pz = dims

    x, dx = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)
    y, dy = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)
    z, dz = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)

    x_int = x[1:-1]
    y_int = y[1:-1]
    z_int = z[1:-1]

    rx, ry, rz = coords

    ix0, nx = split_dimension(N_int, Px, rx)
    iy0, ny = split_dimension(N_int, Py, ry)
    iz0, nz = split_dimension(N_int, Pz, rz)

    subproblem_dimensions = cart.gather((nx, ny, nz), root=0)

    x_local = x_int[ix0:ix0 + nx]
    y_local = y_int[iy0:iy0 + ny]
    z_local = z_int[iz0:iz0 + nz]

    X, Y, Z = np.meshgrid(x_local, y_local, z_local, indexing="ij")

    phi = np.zeros((nx + 2, ny + 2, nz + 2))

    F = force_term(X, Y, Z)
    F = np.pad(
        F,
        pad_width=1,
        mode="constant",
        constant_values=0,
    )
    u = true_sol(X, Y, Z)

    xm, xp = cart.Shift(0, 1)
    ym, yp = cart.Shift(1, 1)
    zm, zp = cart.Shift(2, 1)

    neighbors = [xm, xp, ym, yp, zm, zp]

    # Count the number of grid-point values sent in one halo exchange.
    # Each internal interface is counted once from each neighboring rank,
    # matching the actual bidirectional communication performed per iteration.
    local_comm_points = 0
    if xm != MPI.PROC_NULL:
        local_comm_points += ny * nz
    if xp != MPI.PROC_NULL:
        local_comm_points += ny * nz
    if ym != MPI.PROC_NULL:
        local_comm_points += nx * nz
    if yp != MPI.PROC_NULL:
        local_comm_points += nx * nz
    if zm != MPI.PROC_NULL:
        local_comm_points += nx * ny
    if zp != MPI.PROC_NULL:
        local_comm_points += nx * ny

    comm_points_per_iter = cart.allreduce(local_comm_points, op=MPI.SUM)
    internal_interface_area = comm_points_per_iter // 2

    send_idx = [
        (1, slice(1, -1), slice(1, -1)),
        (-2, slice(1, -1), slice(1, -1)),
        (slice(1, -1), 1, slice(1, -1)),
        (slice(1, -1), -2, slice(1, -1)),
        (slice(1, -1), slice(1, -1), 1),
        (slice(1, -1), slice(1, -1), -2),
    ]

    recv_idx = [
        (0, slice(1, -1), slice(1, -1)),
        (-1, slice(1, -1), slice(1, -1)),
        (slice(1, -1), 0, slice(1, -1)),
        (slice(1, -1), -1, slice(1, -1)),
        (slice(1, -1), slice(1, -1), 0),
        (slice(1, -1), slice(1, -1), -1),
    ]

    ERR = np.inf
    it = 0
    phi_prev = phi.copy()

    # Same instrumentation as the sheet implementation.
    band_time = 0.0
    comm_time = 0.0

    t2 = time()

    while (ERR > max_err) and (it < max_it):
        t1_band = time()
        phi[1:-1, 1:-1, 1:-1] = stencil_update(phi_prev, F, dx)
        t2_band = time()
        band_time += t2_band - t1_band

        err_local = np.max(np.abs(phi - phi_prev))

        it += 1
        phi_prev, phi = phi, phi_prev

        t1_comm = time()
        phi_prev = send_recive(phi_prev, neighbors, send_idx, recv_idx, cart)
        t2_comm = time()
        comm_time += t2_comm - t1_comm

        ERR = cart.allreduce(err_local, op=MPI.MAX)

    t3 = time()

    err_inf_local = np.max(
        np.abs(phi_prev[1:-1, 1:-1, 1:-1] - u)
    )
    err_inf = cart.allreduce(err_inf_local, op=MPI.MAX)

    if rank == 0 and report:
        time_spent = t3 - t2
        Mlup = N_int**3 * it / (time_spent * 1e6)
        MBs = Mlup * 64
        MFlops = N_int**3 * it * 9 / (time_spent * 1e6)
        overhead = t2 - t1
        compute_time = t3 - t2
        total_time = t3 - t1
        overhead_pct = 100 * overhead / total_time
        comm_pct = 100 * comm_time / compute_time
        iter_per_sec = it / compute_time
        grid_kb = (N_tot**3 * 8) / 1e3

        print(
            f"\n{' Simulation Summary ':=^56}\n"
            f"Problem\n"
            f"  {'MPI ranks:':<24} {cores:>10d}\n"
            f"  {'Process grid:':<24} {str(tuple(dims)):>10}\n"
            f"  {'Interior points:':<24} {N_int:>10d}^3\n"
            f"  {'Subproblem dimensions:':<24} {str(subproblem_dimensions)}\n"
            f"  {'Internal interface area:':<24} {internal_interface_area:>10d} points\n"
            f"  {'Communication points/iter:':<24} {comm_points_per_iter:>10d} values\n"
            f"\n"
            f"Timing\n"
            f"  {'Compute time:':<24} {compute_time:>10.5f} s  ({iter_per_sec:,.2f} it/s)\n"
            f"  {'Overhead time:':<24} {overhead:>10.5f} s  ({overhead_pct:.2f}% of total)\n"
            f"  {'Communication time:':<24} {comm_time:>10.5f} s  ({comm_pct:.2f}% of compute time)\n"
            f"  {'Stencil time:':<24} {band_time:>10.5f} s\n"
            f"\n"
            f"Convergence (it = {it:,})\n"
            f"  {'Steady-state error:':<24} {ERR:>10.3e}\n"
            f"  {'True final error:':<24} {err_inf:>10.3e}\n"
            f"\n"
            f"Performance (Grid: {grid_kb:,.1f} KB)\n"
            f"  {'Throughput:':<24} {Mlup:>10.2f} MLUP/s\n"
            f"  {'Memory bandwidth:':<24} {MBs:>10.2f} MB/s\n"
            f"  {'Compute rate:':<24} {MFlops:>10.2f} MFLOP/s\n"
            f"{'=' * 56}\n"
        )

    return err_inf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-int",
        type=int,
        default=100,
        help="Number of interior grid points per dimension",
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=1_000_000,
        help="Maximum number of iterations",
    )

    parser.add_argument(
        "--tolerence",
        type=float,
        default=1e-14,
        help="Error tolerance",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the simulation summary and print only the final true error",
    )

    args = parser.parse_args()

    error = run_Poisson_square(
        args.n_int,
        max_it=args.maxiter,
        max_err=args.tolerence,
        report=not args.quiet,
    )

    if MPI.COMM_WORLD.Get_rank() == 0 and args.quiet:
        print(error)
