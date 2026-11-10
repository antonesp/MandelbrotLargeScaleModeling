import numpy as np
from poisson_helper import force_term, true_sol, stencil_update

N_int = 100
N_bound = 2
N_tot = N_int + N_bound

x, dx = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)
y, dy = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)
z, dz = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)

X, Y, Z = np.meshgrid(x, y, z)

F = force_term(X, Y, Z)
u = true_sol(X, Y, Z)
phi = np.zeros(shape=(N_tot, N_tot, N_tot))

max_it = 1_000_000
max_err = 1e-6

ERR = np.inf
it = 0
phi_prev = phi.copy()

while (ERR > max_err) and (it < max_it):
    phi[1:-1, 1:-1, 1:-1] = stencil_update(phi_prev, F, dx)
    # phi = enforce_bounds(phi)
    ERR = np.max(np.abs(phi - phi_prev))
    it += 1
    phi_prev, phi = phi, phi_prev
    if (it % 100 == 0):
        print(f"{ERR=} at {it=}")
print(f"{ERR=} at {it=}")
print(f"True final error: {np.max(np.abs(phi_prev - u))}")