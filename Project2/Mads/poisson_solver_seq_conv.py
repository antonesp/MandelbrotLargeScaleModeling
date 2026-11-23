import numpy as np
from poisson_helper import force_term, true_sol, stencil_update

# N_int = 100
N_int = np.arange(20, 100, step=2, dtype=np.int64)

dxs = []
errors = []

for n_int in N_int:
    N_bound = 2
    N_tot = n_int + N_bound

    x, dx = np.linspace(-1, 1, num=N_tot, endpoint=True, retstep=True)
    y = np.linspace(-1, 1, num=N_tot, endpoint=True)
    z = np.linspace(-1, 1, num=N_tot, endpoint=True)
    dxs.append(dx)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    F = force_term(X, Y, Z)
    u = true_sol(X, Y, Z)
    phi = np.zeros(shape=(N_tot, N_tot, N_tot))

    max_it = 1_000_000
    max_err = 1e-12

    ERR = np.inf
    it = 0
    phi_prev = phi.copy()

    while (ERR > max_err) and (it < max_it):
        phi[1:-1, 1:-1, 1:-1] = stencil_update(phi_prev, F, dx)
        ERR = np.max(np.abs(phi - phi_prev))
        it += 1
        phi_prev, phi = phi, phi_prev
    true_error = np.max(np.abs(phi_prev - u)) 
    errors.append(true_error)
    print(f"True final error: {true_error} at {n_int} interior points")

dxs = np.array(dxs)
errors = np.array(errors)

np.savetxt('Mads/Results/Convergence/seq_conv.csv', np.column_stack((N_int, dxs, errors)), 
           delimiter=',', header='N_int,dx,error', comments='')