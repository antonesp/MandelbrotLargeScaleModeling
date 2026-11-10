from numpy import pi, sin

def force_term(x, y, z):
    return 3 * pi**2 * sin(pi * x) * sin(pi * y) * sin(pi * z)

def stencil_update(phi, F, step_size):
    phi_new = (
        phi[:-2, 1:-1, 1:-1] + phi[2:, 1:-1, 1:-1] +
        phi[1:-1, :-2, 1:-1] + phi[1:-1, 2:, 1:-1] +
        phi[1:-1, 1:-1, :-2] + phi[1:-1, 1:-1, 2:] +
        step_size**2 * F[1:-1, 1:-1, 1:-1]
    ) / 6

    return phi_new

def enforce_bounds(phi):
    phi[0, :, :] = 0.0
    phi[-1, :, :] = 0.0
    phi[:, 0, :] = 0.0
    phi[:, -1, :] = 0.0
    phi[:, :, 0] = 0.0
    phi[:, :, -1] = 0.0
    return phi

def true_sol(x, y, z):
    return sin(pi * x) * sin(pi * y) * sin(pi * z)