import numpy as np
from scipy.linalg import eigh
import argparse

def build_2d_hamiltonian(N=20, potential='well', bc=None):
    """
    Build a discretized 2D Hamiltonian on an N x N grid.

    The grid includes the boundary points by default.  When `bc=='dirichlet'`
    the boundary is assumed to have zero wavefunction and the corresponding
    points are eliminated from the system, resulting in an `(N-2)^2` matrix.

    Parameters---------
    N : int
        Number of points in each dimension (N^2 total points when bc is None).
    potential : str
        Choose the potential. 'well', 'harmonic' or 'sinusoidal' examples.
    bc : None or 'dirichlet'
        Type of boundary condition.  Only Dirichlet (zero at boundary) is
        currently supported; when specified the boundary points are removed
        from the linear system.

    Returns------
    H : ndarray
        The Hamiltonian matrix approximating -d^2/dx^2 - d^2/dy^2 + V(x,y).
        Shape is (N^2, N^2) for no boundary condition, or ((N-2)^2,
        (N-2)^2) when bc=='dirichlet'.
    """
    dx = 1. / float(N)  # grid spacing, can be arbitrary
    inv_dx2 = float(N * N)  # 1/dx^2 (hbar^2/2m = 1 for simplicity)

    # potential helper
    def V(i, j):
        if potential == 'well':
            return 0.
        elif potential == 'harmonic': # isotropic harmonic oscillator potential with k = 8
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            return 4. * (x**2 + y**2)
        elif potential == 'sinusoidal':
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            return 100. * np.sin(np.pi * x) * np.sin(np.pi * y)
        else:
            return 0.

    if bc == 'dirichlet':
        # build matrix only for interior points (skip i=0, i=N-1, j=0, j=N-1)
        if N < 3:
            raise ValueError("N must be at least 3 to have interior points with Dirichlet BC")
        interior = [(i, j) for i in range(1, N-1) for j in range(1, N-1)]
        M = len(interior)
        H = np.zeros((M, M), dtype=np.float64)
        idx_map = {pt: k for k, pt in enumerate(interior)}

        for (i, j), row in idx_map.items():
            # diagonal term
            H[row, row] = 4. * inv_dx2 + V(i, j)
            # neighbors only if also interior
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if (ni, nj) in idx_map:
                    H[row, idx_map[(ni, nj)]] = -inv_dx2
        return H

    # default: no boundary conditions, same as original implementation
    H = np.zeros((N * N, N * N), dtype=np.float64)

    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            row = idx(i, j)
            H[row, row] = 4. * inv_dx2 + V(i, j)
            if i > 0:  # up
                H[row, idx(i - 1, j)] = -inv_dx2
            if i < N - 1:  # down
                H[row, idx(i + 1, j)] = -inv_dx2
            if j > 0:  # left
                H[row, idx(i, j - 1)] = -inv_dx2
            if j < N - 1:  # right
                H[row, idx(i, j + 1)] = -inv_dx2
    return H

def solve_eigen(N=20, potential='well', n_eigs=None, bc=None):
    """
    Build a 2D Hamiltonian (possibly with boundary conditions) and solve for
    the lowest n_eigs eigenvalues.

    Parameters---------
    N : int
        Grid points in each dimension.
    potential : str
        Potential type.
    n_eigs : int
        Number of eigenvalues to return.
    bc : None or 'dirichlet'
        Type of boundary condition to apply when building the Hamiltonian.

    Returns------
    vals : array_like
        The lowest n_eigs eigenvalues sorted ascending.
    vecs : array_like
        The corresponding eigenvectors.
    """
    H = build_2d_hamiltonian(N, potential, bc=bc)
    vals, vecs = eigh(H)
    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]
    if n_eigs is None:
        return vals_sorted, vecs_sorted
    else:
        return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve 2D eigenvalue problem.')
    parser.add_argument('--N', type=int, default=20, help='Grid size (N x N)')
    parser.add_argument('--potential', type=str, choices=['well', 'harmonic', 'sinusoidal'], default='well', help='Potential type: well or harmonic')
    parser.add_argument('--n_eigs', type=int, default=6, help='Number of eigenvalues to compute')
    parser.add_argument('--bc', type=str, choices=[None, 'dirichlet'], default=None,
                        help='Boundary condition to enforce (currently only "dirichlet" supported)')
    parser.add_argument('--save_gs', type=bool, default=False, help='Flag to save ground state probability density')
    args = parser.parse_args()
    if args.N <= 0:
        parser.error(f"N must be positive, got N={args.N}")
    if args.n_eigs <= 0:
        parser.error(f"n_eigs must be positive, got n_eigs={args.n_eigs}")
    max_size = (args.N - 2)**2 if args.bc == 'dirichlet' else args.N**2
    if args.n_eigs > max_size:
        parser.error(f"n_eigs must be <= matrix size ({max_size}), got n_eigs={args.n_eigs} with N={args.N} and bc={args.bc}")
    # Example local test
    vals, vecs = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.n_eigs, bc=args.bc)
    print(f'Lowest {args.n_eigs} eigenvalues: {vals}')
    np.savetxt(f'results/eigs_N{args.N}_V{args.potential}.txt', vals)
    if args.save_gs:
        # Save ground state (first eigenvector) probability density
        gs_prob = np.abs(vecs[:, 0])**2
        if args.bc == 'dirichlet':
            # reshape to (N-2, N-2) for interior points only
            gs_prob = gs_prob.reshape(args.N - 2, args.N - 2)
            # pad with zeros to restore full N x N grid including boundaries
            padded = np.zeros((args.N, args.N), dtype=gs_prob.dtype)
            # interior points correspond to indices 1..N-2 in each dimension
            padded[1:-1, 1:-1] = gs_prob
            gs_prob = padded
        elif args.bc is None:
            gs_prob = gs_prob.reshape(args.N, args.N)
        np.savetxt(f'results/gsProb_N{args.N}_V{args.potential}.txt', gs_prob)

