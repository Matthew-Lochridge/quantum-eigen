import numpy as np
from scipy.linalg import eigh
import argparse

def build_2d_hamiltonian(N=20, potential='well'):
    """
    Build a discretized 2D Hamiltonian on an N x N grid.
    Parameters---------
    N : int
    Number of points in each dimension (N^2 total points).
    potential : str
    Choose the potential. 'well' or 'harmonic' examples.
    Returns------
    H : ndarray of shape (N^2, N^2)
    The Hamiltonian matrix approximating-d^2/dx^2- d^2/dy^2 + V(x,y).
    """
    dx = 1. / float(N) # grid spacing, can be arbitrary
    inv_dx2 = float(N * N) # 1/dx^2
    H = np.zeros((N*N, N*N), dtype=np.float64)

    # Helper function to map (i,j)-> linear index
    def idx(i, j):
        return i * N + j

    # Potential function
    def V(i, j):
        # Example 1: infinite square well-> zero in interior, large outside
        if potential == 'well':
            # No boundary enforcement here, but can skip boundary wavefunction
            return 0.
        # Example 2: 2D harmonic oscillator around center
        elif potential == 'harmonic':
            x = (i- N/2) * dx
            y = (j- N/2) * dx
            # Quadratic potential V = k * (x^2 + y^2)
            return 4. * (x**2 + y**2)
        elif potential == 'sinusoidal':
            x = (i- N/2) * dx
            y = (j- N/2) * dx
            # Sinusoidal potential V = k * sin(pi*x) * sin(pi*y)
            return 10. * np.sin(np.pi * x) * np.sin(np.pi * y)
        else:
            return 0.

    # Build the matrix: For each (i, j), set diagonal for 2D Laplacian plus V
    for i in range(N):
        for j in range(N):
            row = idx(i,j)
            # Potential
            H[row, row] =-4. * inv_dx2 + V(i,j) # "Kinetic" ~-4/dx^2 in 2D FD
            # Neighbors (assuming no boundary conditions or Dirichlet)
            if i > 0: # up
                H[row, idx(i-1, j)] = inv_dx2
            if i < N-1: # down
                H[row, idx(i+1, j)] = inv_dx2
            if j > 0: # left
                H[row, idx(i, j-1)] = inv_dx2
            if j < N-1: # right
                H[row, idx(i, j+1)] = inv_dx2
    return H

def solve_eigen(N=20, potential='well', n_eigs=None):
    """
    Build a 2D Hamiltonian and solve for the lowest n_eigs eigenvalues.
    Parameters---------
    N : int
    Grid points in each dimension.
    potential : str
    Potential type.
    n_eigs : int
    Number of eigenvalues to return.
    Returns------
    vals : array_like
    The lowest n_eigs eigenvalues sorted ascending.
    vecs : array_like
    The corresponding eigenvectors.
    """
    H = build_2d_hamiltonian(N, potential)
    # Solve entire spectrum (careful for large N)
    vals, vecs = eigh(H)
    # Sort
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
    parser.add_argument('--save_gs', type=bool, default=False, help='Flag to save ground state probability density')
    args = parser.parse_args()
    if args.N <= 0:
        parser.error(f"N must be positive, got N={args.N}")
    if args.n_eigs <= 0:
        parser.error(f"n_eigs must be positive, got n_eigs={args.n_eigs}")
    if args.n_eigs > args.N**2:
        parser.error(f"n_eigs must be <= N^2, got n_eigs={args.n_eigs} with N={args.N}")
    # Example local test
    vals, vecs = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.n_eigs)
    np.savetxt(f'results/eigs_N{args.N}_V{args.potential}.txt', vals)
    if args.save_gs:
        # Save ground state (first eigenvector) probability density
        gs = vecs[:, 0]
        gs_prob = np.abs(gs)**2
        np.savetxt(f'results/gsProb_N{args.N}_V{args.potential}.txt', gs_prob.reshape(args.N, args.N))

