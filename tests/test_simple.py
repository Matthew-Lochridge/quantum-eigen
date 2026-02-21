import numpy as np
from src.eigen import solve_eigen

def test_small_grid():
    vals, _, _ = solve_eigen(N=5, potential='well', n_eigs=3)
    assert len(vals) == 3
    # Basic check: eigenvalues should be ascending
    assert np.all(np.diff(vals) >= 0), "Eigenvalues are not sorted"


def test_bc_matrix_size():
    # With Dirichlet BC the effective matrix is (N-2)^2
    N = 5
    vals, vecs, boundary = solve_eigen(N=N, potential='well', n_eigs=3, bc='dirichlet')
    assert vecs.shape[0] == (N-2)**2
    assert len(vals) == 3
    # boundary array should be returned and have shape N x N
    assert boundary.shape == (N, N)
    # boundary values should correspond to x+y on edges (matching the file's
    # boundary_value definition)
    dx = 1. / N
    def phys(i):
        return (i - N/2) * dx
    for i in range(N):
        for j in range(N):
            if i in (0, N-1) or j in (0, N-1):
                expected = phys(i) + phys(j)
                assert np.isclose(boundary[i, j], expected)


def test_vecs_normalized():
    # Check if the ground state is normalized (L2 norm = 1)
    _, vecs, _ = solve_eigen(N=5, potential='well', n_eigs=1, bc='dirichlet')
    gs = vecs[:, 0]
    norm = np.linalg.norm(gs)
    assert np.isclose(norm, 1.0), "Ground state is not normalized" # default tolerances are sufficient to detect differences if they exist
