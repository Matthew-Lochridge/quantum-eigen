import numpy as np
from src.eigen import solve_eigen

def test_small_grid():
    vals, _ = solve_eigen(N=5, potential='well', n_eigs=3)
    assert len(vals) == 3
    # Basic check: eigenvalues should be ascending
    assert np.all(np.diff(vals) >= 0), "Eigenvalues are not sorted"


def test_bc_matrix_size():
    # With Dirichlet BC the effective matrix is (N-2)^2
    N = 5
    vals, vecs = solve_eigen(N=N, potential='well', n_eigs=3, bc='dirichlet')
    assert vecs.shape[0] == (N-2)**2
    assert len(vals) == 3

def test_vecs_normalized():
    # Check if the ground state is normalized (L2 norm = 1)
    _, vecs = solve_eigen(N=5, potential='well', n_eigs=1)
    gs = vecs[:, 0]
    norm = np.linalg.norm(gs)
    assert np.isclose(norm, 1.0), "Ground state is not normalized" # default tolerances are sufficient to detect differences if they exist
