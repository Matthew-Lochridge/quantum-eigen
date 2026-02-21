import numpy as np
from src.eigen import solve_eigen, pad_and_smooth_probability

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
    # boundary values should correspond to ax+by on edges (matching the file's
    # boundary_value definition)
    dx = 1. / N
    a = 0.003
    b = 0.003
    def phys(i):
        return (i - N/2) * dx
    for i in range(N):
        for j in range(N):
            if i in (0, N-1) or j in (0, N-1):
                expected = a * phys(i) + b * phys(j)
                assert np.isclose(boundary[i, j], expected)


def test_vecs_normalized():
    # Check if the ground state is normalized (L2 norm = 1)
    _, vecs, _ = solve_eigen(N=5, potential='well', n_eigs=1, bc='dirichlet')
    gs = vecs[:, 0]
    norm = np.linalg.norm(gs)
    assert np.isclose(norm, 1.0), "Ground state is not normalized" # default tolerances are sufficient to detect differences if they exist


def test_bc_smoothing_and_positivity():
    # make sure the iterative pad/smooth helper returns a non-negative array
    # and that the boundary average influences the interior values
    N = 6
    vals, vecs, boundary = solve_eigen(N=N, potential='well', n_eigs=1, bc='dirichlet')
    gs = vecs[:, 0]
    gs_prob = np.abs(gs)**2
    gs_prob = gs_prob.reshape(N-2, N-2)
    padded = pad_and_smooth_probability(gs_prob, N, boundary)
    assert np.all(padded >= 0), "Smoothed probability contained negative values"
    # interior should not equal the raw eigenvector values after smoothing;
    # at least one value should have moved toward the mean boundary value
    mask = np.zeros((N, N), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    boundary_avg = np.mean(boundary[mask])
    assert not np.allclose(padded[1:-1, 1:-1], gs_prob)
    print("Boundary average:", boundary_avg)
    print("Interior mean after smoothing:", padded[1:-1, 1:-1].mean())


def test_smoothing_iteration_convergence():
    # when tol is reasonable the helper should converge well before max_iter;
    # forcing an extremely tight tolerance produces a warning that max_iter was
    # reached.
    N = 6
    vals, vecs, boundary = solve_eigen(N=N, potential='well', n_eigs=1, bc='dirichlet')
    gs = np.abs(vecs[:, 0])**2
    gs = gs.reshape(N-2, N-2)
    import warnings
    # no warnings with normal tolerance
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('error')
        pad_and_smooth_probability(gs, N, boundary, tol=1e-6, max_iter=500)
    # expect a RuntimeWarning when max_iter is too small for the tolerance
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        pad_and_smooth_probability(gs, N, boundary, tol=1e-20, max_iter=10)
        assert any(isinstance(x.message, RuntimeWarning) for x in w), \
            "Expected convergence warning when max_iter is reached"

