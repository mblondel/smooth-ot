import numpy as np
from sklearn.utils.testing import assert_array_almost_equal

from smoothot.dual_solvers import projection_simplex


def _projection_simplex(v, z=1):
    """
    Old implementation for test and benchmark purposes.
    The arguments v and z should be a vector and a scalar, respectively.
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def test_projection_simplex():
    rng = np.random.RandomState(0)
    V = rng.rand(100, 10)

    # Axis = None case.
    w = projection_simplex(V[0], z=1, axis=None)
    w2 = _projection_simplex(V[0], z=1)
    assert_array_almost_equal(w, w2)

    w = projection_simplex(V, z=1, axis=None)
    w2 = _projection_simplex(V.ravel(), z=1)
    assert_array_almost_equal(w, w2)

    # Axis = 1 case.
    W = projection_simplex(V, axis=1)

    # Check same as with for loop.
    W2 = np.array([_projection_simplex(V[i]) for i in range(V.shape[0])])
    assert_array_almost_equal(W, W2)

    # Check works with vector z.
    W3 = projection_simplex(V, np.ones(V.shape[0]), axis=1)
    assert_array_almost_equal(W, W3)

    # Axis = 0 case.
    W = projection_simplex(V, axis=0)

    # Check same as with for loop.
    W2 = np.array([_projection_simplex(V[:, i]) for i in range(V.shape[1])]).T
    assert_array_almost_equal(W, W2)

    # Check works with vector z.
    W3 = projection_simplex(V, np.ones(V.shape[1]), axis=0)
    assert_array_almost_equal(W, W3)
