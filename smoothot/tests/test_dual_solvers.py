# Author: Mathieu Blondel
# License: Simplified BSD

import numpy as np
from sklearn.utils.testing import assert_almost_equal

from smoothot.dual_solvers import solve_dual, solve_semi_dual
from smoothot.dual_solvers import dual_obj_grad, semi_dual_obj_grad
from smoothot.dual_solvers import Entropy, SquaredL2

rng = np.random.RandomState(0)

a = rng.rand(5)
b = rng.rand(8)
a /= np.sum(a)
b /= np.sum(b)
C = rng.rand(5, 8)


def test_dual_and_semi_dual():
    for gamma in (0.1, 1.0, 10.0, 50.0):
        for regul in (Entropy(gamma), SquaredL2(gamma)):
            alpha, beta = solve_dual(a, b, C, regul, max_iter=1000)
            val_dual = dual_obj_grad(alpha, beta, a, b, C, regul)[0]

            alpha = solve_semi_dual(a, b, C, regul, max_iter=1000)
            val_semi_dual = semi_dual_obj_grad(alpha, a, b, C, regul)[0]

            assert_almost_equal(val_dual, val_semi_dual, 2)
