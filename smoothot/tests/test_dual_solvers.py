# Author: Mathieu Blondel
# License: Simplified BSD

import numpy as np
from sklearn.utils.testing import assert_almost_equal

from smoothot.dual_solvers import solve_dual, solve_semi_dual
from smoothot.dual_solvers import dual_obj_grad, semi_dual_obj_grad
from smoothot.dual_solvers import NegEntropy, SquaredL2, GroupLasso
from smoothot.dual_solvers import get_plan_from_dual, get_plan_from_semi_dual

rng = np.random.RandomState(0)

a = rng.rand(5)
b = rng.rand(8)
a /= np.sum(a)
b /= np.sum(b)
C = rng.rand(5, 8)


def test_dual_and_semi_dual():
    for gamma in (0.1, 1.0, 10.0):
        for regul in (NegEntropy(gamma), SquaredL2(gamma)):
            alpha, beta = solve_dual(a, b, C, regul, max_iter=1000)
            val_dual = dual_obj_grad(alpha, beta, a, b, C, regul)[0]

            alpha_sd = solve_semi_dual(a, b, C, regul, max_iter=1000)
            val_sd = semi_dual_obj_grad(alpha_sd, a, b, C, regul)[0]

            # Check that dual and semi-dua indeed get the same objective value.
            assert_almost_equal(val_dual, val_sd, 2)

            # Check primal value too.
            T = get_plan_from_dual(alpha, beta, C, regul)
            val_primal = np.sum(T * C) + regul.Omega(T)
            assert_almost_equal(val_dual, val_primal, 1)

            T = get_plan_from_semi_dual(alpha, b, C, regul)
            val_primal = np.sum(T * C) + regul.Omega(T)
            assert_almost_equal(val_sd, val_primal, 1)


def test_group_lasso_regul():
    X = rng.rand(len(a), len(b))
    groups = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 1]])
    gl = GroupLasso(groups)
    v, G = gl.delta_Omega(X)

