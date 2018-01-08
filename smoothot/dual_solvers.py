# Author: Mathieu Blondel
# License: Simplified BSD

"""
Implementation of

Smooth and Sparse Optimal Transport.
Mathieu Blondel, Vivien Seguy, Antoine Rolet.
In Proc. of AISTATS 2018.
https://arxiv.org/abs/1710.06276
"""

import numpy as np
from scipy.optimize import minimize

from .projection import projection_simplex


class Regularization(object):

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def delta_Omega(X):
        raise NotImplementedError

    def max_Omega(X, b):
        raise NotImplementedError


class NegEntropy(Regularization):

    def delta_Omega(self, X):
        G = np.exp(X / self.gamma - 1)
        val = self.gamma * np.sum(G, axis=0)
        return val, G

    def max_Omega(self, X, b):
        max_X = np.max(X, axis=0) / self.gamma
        exp_X = np.exp(X / self.gamma - max_X)
        val = self.gamma * (np.log(np.sum(exp_X, axis=0)) + max_X)
        val -= self.gamma * np.log(b)
        G = exp_X / np.sum(exp_X, axis=0)
        return val, G

    def Omega(self, T):
        return self.gamma * np.sum(T * np.log(T))


class SquaredL2(Regularization):

    def delta_Omega(self, X):
        max_X = np.maximum(X, 0)
        val = np.sum(max_X ** 2, axis=0) / (2 * self.gamma)
        G = max_X / self.gamma
        return val, G

    def max_Omega(self, X, b):
        G = projection_simplex(X / (b * self.gamma), axis=0)
        val = np.sum(X * G, axis=0)
        val -= 0.5 * self.gamma * b * np.sum(G * G, axis=0)
        return val, G

    def Omega(self, T):
        return 0.5 * self.gamma * np.sum(T ** 2)


def dual_obj_grad(alpha, beta, a, b, C, regul):
    obj = np.dot(alpha, a) + np.dot(beta, b)
    grad_alpha = a.copy()
    grad_beta = b.copy()

    # X[:, j] = alpha + beta[j] - C[:, j]
    X = alpha[:, np.newaxis] + beta - C

    # val.shape = len(b)
    # G.shape = len(a) x len(b)
    val, G = regul.delta_Omega(X)

    obj -= np.sum(val)
    grad_alpha -= G.sum(axis=1)
    grad_beta -= G.sum(axis=0)

    return obj, grad_alpha, grad_beta


def solve_dual(a, b, C, regul, method="L-BFGS-B", tol=1e-3, max_iter=500):
    """
    Solve the "smoothed" dual objective.

    Parameters
    ----------
    a: array, shape = len(a)
    b: array, shape = len(b)
        Input histograms (should be non-negative and sum to 1).

    C: array, shape = len(a) x len(b)
        Ground cost matrix.

    regul: Regularization object
        Should implement a delta_Omega(X) method.

    method: str
        Solver to be used (passed to `scipy.optimize.minimize`).

    tol: float
        Tolerance parameter.

    max_iter: int
        Maximum number of iterations.

    Returns
    -------
    alpha: array, shape = len(a)
    beta: array, shape = len(b)
        Dual potentials.
    """

    def _func(params):
        # Unpack alpha and beta.
        alpha = params[:len(a)]
        beta = params[len(a):]

        obj, grad_alpha, grad_beta = dual_obj_grad(alpha, beta, a, b, C, regul)

        # Pack grad_alpha and grad_beta.
        grad = np.concatenate((grad_alpha, grad_beta))

        # We need to maximize the dual.
        return -obj, -grad

    # Unfortunately, `minimize` only supports functions whose argument is a
    # vector. So, we need to concatenate alpha and beta.
    alpha_init = np.zeros(len(a))
    beta_init = np.zeros(len(b))
    params_init = np.concatenate((alpha_init, beta_init))

    res = minimize(_func, params_init, method=method, jac=True,
                   tol=tol, options=dict(maxiter=max_iter, disp=False))

    alpha = res.x[:len(a)]
    beta = res.x[len(a):]

    return alpha, beta


def semi_dual_obj_grad(alpha, a, b, C, regul):
    obj = np.dot(alpha, a)
    grad = a.copy()

    # X[:, j] = alpha - C[:, j]
    X = alpha[:, np.newaxis] - C

    # val.shape = len(b)
    # G.shape = len(a) x len(b)
    val, G = regul.max_Omega(X, b)

    obj -= np.dot(b, val)
    grad -= np.dot(G, b)

    return obj, grad


def solve_semi_dual(a, b, C, regul, method="L-BFGS-B", tol=1e-3, max_iter=500):
    """
    Solve the "smoothed" semi-dual objective.

    Parameters
    ----------
    a: array, shape = len(a)
    b: array, shape = len(b)
        Input histograms (should be non-negative and sum to 1).

    C: array, shape = len(a) x len(b)
        Ground cost matrix.

    regul: Regularization object
        Should implement a delta_Omega(X) method.

    method: str
        Solver to be used (passed to `scipy.optimize.minimize`).

    tol: float
        Tolerance parameter.

    max_iter: int
        Maximum number of iterations.

    Returns
    -------
    alpha: array, shape = len(a)
        Semi-dual potentials.
    """

    def _func(alpha):
        obj, grad = semi_dual_obj_grad(alpha, a, b, C, regul)
        # We need to maximize the semi-dual.
        return -obj, -grad

    alpha_init = np.zeros(len(a))

    res = minimize(_func, alpha_init, method=method, jac=True,
                   tol=tol, options=dict(maxiter=max_iter, disp=False))

    return res.x


def get_plan_from_dual(alpha, beta, C, regul):
    """
    Retrieve optimal transportation plan from optimal dual potentials.

    Parameters
    ----------
    alpha: array, shape = len(a)
    beta: array, shape = len(b)
        Optimal dual potentials.

    C: array, shape = len(a) x len(b)
        Ground cost matrix.

    regul: Regularization object
        Should implement a delta_Omega(X) method.

    Returns
    -------
    T: array, shape = len(a) x len(b)
        Optimal transportation plan.
    """
    X = alpha[:, np.newaxis] + beta - C
    return regul.delta_Omega(X)[1]


def get_plan_from_semi_dual(alpha, b, C, regul):
    """
    Retrieve optimal transportation plan from optimal semi-dual potentials.

    Parameters
    ----------
    alpha: array, shape = len(a)
        Optimal semi-dual potentials.

    b: array, shape = len(b)
        Second input histogram (should be non-negative and sum to 1).

    C: array, shape = len(a) x len(b)
        Ground cost matrix.

    regul: Regularization object
        Should implement a delta_Omega(X) method.

    Returns
    -------
    T: array, shape = len(a) x len(b)
        Optimal transportation plan.
    """
    X = alpha[:, np.newaxis] - C
    return regul.max_Omega(X, b)[1] * b
