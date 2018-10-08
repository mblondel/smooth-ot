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
        """
        Parameters
        ----------
        gamma: float
            Regularization parameter.
            We recover unregularized OT when gamma -> 0.
        """
        self.gamma = gamma

    def delta_Omega(X):
        """
        Compute delta_Omega(X[:, j]) for each X[:, j].

        delta_Omega(x) = sup_{y >= 0} y^T x - Omega(y).

        Parameters
        ----------
        X: array, shape = len(a) x len(b)
            Input array.

        Returns
        -------
        v: array, len(b)
            Values: v[j] = delta_Omega(X[:, j])

        G: array, len(a) x len(b)
            Gradients: G[:, j] = nabla delta_Omega(X[:, j])
        """
        raise NotImplementedError

    def max_Omega(X, b):
        """
        Compute max_Omega_j(X[:, j]) for each X[:, j].

        max_Omega_j(x) = sup_{y >= 0, sum(y) = 1} y^T x - Omega(b[j] y) / b[j].

        Parameters
        ----------
        X: array, shape = len(a) x len(b)
            Input array.

        Returns
        -------
        v: array, shape = len(b)
            Values: v[j] = max_Omega_j(X[:, j])

        G: array, shape = len(a) x len(b)
            Gradients: G[:, j] = nabla max_Omega_j(X[:, j])
        """
        raise NotImplementedError

    def Omega(T):
        """
        Compute regularization term.

        Parameters
        ----------
        T: array, shape = len(a) x len(b)
            Input array.

        Returns
        -------
        value: float
            Regularization term.
        """
        raise NotImplementedError


class NegEntropy(Regularization):
    """
    Omega(x) = gamma * np.dot(x, log(x))
    """

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
    """
    Omega(x) = 0.5 * gamma * ||x||^2
    """

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


class GroupLasso(Regularization):
    """
    Omega(x[g]) = 0.5 * gamma * (1-rho) ||x[g]||^2 + gamma * rho * ||x[g]||
    """

    def __init__(self, groups, gamma=1.0, rho=0.2):
        """
        Parameters
        ----------
        groups: array, shape = n_groups x len(b)
            Definition of non-overlapping groups. E.g.:

            [[True True, False, False],
             [False, False, True, False],
             [False, False, False, True]]

             defines three groups with len(b) = 4.

        gamma: float
            Regularization parameter.
            We recover unregularized OT when gamma -> 0.

        rho: float
            Proportion of squared 2-norm and of 2-norm.
        """
        self.groups = groups
        self.gamma = float(gamma)
        self.rho = float(rho)

    def _omega_g(self, t):
        sq_norm2 = np.sum(t ** 2)
        norm2 = np.sqrt(sq_norm2)
        ret = 0.5 * (1 - self.rho) * sq_norm2
        ret += self.rho * norm2
        return self.gamma * ret

    def Omega(self, T):
        ret = 0
        for j in range(T.shape[1]):
            for g in self.groups:
                ret += self._omega_g(T[:, j][g])
        return ret

    def _delta_omega(self, x):
        gamma = self.gamma * (1 - self.rho)
        mu = self.rho / (1 - self.rho)

        grad = np.zeros_like(x)
        val = 0

        for g in self.groups:
            x_g = x[g]
            x_g_plus = np.maximum(x_g, 0) / gamma
            norm2 = np.sqrt(np.sum(x_g_plus ** 2))

            if norm2 > 0:
                y_g = max(0, 1 - mu / norm2) * x_g_plus
            else:
                y_g = np.zeros_like(x_g_plus)

            val += np.dot(y_g, x_g) - self._omega_g(y_g)
            grad[g] = y_g

        return val, grad

    def delta_Omega(self, X):
        v = np.zeros(X.shape[1], dtype=np.float64)
        G = np.zeros(X.shape, dtype=np.float64)
        # FIXME: vectorize this code.
        for i in range(X.shape[1]):
            v[i], G[:, i] = self._delta_omega(X[:, i])
        return v, G


def dual_obj_grad(alpha, beta, a, b, C, regul):
    """
    Compute objective value and gradients of dual objective.

    Parameters
    ----------
    alpha: array, shape = len(a)
    beta: array, shape = len(b)
        Current iterate of dual potentials.

    a: array, shape = len(a)
    b: array, shape = len(b)
        Input histograms (should be non-negative and sum to 1).

    C: array, shape = len(a) x len(b)
        Ground cost matrix.

    regul: Regularization object
        Should implement a delta_Omega(X) method.

    Returns
    -------
    obj: float
        Objective value (higher is better).

    grad_alpha: array, shape = len(a)
        Gradient w.r.t. alpha.

    grad_beta: array, shape = len(b)
        Gradient w.r.t. beta.
    """
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
    """
    Compute objective value and gradient of semi-dual objective.

    Parameters
    ----------
    alpha: array, shape = len(a)
        Current iterate of semi-dual potentials.

    a: array, shape = len(a)
    b: array, shape = len(b)
        Input histograms (should be non-negative and sum to 1).

    C: array, shape = len(a) x len(b)
        Ground cost matrix.

    regul: Regularization object
        Should implement a max_Omega(X) method.

    Returns
    -------
    obj: float
        Objective value (higher is better).

    grad: array, shape = len(a)
        Gradient w.r.t. alpha.
    """
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
        Should implement a max_Omega(X) method.

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
