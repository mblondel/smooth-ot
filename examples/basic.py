import numpy as np

from smoothot.dual_solvers import solve_dual, solve_semi_dual
from smoothot.dual_solvers import NegEntropy, SquaredL2
from smoothot.dual_solvers import get_plan_from_dual, get_plan_from_semi_dual


# Generate artificial data.
rng = np.random.RandomState(0)

a = rng.rand(5)
b = rng.rand(8)
a /= np.sum(a)
b /= np.sum(b)
C = rng.rand(5, 8)

# Select regularization term.
# Can also use NegEntropy(gamma=1.0).
regul = SquaredL2(gamma=1.0)

# Solve smoothed dual.
alpha, beta = solve_dual(a, b, C, regul, max_iter=1000)
T = get_plan_from_dual(alpha, beta, C, regul)
print("Dual")
print("Value:", np.sum(T * C) + regul.Omega(T))
print("Sparsity:", np.sum(T != 0) / T.size)
print()


# Solve smoothed semi-dual.
alpha = solve_semi_dual(a, b, C, regul, max_iter=1000)
T = get_plan_from_semi_dual(alpha, b, C, regul)
print("Semi-dual")
print("Value:", np.sum(T * C) + regul.Omega(T))
print("Sparsity:", np.sum(T != 0) / T.size)
print()
