import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

np.random.seed(0)
A = np.random.randn(30, 2)
b = np.random.randn(30)

def solve_lasso(lam: float = 0.1) -> np.ndarray:
    x = cp.Variable(2)
    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lam * cp.norm1(x))
    prob = cp.Problem(objective)
    prob.solve()
    return np.asarray(x.value, dtype=float).ravel()

lambdas = [0.05, 0.2, 0.8, 1.5]

for lam in lambdas:
    x_val = solve_lasso(lam)
    plt.figure()
    plt.scatter(A[:, 0], A[:, 1], c=b)
    plt.plot([0, x_val[0]], [0, x_val[1]], "r-o")
    plt.title(f"Lasso solution λ={lam:.2f}")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.grid(True)

# Show everything at once
plt.show()
