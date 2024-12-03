import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Define the Goldstein-Price function
def goldstein_price(x, y):
    return (
        1
        + ((x + y + 1) ** 2)
        * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + ((2 * x - 3 * y) ** 2)
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )

# Generate data for visualization
grid_size = 100
x = np.linspace(-2, 2, grid_size)
y = np.linspace(-2, 2, grid_size)
X, Y = np.meshgrid(x, y)
Z = goldstein_price(X, Y)

# Normalize the function for better modeling
Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

# Random initial samples for Bayesian optimization
np.random.seed(42)
n_initial_samples = 10
X_samples = np.random.uniform(-2, 2, (n_initial_samples, 2))
y_samples = goldstein_price(X_samples[:, 0], X_samples[:, 1])

# Train a Gaussian Process surrogate model
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
gp.fit(X_samples, y_samples)

# Define QEI acquisition function
def qei(x, gp, y_best):
    x = x.reshape(-1, 2)
    mu, sigma = gp.predict(x, return_std=True)
    sigma = sigma.reshape(-1, 1)
    improvement = mu - y_best
    z = improvement / sigma
    qei_value = sigma * (z * norm.cdf(z) + norm.pdf(z))
    return qei_value.ravel()

# Compute QEI values for the grid
y_best = np.min(y_samples)  # Current best value
grid_points = np.c_[X.ravel(), Y.ravel()]
qei_values = qei(grid_points, gp, y_best).reshape(X.shape)

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, qei_values, levels=100, cmap="viridis")
plt.colorbar(label="QEI Value")
plt.scatter(X_samples[:, 0], X_samples[:, 1], color="red", label="Sampled Points")
plt.title("q-Expected Improvement Heatmap for Goldstein-Price Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()