import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# Define the Goldstein–Price function
def goldstein_price(x1, x2):
    term1 = (1 + ((x1 + x2 + 1) ** 2) * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2))
    term2 = (30 + ((2 * x1 - 3 * x2) ** 2) * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))
    return term1 * term2

# Generate 32 training points using a Sobol sequence
sobol = qmc.Sobol(d=2, scramble=False)
points = sobol.random_base2(m=5)  # 2^5 = 32 points
scaled_pts = qmc.scale(points, l_bounds=[-2, -2], u_bounds=[2, 2])
values = np.array([goldstein_price(x1, x2) for x1, x2 in scaled_pts])
log_values = np.log(values)
D = pd.DataFrame(scaled_pts, columns=['x1', 'x2'])
D['data'] = log_values

# Fit a Gaussian process model to the data
X = D[['x1', 'x2']]
y = D['data']
kernel = ConstantKernel() * RBF()
gp = GPR(kernel=kernel, alpha=0.001, normalize_y=True)
gp.fit(X, y)

# Get the learned hyperparameters
length_scale = gp.kernel_.k2.length_scale
output_scale = gp.kernel_.k1.constant_value
print(f"Learned length scale: {length_scale}")
print(f"Learned output scale: {output_scale}")

# Create heatmaps of the Gaussian process posterior mean and standard deviation
x1 = np.linspace(-2, 2, 1000)
x2 = np.linspace(-2, 2, 1000)
X1, X2 = np.meshgrid(x1, x2)
grid_points = np.c_[X1.ravel(), X2.ravel()]

y_mean, y_std = gp.predict(grid_points, return_std=True)
Z_mean = y_mean.reshape(X1.shape)
Z_std = y_std.reshape(X1.shape)

# Plot the posterior mean heatmap
plt.figure(figsize=(8, 6))
ax1 = sns.heatmap(Z_mean, cbar_kws={'label': 'Predicted Value'})
tick_positions = np.linspace(0, len(x1) - 1, 5)
tick_labels = [-2, -1, 0, 1, 2]
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels)
ax1.set_yticks(tick_positions)
ax1.set_yticklabels(tick_labels)
plt.title("GP Posterior Mean Heatmap")
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Plot the posterior standard deviation heatmap
plt.figure(figsize=(8, 6))
ax2 = sns.heatmap(Z_std, cbar_kws={'label': 'Predicted Value'})
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels)
ax2.set_yticks(tick_positions)
ax2.set_yticklabels(tick_labels)
plt.title("GP Posterior Standard Deviation Heatmap")
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Make a kernel density estimate of the z-scores of the residuals
y_pred, y_std = gp.predict(X, return_std=True)
residuals = y - y_pred
z_scores = residuals / y_std

sns.kdeplot(z_scores, bw_method='scott')
plt.title("KDE of Z-Scores of Residuals")
plt.xlabel("Z-Score")
plt.ylabel("Density")
plt.show()