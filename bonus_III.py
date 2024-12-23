import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_rel
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Define the Goldstein-Price function
def goldstein_price(x1, x2):
    term1 = (1 + ((x1 + x2 + 1) ** 2) * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2))
    term2 = (30 + ((2 * x1 - 3 * x2) ** 2) * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))
    return term1 * term2

# Expected Improvement (EI) function
def EI(X, gp_model, y_best, minimize=True):
    mu, sigma = gp_model.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    if minimize:
        mu = -mu
        y_best = -y_best
    z = (y_best - mu) / sigma
    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
    return ei

# Batch Expected Improvement (q-EI) function
def q_EI(X, gp_model, y_best, q, minimize=True):
    ei_values = EI(X, gp_model, y_best, minimize)
    sorted_indices = np.argsort(-ei_values)  # Sort in descending order
    return X[sorted_indices[:q]]

# Bayesian optimization with batch selection
def bayesian_optimization_with_batch(num_initial=5, num_iterations=30, batch_size=5, random_seed=42):
    np.random.seed(random_seed)
    initial_points = np.random.uniform(-2, 2, size=(num_initial, 2))
    initial_values = np.array([goldstein_price(x1, x2) for x1, x2 in initial_points])
    initial_data = pd.DataFrame(initial_points, columns=['x1', 'x2'])
    initial_data['values'] = initial_values
    labeled_data = initial_data.copy()
    kernel = ConstantKernel() * RBF()
    gp_model = GPR(kernel=kernel, alpha=0.001, normalize_y=True)
    gp_model.fit(initial_data[['x1', 'x2']], initial_data['values'])
    new_points = []

    for iteration in range(num_iterations):
        x1 = np.linspace(-2, 2, 1000)
        x2 = np.linspace(-2, 2, 1000)
        X1, X2 = np.meshgrid(x1, x2)
        grid_points = np.c_[X1.ravel(), X2.ravel()]
        y_best = labeled_data['values'].min()
        next_points = q_EI(grid_points, gp_model, y_best, batch_size, minimize=True)
        next_values = [goldstein_price(x1, x2) for x1, x2 in next_points]
        new_points.extend(list(zip(next_points[:, 0], next_points[:, 1], next_values)))
        new_rows = pd.DataFrame(next_points, columns=['x1', 'x2'])
        new_rows['values'] = next_values
        labeled_data = pd.concat([labeled_data, new_rows], ignore_index=True)
        gp_model.fit(labeled_data[['x1', 'x2']], labeled_data['values'])

    new_points_df = pd.DataFrame(new_points, columns=['x1', 'x2', 'values'])
    return initial_data, new_points_df

# Function to perform paired t-test and calculate p-value
def paired_t_test_and_p_value(bayesian_runs, random_runs, f_min, max_observations):
    bayesian_gaps = calculate_average_gap_multiple_runs(bayesian_runs, f_min, max_observations)
    random_gaps = calculate_average_gap_multiple_runs(random_runs, f_min, max_observations)
    t_stat, p_value = ttest_rel(bayesian_gaps, random_gaps)
    return p_value

if __name__ == "__main__":
    # Perform Bayesian optimization with batch selection
    initial_data, new_points_df = bayesian_optimization_with_batch(num_initial=5, num_iterations=30, batch_size=5)
    print("Initial Data:")
    print(initial_data)
    print("New Points Data:")
    print(new_points_df)

    # Example usage of paired t-test and p-value calculation
    # Assuming bayesian_runs and random_runs are available
    f_min = find_min_on_grid(goldstein_price)
    p_value = paired_t_test_and_p_value(bayesian_runs, random_runs, f_min, max_observations=30)
    print(f"P-value: {p_value}")