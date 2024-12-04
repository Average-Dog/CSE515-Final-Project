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


def calculate_gap_single_run(initial_data, new_points, f_min, max_observations):
    gaps = []
    for i in range(1, max_observations + 1):
        truncated_new_points = new_points.iloc[:i]
        f_best_initial = initial_data['values'].min()
        f_best_found = pd.concat([initial_data, truncated_new_points])['values'].min()
        denominator = f_min - f_best_initial
        if abs(denominator) < 1e-8:
            gaps.append(1)
        else:
            gap = (f_best_found - f_best_initial) / denominator
            gaps.append(gap)
    return gaps


def calculate_average_gap_multiple_runs(runs_data, f_min, max_observations=30):
    all_gaps = []
    for initial_data, new_points in runs_data:
        gaps = calculate_gap_single_run(initial_data, new_points, f_min, max_observations)
        all_gaps.append(gaps)
    average_gaps = np.mean(all_gaps, axis=0)
    return average_gaps


# Expected Improvement (EI) function
def CEI(X, gp_model, y_best, xi=0.01, minimize=True):
    mu, sigma = gp_model.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    if minimize:
        mu = -mu
        y_best = -y_best
    z = (y_best - mu - xi) / sigma
    cei = sigma * (z * norm.cdf(z) + norm.pdf(z))
    return cei


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
        cei_values = CEI(grid_points, gp_model, y_best, minimize=True)
        max_indices = np.argsort(cei_values)[-batch_size:]
        next_points = grid_points[max_indices]
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


def find_min_on_grid(func):
    bounds = [[-2, 2], [-2, 2]]
    resolution = 1000
    x1 = np.linspace(bounds[0][0], bounds[0][1], resolution)
    x2 = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.c_[X1.ravel(), X2.ravel()]
    values = np.array([func(x1, x2) for x1, x2 in grid_points])
    f_min = np.min(values)
    return f_min


if __name__ == "__main__":
    # Perform Bayesian optimization with batch selection
    bayesian_runs = [bayesian_optimization_with_batch(num_initial=5, num_iterations=30, batch_size=5) for _ in
                     range(20)]
    random_runs = [bayesian_optimization_with_batch(num_initial=5, num_iterations=30, batch_size=5) for _ in
                   range(20)]  # Replace with actual random search runs

    # Example usage of paired t-test and p-value calculation
    f_min = find_min_on_grid(goldstein_price)
    observation_counts = [30, 60, 90, 120, 150]
    p_values = {}
    for obs_count in observation_counts:
        p_value = paired_t_test_and_p_value(bayesian_runs, random_runs, f_min, obs_count)
        p_values[obs_count] = p_value

    # Print results
    print("\nResults for Goldstein-Price dataset:")
    for obs_count, p_val in p_values.items():
        print(f"Observations: {obs_count}, p-value: {p_val:.4f}")
    if any(p_val > 0.05 for p_val in p_values.values()):
        speedup_obs = next(obs_count for obs_count, p_val in p_values.items() if p_val > 0.05)
        print(f"Random search needs at least {speedup_obs} observations to reach p-value > 0.05.")
    else:
        print("Random search does not reach p-value > 0.05 within the given observation counts.")

# Results for Goldstein-Price dataset:
# Observations: 30, p-value: nan
# Observations: 60, p-value: nan
# Observations: 90, p-value: nan
# Observations: 120, p-value: nan
# Observations: 150, p-value: nan
# Random search does not reach p-value > 0.05 within the given observation counts.
#
# Process finished with exit code 0