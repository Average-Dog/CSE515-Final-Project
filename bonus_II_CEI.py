import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, WhiteKernel, DotProduct
from scipy.stats import ttest_rel
from scipy.stats import norm
from scipy.stats import kstest
#1.Make a heatmap of the value of the Goldstein–Price function
def goldstein_price(x1, x2):
    term1 = (1 + ((x1 + x2 + 1) ** 2) * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2))
    term2 = (30 + ((2 * x1 - 3 * x2) ** 2) * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))
    return term1 * term2

def plot_goldstein_price():

    # Compute the values for the grid
    x1 = np.linspace(-2, 2, 1000)
    x2 = np.linspace(-2, 2, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Z = goldstein_price(X1, X2)
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(Z, cbar_kws={'label': 'Value'})
    tick_positions = np.linspace(0, len(x1) - 1, 5)
    tick_labels = [-2, -1, 0, 1, 2]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    plt.title("Heatmap of the Goldstein–Price Function")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
#2.The Goldstein–Price function is not stationary.
# Its values vary dramatically throughout its domain, characterized by regions of sharp peaks.
# Regions with high function values are near (x1,x2)=(-2,2) or (2,-2), where the function will have a very high value.
#That is, the behavior of the function appears not to be constant throughout the domain
#3.Can you find a transformation of the data that makes it more stationary?
#Picheny et al. (2012) use the following logarithmic form of the Goldstein-Price function, on [0, 1]**2
# Define the transformed Goldstein–Price function
def plot_transformed_goldstein_price():
    # Define the transformed Goldstein–Price function
    def transformed_goldstein_price(x1, x2):
        x1_bar = 4 * x1 - 2
        x2_bar = 4 * x2 - 2
        term1 = (1 + (x1_bar + x2_bar + 1) ** 2 * (
                    19 - 14 * x1_bar + 3 * x1_bar ** 2 - 14 * x2_bar + 6 * x1_bar * x2_bar + 3 * x2_bar ** 2))
        term2 = (30 + (2 * x1_bar - 3 * x2_bar) ** 2 * (
                    18 - 32 * x1_bar + 12 * x1_bar ** 2 + 48 * x2_bar - 36 * x1_bar * x2_bar + 27 * x2_bar ** 2))
        return (1 / 2.427) * np.log(term1 * term2) - 8.693
    x1 = np.linspace(-2, 2, 1000)
    x2 = np.linspace(-2, 2, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Z_transformed = transformed_goldstein_price(X1, X2)
    # Plot the heatmap for the transformed function
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(Z_transformed, cbar_kws={'label': 'Transformed Value'})
    tick_positions = np.linspace(0, len(x1) - 1, 5)
    tick_labels = [-2,-1,0,1,2]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    plt.title("Heatmap of the Transformed Goldstein–Price Function")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
#4.Make a kernel density estimate of the distribution of the values for the lda and svm benchmarks
def KDE_lda_and_svm(lda_file_path, svm_file_path):
    # Load data
    lda_data = pd.read_csv(lda_file_path)
    svm_data = pd.read_csv(svm_file_path)
    lda_target_values = pd.to_numeric(lda_data.iloc[:, 3], errors='coerce').dropna().values
    svm_target_values = pd.to_numeric(svm_data.iloc[:, 3], errors='coerce').dropna().values
    sns.set()
    # Plot KDE for LDA
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(data=lda_target_values, bw_method='scott', ax=ax, color="blue", label="LDA KDE")
    ax.plot(lda_target_values, np.zeros_like(lda_target_values), 'o', markersize=5, color='blue', alpha=0.7, label="LDA Points")
    ax.set_title("LDA Kernel Density Estimate")
    ax.set_xlabel("Benchmark Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.show()
    # Plot KDE for SVM
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(data=svm_target_values, bw_method='scott', ax=ax, color="green", label="SVM KDE")
    ax.plot(svm_target_values, np.zeros_like(svm_target_values), 'o', markersize=5, color='green', alpha=0.7, label="SVM Points")
    ax.set_title("SVM Kernel Density Estimate")
    ax.set_xlabel("Benchmark Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.show()

def transformed_KDE(lda_file_path, svm_file_path):
    lda_data = pd.read_csv(lda_file_path)
    svm_data = pd.read_csv(svm_file_path)
    lda_target_values = pd.to_numeric(lda_data.iloc[:, 3], errors='coerce').dropna().values
    svm_target_values = pd.to_numeric(svm_data.iloc[:, 3], errors='coerce').dropna().values
    power_transformer = PowerTransformer(method='box-cox', standardize=True)
    lda_transformed = power_transformer.fit_transform(lda_target_values.reshape(-1, 1)).flatten()
    svm_transformed = power_transformer.fit_transform(svm_target_values.reshape(-1, 1)).flatten()
    sns.set()
    # Plot KDE for LDA
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(data=lda_transformed, bw_method='scott', ax=ax, color="blue", label="LDA KDE")
    ax.plot(lda_transformed, np.zeros_like(lda_transformed), 'o', markersize=5, color='blue', alpha=0.7,
            label="LDA Points")
    ax.set_title("LDA Kernel Density Estimate")
    ax.set_xlabel("Benchmark Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.show()
    # Plot KDE for SVM
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(data=svm_transformed, bw_method='scott', ax=ax, color="green", label="SVM KDE")
    ax.plot(svm_transformed, np.zeros_like(svm_transformed), 'o', markersize=5, color='green', alpha=0.7,
            label="SVM Points")
    ax.set_title("SVM Kernel Density Estimate")
    ax.set_xlabel("Benchmark Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.show()

# Model fitting
def goldstein_price_dataset():
    sobol = qmc.Sobol(d=2, scramble=False)
    points = sobol.random_base2(m=5)
    scaled_pts = qmc.scale(points, l_bounds=[-2, -2], u_bounds=[2, 2])
    values = np.array([goldstein_price(x1, x2) for x1, x2 in scaled_pts])
    D = pd.DataFrame(scaled_pts, columns=['x1', 'x2'])
    D['data'] = values

    return D

# Fit a Gaussian process model to the data
def fit_gaussian_process(D):
    X = D[['x1', 'x2']]
    y = D['data']
    kernel = ConstantKernel() * RBF()
    gp = GPR(kernel= kernel, alpha=0.001, normalize_y=True)
    gp.fit(X, y)
    log_marginal_likelihood = gp.log_marginal_likelihood()
    marginal_likelihood = 2**log_marginal_likelihood
    print("Marginal Likelihood: ", marginal_likelihood)

    return gp

# Plot the posterior mean and standard deviation heatmaps
def gp_heatmap(D, gp_model):
    x1 = np.linspace(-2, 2, 1000)
    x2 = np.linspace(-2, 2, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.c_[X1.ravel(), X2.ravel()]

    y_mean, y_std = gp_model.predict(grid_points, return_std=True)
    print(min(y_std))
    Z_mean = y_mean.reshape(X1.shape)
    Z_std = y_std.reshape(X1.shape)

    # mean heatmap
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

    # std heatmap
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

#kenerl density
def kde_z_scores(D, gp_model):
    x1 = D['x1']
    x2 = D['x2']
    X = np.c_[x1, x2]
    y_true = D['data']
    y_pred, y_std = gp_model.predict(X, return_std=True)
    residuals = y_true - y_pred
    z_scores = residuals / y_std

    sns.kdeplot(z_scores, bw_method='scott')
    plt.title("KDE of Z-Scores of Residuals")
    plt.xlabel("Z-Score")
    plt.ylabel("Density")
    plt.show()

#Repeat the above using a log transformation to the output of the Goldstein–Price function. Does the marginal likelihood improve? Does the model appear better calibrated?
def log_goldstein_price(x1, x2):
    return np.log(goldstein_price(x1, x2))

def generate_log_dataset():
    sobol = qmc.Sobol(d=2, scramble=False)
    points = sobol.random_base2(m=5)
    scaled_pts = qmc.scale(points, l_bounds=[-2, -2], u_bounds=[2, 2])
    values = np.array([log_goldstein_price(x1, x2) for x1, x2 in scaled_pts])
    log_D = pd.DataFrame(scaled_pts, columns=['x1', 'x2'])
    log_D['data'] = values
    return log_D

# Fit a Gaussian process model to the data
def fit_log_gaussian_process(log_D):
    X = log_D[['x1', 'x2']]
    y = log_D['data']
    kernel = ConstantKernel() * RBF()
    log_gp = GPR(kernel= kernel, alpha=0.001, normalize_y=True)
    log_gp.fit(X, y)
    log_marginal_likelihood = log_gp.log_marginal_likelihood()
    marginal_likelihood = 2**log_marginal_likelihood
    print("Log Marginal Likelihood: ", marginal_likelihood)

    return log_gp

# Plot the posterior mean and standard deviation heatmaps
def log_gp_heatmap(log_D, log_gp):
    x1 = np.linspace(-2, 2, 1000)
    x2 = np.linspace(-2, 2, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.c_[X1.ravel(), X2.ravel()]

    y_mean, y_std = log_gp.predict(grid_points, return_std=True)
    print(min(y_std))
    Z_mean = y_mean.reshape(X1.shape)
    Z_std = y_std.reshape(X1.shape)

    # mean heatmap
    plt.figure(figsize=(8, 6))
    ax1 = sns.heatmap(Z_mean, cbar_kws={'label': 'Predicted Value'})
    tick_positions = np.linspace(0, len(x1) - 1, 5)
    tick_labels = [-2, -1, 0, 1, 2]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels)
    plt.title("GP Posterior Mean Heatmap for Log Transformed Goldstein-Price Function")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    # std heatmap
    plt.figure(figsize=(8, 6))
    ax2 = sns.heatmap(Z_std, cbar_kws={'label': 'Predicted Value'})
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels)
    plt.title("GP Posterior Standard Deviation Heatmap for Log Transformed Goldstein-Price Function")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#kenerl density
def log_kde_z_scores(log_D, log_gp):
    x1 = log_D['x1']
    x2 = log_D['x2']
    X = np.c_[x1, x2]
    y_true = log_D['data']
    y_pred, y_std = log_gp.predict(X, return_std=True)
    residuals = y_true - y_pred
    z_scores = residuals / y_std
    sns.kdeplot(z_scores, bw_method='scott')
    plt.title("KDE of Z-Scores of Residuals for Log Transformed Goldstein-Price Function")
    plt.xlabel("Z-Score")
    plt.ylabel("Density")
    plt.show()

# Bayesian information criterion (bic) ---------------------------------------------------------------------------------
def BIC(gp, D):
    k = gp.kernel_.n_dims
    n = len(D)
    L = gp.log_marginal_likelihood()
    bic = k * np.log(n) - 2 * L
    return bic

# What is the best model you found and its bic score?
def best_model(D):
    X = D[['x1', 'x2']]
    y = D['data']
    models = [
        (ConstantKernel() * RBF(), "RBF"),
        (ConstantKernel() * Matern(), "Matern"),
        (ConstantKernel() * RationalQuadratic(), "RationalQuadratic"),
        (ConstantKernel() * WhiteKernel(), "WhiteKernel"),
        (ConstantKernel() * DotProduct(), "DotProduct")
    ]
    best_bic = float('inf')
    best_model = None
    best_kernel = None

    for kernel, name in models:
        gp = GPR(kernel=kernel, alpha=0.001, normalize_y=True)
        gp.fit(X, y)
        bic = BIC(gp, D)
        if bic < best_bic:
            best_bic = bic
            best_model = gp
            best_kernel = name

    print(f"Best Model: {best_kernel}, BIC score: {best_bic}")

# Performasimilarsearchforthesvmandldadatasets
def file_search(file_path):
    lda_data = pd.read_csv(file_path)
    X = pd.to_numeric(lda_data.iloc[:, 3], errors='coerce').dropna().values.reshape(-1, 1)
    y = np.zeros_like(X)

    models = [
        (ConstantKernel() * RBF(), "RBF"),
        (ConstantKernel() * Matern(), "Matern"),
        (ConstantKernel() * RationalQuadratic(), "RationalQuadratic"),
        (ConstantKernel() * WhiteKernel(), "WhiteKernel"),
        (ConstantKernel() * DotProduct(), "DotProduct")
    ]
    best_bic = float('inf')
    best_model = None
    best_kernel = None

    for kernel, name in models:
        gp = GPR(kernel=kernel, alpha=0.001, normalize_y=True)
        gp.fit(X, y)
        bic = BIC(gp, D)
        if bic < best_bic:
            best_bic = bic
            best_model = gp
            best_kernel = name

    print(f"For {file_path}, best Model: {best_kernel}, BIC score: {best_bic}")
    return best_kernel, best_bic, best_model
#bayesian optimazation
def fit_gaussian_process_with_different_kernel(D,kernel_1):
    X = D[['x1', 'x2']]
    y = D['data']
    kernel_1=kernel_1
    kernel = ConstantKernel() * kernel_1
    gp = GPR(kernel= kernel, alpha=0.001, normalize_y=True)
    gp.fit(X, y)
    log_marginal_likelihood = gp.log_marginal_likelihood()
    marginal_likelihood = np.exp(log_marginal_likelihood)
    return gp
# Plot the posterior mean and standard deviation heatmaps
def gp_heatmap(D, gp_model):
    x1 = np.linspace(-2, 2, 500)
    x2 = np.linspace(-2, 2, 500)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.c_[X1.ravel(), X2.ravel()]

    y_mean, y_std = gp_model.predict(grid_points, return_std=True)
    print(min(y_std))
    Z_mean = y_mean.reshape(X1.shape)
    Z_std = y_std.reshape(X1.shape)
    # mean heatmap
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

    # std heatmap
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
def CEI(X, gp_model, y_best, xi=0.01, minimize=True):
    mu, sigma = gp_model.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    if minimize:
        mu = -mu
        y_best = -y_best
    z = (y_best - mu - xi) / sigma
    cei = sigma * (z * norm.cdf(z) + norm.pdf(z))
    return cei
def gp_heatmap_for_training_points(gp_model, D):
    X = D[['x1', 'x2']].values
    y = D['data'].values
    y_mean, y_std = gp_model.predict(X, return_std=True)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_mean, s=100)
    plt.colorbar(scatter, label='Posterior Mean')
    plt.title("Posterior Mean at Training Points")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_std, s=100)
    plt.colorbar(scatter, label='Posterior Standard Deviation')
    plt.title("Posterior Standard Deviation at Training Points")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def plot_cei_for_full_region(gp_model, y_best):
    x1 = np.linspace(-2, 2, 1000)
    x2 = np.linspace(-2, 2, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.c_[X1.ravel(), X2.ravel()]
    cei_values = CEI(grid_points, gp_model, y_best, minimize=False)
    max_idx = np.argmax(cei_values)
    max_cei_point = grid_points[max_idx]
    max_cei = cei_values[max_idx]
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(grid_points[:, 0], grid_points[:, 1], c=cei_values, s=1)
    plt.colorbar(scatter, label='Constrained Expected Improvement (CEI)')
    plt.scatter(max_cei_point[0], max_cei_point[1], color='red', s=150, label='Max CEI', marker='X')
    plt.title("Constrained Expected Improvement (CEI) in Full Region")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()
    print(f"Maximum CEI: {max_cei:.4f} at point X1={max_cei_point[0]:.3f}, X2={max_cei_point[1]:.3f}")
    return max_cei_point

def labeled_data_for_file(file_path, num_initial=5, num_iterations=30,random_seed=42,kernel_1=Matern()):
    np.random.seed(random_seed)
    data = pd.read_csv(file_path).iloc[:, :-1]
    initial_indices = np.random.choice(data.index, size=num_initial, replace=False)
    initial_data = data.loc[initial_indices]
    labeled_data = initial_data.copy()
    unlabeled_data = data.drop(index=initial_indices)
    kernel = ConstantKernel() * kernel_1
    gp_model = GPR(kernel=kernel, alpha=0.001, normalize_y=True)
    new_points = []
    for iteration in range(num_iterations):
        X_labeled = labeled_data.iloc[:, :3].values
        y_labeled = labeled_data.iloc[:, 3].values
        gp_model.fit(X_labeled, y_labeled)
        y_best = np.min(y_labeled)
        X_unlabeled = unlabeled_data.iloc[:, :-1].values
        ei_values = CEI(X_unlabeled, gp_model, y_best, minimize=False)
        max_idx = np.argmax(ei_values)
        next_point = unlabeled_data.iloc[max_idx]
        new_points.append(next_point)
        labeled_data = pd.concat([labeled_data, next_point.to_frame().T], ignore_index=True)
        unlabeled_data = unlabeled_data.drop(index=next_point.name)
    new_points_df = pd.DataFrame(new_points)
    new_points_df.columns = ['x1', 'x2', 'x3', 'values']
    initial_data.columns = ['x1', 'x2', 'x3', 'values']
    return initial_data, new_points_df
def labeled_data_for_Goldstein(num_initial=5, num_iterations=30, random_seed=42):
    np.random.seed(random_seed)
    initial_points = np.random.uniform(-2, 2, size=(num_initial, 2))
    initial_values = np.array([goldstein_price(x1, x2) for x1, x2 in initial_points])
    initial_data = pd.DataFrame(initial_points, columns=['x1', 'x2'])
    initial_data['values'] = initial_values
    kernel = ConstantKernel() * RBF()
    gp_model = GPR(kernel=kernel, alpha=0.001, normalize_y=True)
    new_points = []
    labeled_data = initial_data.copy()
    for iteration in range(num_iterations):
        X_labeled = labeled_data[['x1', 'x2']].values
        y_labeled = labeled_data['values'].values
        gp_model.fit(X_labeled, y_labeled)
        x1 = np.linspace(-2, 2, 1000)
        x2 = np.linspace(-2, 2, 1000)
        X1, X2 = np.meshgrid(x1, x2)
        candidates = np.c_[X1.ravel(), X2.ravel()]
        y_best = np.min(y_labeled)
        ei_values = CEI(candidates, gp_model, y_best, minimize=False)
        max_idx = np.argmax(ei_values)
        next_point = candidates[max_idx]
        next_value = goldstein_price(next_point[0], next_point[1])
        new_points.append([next_point[0], next_point[1], next_value])
        new_row = pd.DataFrame([[next_point[0], next_point[1], next_value]], columns=labeled_data.columns)
        labeled_data = pd.concat([labeled_data, new_row], ignore_index=True)
    new_points_df = pd.DataFrame(new_points, columns=['x1', 'x2', 'values'])
    return initial_data, new_points_df
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
def gap(initial_data, new_points, f_min):
    f_best_initial = initial_data['values'].min()
    f_best_found = pd.concat([initial_data, new_points])['values'].min()
    gap = (f_best_found - f_best_initial) / (f_min - f_best_initial)
    return gap
def labeled_data_optimization_with_baseline(
        func=None,
        file_path=None,
        num_initial=5,
        num_iterations=30,
        random_search_budget=150,
        num_runs=20,
        kernel_1=Matern()
):
    bounds = np.array([[-2, 2], [-2, 2]]) if func is not None else None
    bayesian_runs = []
    random_search_runs = []
    for run in range(num_runs):
        random_seed = np.random.randint(0, 100000)
        if file_path:
            initial_data_bayes, new_points_bayes = labeled_data_for_file(
                file_path, num_initial, num_iterations, random_seed=random_seed, kernel_1=kernel_1
            )
        elif func is not None:
            initial_data_bayes, new_points_bayes = labeled_data_for_Goldstein(
                num_initial, num_iterations, random_seed=random_seed
            )
        else:
            raise ValueError("Either `func` or `file_path` must be provided.")
        bayesian_runs.append((initial_data_bayes, new_points_bayes))

        labeled_data_random = initial_data_bayes.copy()
        new_points_random = []
        for _ in range(random_search_budget - len(initial_data_bayes)):
            if func is not None:
                random_point = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, bounds.shape[0]))
                random_value = func(*random_point[0])
                new_points_random.append([*random_point[0], random_value])
            else:
                data=read_csv(file_path)
                file_without_last_col = data.iloc[:, :-1]
                random_idx = np.random.choice(file_without_last_col.shape[0], size=1)
                random_row = file_without_last_col.iloc[random_idx].values.flatten().tolist()
                new_points_random.append(random_row)
        new_points_random_df = pd.DataFrame(new_points_random, columns=labeled_data_random.columns)
        random_search_runs.append((initial_data_bayes, new_points_random_df))

    return bayesian_runs, random_search_runs
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

def plot_learning_curves_per_dataset(
    dataset_name,
    bayesian_runs,
    random_runs,
    f_min,
    max_observations=30
):

    bayesian_avg_gaps = calculate_average_gap_multiple_runs(bayesian_runs, f_min, max_observations)
    random_avg_gaps = calculate_average_gap_multiple_runs(random_runs, f_min, max_observations)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_observations + 1), bayesian_avg_gaps, label="Bayesian Optimization", marker='o')
    plt.plot(range(1, max_observations + 1), random_avg_gaps, label="Random Search", marker='s', color='orange')
    plt.title(f"Learning Curves - {dataset_name}")
    plt.xlabel("Number of Observations")
    plt.ylabel("Average Gap")
    plt.grid()
    plt.legend()
    plt.show()
def paired_t_test_and_speedup(bayesian_runs, random_runs, f_min, max_observations_list):
    p_values = {}
    speedup_observations = None
    for max_obs in max_observations_list:
        bayesian_gaps = calculate_average_gap_multiple_runs(bayesian_runs, f_min, max_observations=max_obs)
        random_gaps = calculate_average_gap_multiple_runs(random_runs, f_min, max_observations=max_obs)
        t_stat, p_value = ttest_rel(bayesian_gaps, random_gaps)
        p_values[max_obs] = p_value
        if speedup_observations is None and p_value > 0.05:
            speedup_observations = max_obs
    return p_values, speedup_observations

if __name__ == "__main__":
    #Data visualization
    plot_goldstein_price()
    plot_transformed_goldstein_price()

    KDE_lda_and_svm("lda.csv","svm.csv")
    transformed_KDE("lda.csv","svm.csv")

    # Model fitting
    D = goldstein_price_dataset()
    gp = fit_gaussian_process(D)
    gp_heatmap(D, gp)
    kde_z_scores(D, gp)

    log_D = generate_log_dataset()
    log_gp = fit_log_gaussian_process(log_D)
    log_gp_heatmap(log_D, log_gp)
    log_kde_z_scores(log_D, log_gp)

    #normal Goldstein-Price function:
    #no systematic errors
    #make sense, make sense, no
    #yes

    #log transformed Goldstein-Price function:
    #systematic errors
    #dont make sense, dont make sense, no
    #yes

    #Goldstein-Price function marginal likehood: 0.0006025128589123176
    #log transformed Goldstein-Price function marginal likehood: 2.1449741568070014e-14
    # The log transformed Goldstein-Price function has a lower marginal likelihood than the original Goldstein-Price function.

    bic = BIC(gp, D)
    print(f"BIC Score: {bic}")
    # BIC Score: 28.324912500963972

    best_model(D)
    # Best Model: RBF, BIC score: 28.324912500963972

    file_search('svm.csv')
    #For svm.csv, best Model: Matern, BIC score: 61.06260490629114

    file_search('lda.csv')
    #For lda.csv, best Model: RationalQuadratic, BIC score: 84.95449518671941

    #bayesian optimization
    D = goldstein_price_dataset()
    gp_model=fit_gaussian_process_with_different_kernel(D,RBF())
    gp_heatmap_for_training_points(gp_model, D)
    y_best = np.min(D['data'])
    max_ei_point = plot_cei_for_full_region(gp_model, y_best)
    print(f"Recommended next observation point: {max_ei_point}")
    #it seems like a good next observation location
    #Maximum EI: 627904.7684 at point X1 = -1.123, X2 = 0.182
     # svm
    svm_initial_data,svm_labeled_data = labeled_data_for_file('svm.csv', num_initial=5, num_iterations=30)
    print("Initial Data in SVM")
    print(svm_initial_data)
    print("Labeled data in SVM:")
    print(svm_labeled_data)
    svm_data = pd.read_csv('svm.csv')
    f_min_svm = svm_data.iloc[:, 3].min()
    svm_labeled_data_gap = gap(svm_initial_data,svm_labeled_data, f_min_svm)
    print("Gap for SVM:")
    print(svm_labeled_data_gap)
    # Gap for SVM:0.5547493403693948
    # lda
    lda_initial_data,lda_labeled_data = labeled_data_for_file('lda.csv', num_initial=5, num_iterations=30,kernel_1=RationalQuadratic())
    print("Initial Data in LDA")
    print(lda_initial_data)
    print("Labeled data in LDA:")
    print(lda_labeled_data)
    lda_data = pd.read_csv('lda.csv')
    f_min_lda = lda_data.iloc[:, 3].min()
    lda_labeled_data_gap = gap(lda_initial_data,lda_labeled_data, f_min_lda)
    print("Gap for LDA:")
    print(lda_labeled_data_gap)
    # Gap for LDA:1.0
    # Goldstein
    Goldstein_initial_data,Goldstein_labeled_data = labeled_data_for_Goldstein(num_initial=5, num_iterations=30)
    print("Initial Data in Goldstein:")
    print(Goldstein_initial_data)
    print("Labeled data in Goldstein–Price:")
    print(Goldstein_labeled_data)
    f_min = find_min_on_grid(goldstein_price)
    Goldstein_labeled_data_gap = gap(Goldstein_initial_data,Goldstein_labeled_data, f_min)
    print("Gap for Goldstein–Price:")
    print(Goldstein_labeled_data_gap)
    # Gap for Goldstein–Price:0.9993656638297461
    Goldstein_bayesian_runs, Goldstein_random_search_runs = labeled_data_optimization_with_baseline(
        func=goldstein_price,
        num_initial=5,
        num_iterations=30,
        random_search_budget=150,
        num_runs=20
    )
    SVM_bayesian_runs, SVM_random_search_runs = labeled_data_optimization_with_baseline(
        file_path=r"C:\Users\Lenovo\Desktop\svm.csv",
        num_initial=5,
        num_iterations=30,
        random_search_budget=150,
        num_runs=20,
        kernel_1=Matern()
    )
    LDA_bayesian_runs, LDA_random_search_runs = labeled_data_optimization_with_baseline(
        file_path=r"C:\Users\Lenovo\Desktop\lda.csv",
        num_initial=5,
        num_iterations=30,
        random_search_budget=150,
        num_runs=20,
        kernel_1=RationalQuadratic()
    )
    Goldstein_bayesian_runs, Goldstein_random_search_runs = labeled_data_optimization_with_baseline(
        func=goldstein_price,
        num_initial=5,
        num_iterations=30,
        random_search_budget=150,
        num_runs=20
    )
    SVM_bayesian_runs, SVM_random_search_runs = labeled_data_optimization_with_baseline(
        file_path=r"C:\Users\Lenovo\Desktop\svm.csv",
        num_initial=5,
        num_iterations=30,
        random_search_budget=150,
        num_runs=20,
        kernel_1=Matern()
    )
    LDA_bayesian_runs, LDA_random_search_runs = labeled_data_optimization_with_baseline(
        file_path=r"C:\Users\Lenovo\Desktop\lda.csv",
        num_initial=5,
        num_iterations=30,
        random_search_budget=150,
        num_runs=20,
        kernel_1=RationalQuadratic()
    )
    print(f"Goldstein Bayesian Runs: {len(Goldstein_bayesian_runs)}")
    print(f"SVM Bayesian Runs: {len(SVM_bayesian_runs)}")
    print(f"LDA Bayesian Runs: {len(LDA_bayesian_runs)}")
    lda_data = pd.read_csv(r"C:\Users\Lenovo\Desktop\lda.csv")
    f_min_lda = lda_data.iloc[:, 3].min()
    svm_data = pd.read_csv(r"C:\Users\Lenovo\Desktop\svm.csv")
    f_min_svm = svm_data.iloc[:, 3].min()
    f_min_goldstein = find_min_on_grid(goldstein_price)
    plot_learning_curves_per_dataset(
        dataset_name="Goldstein-Price",
        bayesian_runs=Goldstein_bayesian_runs,
        random_runs=Goldstein_random_search_runs,
        f_min=f_min_goldstein,
        max_observations=30
    )
    plot_learning_curves_per_dataset(
        dataset_name="SVM",
        bayesian_runs=SVM_bayesian_runs,
        random_runs=SVM_random_search_runs,
        f_min=f_min_svm,
        max_observations=30
    )
    plot_learning_curves_per_dataset(
        dataset_name="LDA",
        bayesian_runs=LDA_bayesian_runs,
        random_runs=LDA_random_search_runs,
        f_min=f_min_lda,
        max_observations=30
    )
    # Observation counts to evaluate
    observation_counts = [30, 60, 90, 120, 150]
    datasets = [
        ("Goldstein-Price", Goldstein_bayesian_runs, Goldstein_random_search_runs, f_min_goldstein),
        ("SVM", SVM_bayesian_runs, SVM_random_search_runs, f_min_svm),
        ("LDA", LDA_bayesian_runs, LDA_random_search_runs, f_min_lda)
    ]

    for dataset_name, bayesian_runs, random_runs, f_min in datasets:
        print(f"\nMean gaps for {dataset_name} dataset:")
        for count in observation_counts:
            bayesian_gaps = calculate_average_gap_multiple_runs(bayesian_runs, f_min, max_observations=count)
            random_gaps = calculate_average_gap_multiple_runs(random_runs, f_min, max_observations=count)
            print(f"For {count} observations:")
            print(f"  Bayesian Optimization (EI): {np.mean(bayesian_gaps):.4f}")
            print(f"  Random Search: {np.mean(random_gaps):.4f}")
    observation_counts = [30, 60, 90, 120, 150]
    for dataset_name, bayesian_runs, random_runs, f_min in datasets:
        p_values, speedup_obs = paired_t_test_and_speedup(bayesian_runs, random_runs, f_min, observation_counts)

        print(f"\nResults for {dataset_name} dataset:")
        for obs_count, p_val in p_values.items():
            print(f"Observations: {obs_count}, p-value: {p_val:.4f}")
        if speedup_obs:
            print(f"Random search needs at least {speedup_obs} observations to reach p-value > 0.05.")
        else:
            print("Random search does not reach p-value > 0.05 within the given observation counts.")
