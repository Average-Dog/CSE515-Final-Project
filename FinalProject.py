import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, WhiteKernel, DotProduct

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

if __name__ == "__main__":
    # Data visualization
    # plot_goldstein_price()
    # plot_transformed_goldstein_price()
    #
    # KDE_lda_and_svm("lda.csv","svm.csv")
    # transformed_KDE("lda.csv","svm.csv")

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
    #For svm.csv, best Model: WhiteKernel, BIC score: -7085.828007683744

    file_search('lda.csv')
    #For lda.csv, best Model: WhiteKernel, BIC score: -1448.1235465027933