import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial import distance_matrix
from scipy.stats import wasserstein_distance
from tqdm import tqdm

def calculate_kde(cell_coords, bandwidth=2, grid_size=10):
    """
    Calculate kernel density estimation on a grid
    
    Parameters:
    cell_coords (numpy.ndarray): Array of coordinates with shape (n_samples, 2)
    bandwidth (float): Bandwidth for kernel density estimation
    grid_size (int): Size of the grid for density estimation
    
    Returns:
    tuple: (grid coordinates, density values)
    """
    X = cell_coords[:, 1]
    Y = cell_coords[:, 0]
    x = np.linspace(X.min(), X.max(), int((X.max() - X.min()) / grid_size))
    y = np.linspace(Y.min(), Y.max(), int((Y.max() - Y.min()) / grid_size))
    X_grid, Y_grid = np.meshgrid(x, y)
    xy = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    kde = KernelDensity(bandwidth=bandwidth).fit(cell_coords)
    density = np.exp(kde.score_samples(xy))

    return xy, density

def calculate_morans_i(xy, density, threshold_distance=50):
    """
    Calculate Moran's I statistic for spatial autocorrelation
    
    Parameters:
    xy (numpy.ndarray): Grid coordinates with shape (n_grid_points, 2)
    density (numpy.ndarray): Density values corresponding to the grid coordinates
    threshold_distance (float): Threshold distance for constructing the weight matrix
    
    Returns:
    float: Computed Moran's I statistic
    """
    dist_matrix = distance_matrix(xy, xy)
    W = np.where(dist_matrix <= threshold_distance, 1, 0)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    W = W / row_sums

    x_bar = np.mean(density)
    n = len(density)
    numerator = np.sum(W * np.outer(density - x_bar, density - x_bar))
    denominator = np.sum((density - x_bar) ** 2)
    morans_i = (n / np.sum(W)) * (numerator / denominator)

    return morans_i

def calculate_wasserstein_distance(xy, density, unique_genes):
    # Calculate Wasserstein distances
    w_dis = np.zeros((len(unique_genes), len(unique_genes)))
    for i in tqdm(range(len(unique_genes)), desc="Calculating Wasserstein distances"):
        for n in range(len(unique_genes)):
            w_dist = wasserstein_distance(np.arange(xy.shape[0]), np.arange(xy.shape[0]),
                                          density[i], density[n])
            w_dis[i][n] = w_dist

    return w_dis

# from scipy.spatial.distance import correlation

# def calculate_correlation_distance(xy, density, unique_genes):
#     # Calculate Correlation distances
#     corr_dis = np.zeros((len(unique_genes), len(unique_genes)))
#     for i in tqdm(range(len(unique_genes)), desc="Calculating Correlation distances"):
#         for n in range(len(unique_genes)):
#             corr_dist = correlation(np.arange(xy.shape[0]), np.arange(xy.shape[0]),
#                                     density[i], density[n])
#             corr_dis[i][n] = corr_dist
#     return corr_dis
