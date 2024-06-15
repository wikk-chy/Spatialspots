import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial import distance_matrix

def calculate_morans_i(cell_coords, bandwidth=2, grid_size=10, threshold_distance=50):
    """
    Calculate Moran's I statistic for spatial autocorrelation
    
    Parameters:
    cell_coords (numpy.ndarray): Array of coordinates with shape (n_samples, 2)
    bandwidth (float): Bandwidth for kernel density estimation
    grid_size (int): Size of the grid for density estimation
    threshold_distance (float): Threshold distance for constructing the weight matrix
    
    Returns:
    float: Computed Moran's I statistic
    """
    # Extract coordinates
    X = cell_coords[:, 1]
    Y = cell_coords[:, 0]

    # Generate grid
    x = np.linspace(X.min(), X.max(), int((X.max() - X.min()) / grid_size))
    y = np.linspace(Y.min(), Y.max(), int((Y.max() - Y.min()) / grid_size))
    X_grid, Y_grid = np.meshgrid(x, y)
    xy = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

    # Kernel density estimation
    kde = KernelDensity(bandwidth=bandwidth).fit(cell_coords)
    density = np.exp(kde.score_samples(xy))

    # Distance matrix and weight matrix
    dist_matrix = distance_matrix(xy, xy)
    W = np.where(dist_matrix <= threshold_distance, 1, 0)

    # Row-normalize the weight matrix and handle potential NaN values
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    W = W / row_sums

    # Calculate Moran's I
    x_bar = np.mean(density)
    n = len(density)
    numerator = np.sum(W * np.outer(density - x_bar, density - x_bar))
    denominator = np.sum((density - x_bar) ** 2)
    morans_i = (n / np.sum(W)) * (numerator / denominator)

    return morans_i