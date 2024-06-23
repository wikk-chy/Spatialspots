import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial import distance_matrix
from scipy.stats import wasserstein_distance
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from tqdm import tqdm

def calculate_kde(df, genes, bandwidth=2, grid_size=10):
    """
    Calculate kernel density estimations for each unique gene in the DataFrame over a specified grid.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the data.
    unique_genes (list): List of unique genes to calculate KDE for.
    csv_dir (str): Directory of the CSV file.
    csv_path (str): Path to the CSV file.
    bandwidth (float): Bandwidth for the KDE.
    grid_size (int): Size of the grid division for density estimation.
    
    Returns:
    list: A list of density arrays for each unique gene.
    """
    points = df[["dim_1", "dim_2"]].values
    X = points[:, 1]
    Y = points[:, 0]
    x = np.linspace(X.min(), X.max(), int((X.max() - X.min()) / grid_size))
    y = np.linspace(Y.min(), Y.max(), int((Y.max() - Y.min()) / grid_size))
    X_grid, Y_grid = np.meshgrid(x, y)
    xy = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

    densities = []
    for gene in tqdm(genes):
        gene_points = df[df["gene"] == gene][["dim_1", "dim_2"]].values
        kde = KernelDensity(bandwidth=bandwidth).fit(gene_points)
        density = np.exp(kde.score_samples(xy))
        densities.append(density)

    return xy, densities

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
    tree = KDTree(xy)
    n = len(density)
    pairs = tree.query_pairs(threshold_distance)
    
    row_ind, col_ind = zip(*pairs)
    data = np.ones(len(pairs))
    W = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    W = W + W.T
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    W = W.multiply(1 / row_sums[:, np.newaxis])
    x_bar = np.mean(density)
    density_diff = density - x_bar
    numerator = W.multiply(np.outer(density_diff, density_diff)).sum()
    denominator = np.sum(density_diff ** 2)
    
    morans_i = (n / W.sum()) * (numerator / denominator)

    return morans_i