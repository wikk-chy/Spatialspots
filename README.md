# Usage Example

This document explains how to use three functions to calculate kernel density estimation (KDE), Moran's I statistic, and Wasserstein distance.

## Moran's I
Calculate Moran's I for gene or cell distribution.

## Wasserstein metric
Calculate the distribution distance between spots using the Wasserstein metric.

## Correlation

## Function Imports

First, we need to import the necessary libraries and functions.

```python
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Import custom functions
from your_module import calculate_kde, calculate_morans_i, calculate_wasserstein_distance

# Generate sample data
data = {
    "dim_1": np.random.rand(100),
    "dim_2": np.random.rand(100),
    "gene": np.random.choice(["geneA", "geneB", "geneC"], 100)
}
df = pd.DataFrame(data)
genes = ["geneA", "geneB", "geneC"]

# Calculate Kernel Density Estimation (KDE)
xy, densities = calculate_kde(df, genes, bandwidth=2, grid_size=10)

# Calculate Moran's I Statistic
morans_i_values = []
for density in densities:
    morans_i = calculate_morans_i(xy, density, threshold_distance=50)
    morans_i_values.append(morans_i)

# Calculate Wasserstein Distance
w_dis = calculate_wasserstein_distance(densities, xy, genes)


Ensure that the three custom functions `calculate_kde`, `calculate_morans_i`, and `calculate_wasserstein_distance` are included in `your_module`.

With the above example code, users can easily understand how to import and use these three functions to calculate kernel density estimation, Moran's I statistic, and Wasserstein distance.

