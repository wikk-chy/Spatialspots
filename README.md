# Usage Example

This document explains how to use three functions to calculate kernel density estimation (KDE), Moran's I statistic, and Wasserstein distance.

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

# Moran's I
Calculate Moran's I for gene or cell distribution.

# Wasserstein metric
Calculate the distribution distance between spots using the Wasserstein metric.
