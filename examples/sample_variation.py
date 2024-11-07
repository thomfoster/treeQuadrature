from treeQuadrature.example_problems import SimpleGaussian
from treeQuadrature.samplers import McmcSampler, Sampler

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

sampler = McmcSampler(burning=100)
D = 10
N = 2000 # number of samples
num_runs = 5
dimensions = np.arange(1, 17, 1)

if __name__ == '__main__':
    problem = SimpleGaussian(D)
    X, y = sampler.rvs(N, mins=problem.lows, maxs=problem.highs, 
                    f = problem.integrand)
    
    distances = distance_matrix(X, X)

    # Calculate pairwise integrand-value differences
    y_diff = np.abs(y - y.T)  # shape (N, N)

    # Flatten the matrices for plotting
    distances_flat = distances.flatten()
    y_diff_flat = y_diff.flatten()

    # exclude self-differences (distance 0)
    mask = distances_flat != 0
    distances_filtered = distances_flat[mask]
    y_diff_filtered = y_diff_flat[mask]

    # Plot pairwise y-value difference vs. Euclidean distance
    plt.figure(figsize=(10, 6))
    plt.scatter(distances_filtered, y_diff_filtered, alpha=0.5, s=5)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Integrand-Value Difference")
    plt.title("Pairwise Integrand Value Difference vs. Euclidean Distance of Samples")
    plt.grid(True)
    plt.show()