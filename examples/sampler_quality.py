from treeQuadrature.example_problems import SimpleGaussian
from treeQuadrature.samplers import McmcSampler

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


sampler = McmcSampler(burning=100)
N = 20000 # number of samples
num_runs = 5
dimensions = np.arange(1, 17, 1)

def sample_from_gaussian(D: int,
                         plot: bool=False):
    print(f"Testing D = {D}")
    problem = SimpleGaussian(D)

    X, y = sampler.rvs(N, mins=problem.lows, maxs=problem.highs, 
                    f = problem.integrand)

    if plot:
        distances = np.linalg.norm(X, axis=1)
        # Plot histogram of distances
        plt.figure(figsize=(10, 4))
        plt.hist(distances, bins=30, density=True,
                alpha=0.6, color='b')
        plt.title("Histogram of Distances from Mode (0)" +
                    "\n" + f"D = {D}")
        plt.xlabel("Distance to Mode")
        plt.ylabel("Frequency")
        plt.savefig(f"figures/samples_gaussian/hist_{D}.png")

        # Q-Q plot
        plt.figure(figsize=(10, 4))
        stats.probplot(X.flatten(), dist="norm", plot=plt)
        plt.title("Q-Q Plot of Sampled Data vs. Gaussian Distribution" + 
                    "\n" + f"D = {D}")
    
        plt.savefig(f"figures/samples_gaussian/qq_{D}.png")

    return X, y


if __name__ == '__main__':
    mean_discrepancies_all = []
    var_discrepancies_all = []

    for _ in range(num_runs):
        mean_discrepancy = []
        var_discrepancy = []
        
        for D in dimensions:
            X, y = sample_from_gaussian(D)
            
            # Compute discrepancies
            var_discrepancy.append(np.abs(np.var(X) - 1/200))
            mean_discrepancy.append(np.abs(np.mean(X)))
        
        # Collect results for this run
        mean_discrepancies_all.append(mean_discrepancy)
        var_discrepancies_all.append(var_discrepancy)

    # Convert to numpy arrays for easier calculations
    mean_discrepancies_all = np.array(mean_discrepancies_all)
    var_discrepancies_all = np.array(var_discrepancies_all)

    # Calculate mean and standard deviation across runs for each dimension
    mean_discrepancy_avg = np.mean(mean_discrepancies_all, axis=0)
    mean_discrepancy_std = np.std(mean_discrepancies_all, axis=0)
    var_discrepancy_avg = np.mean(var_discrepancies_all, axis=0)
    var_discrepancy_std = np.std(var_discrepancies_all, axis=0)

    # Plot mean discrepancy vs. D with error bars
    plt.figure(figsize=(10, 5))
    plt.errorbar(dimensions, mean_discrepancy_avg, yerr=mean_discrepancy_std,
                 marker='o', color='r')
    plt.xlabel("Dimension")
    plt.ylabel("Discrepancy")
    plt.title("Discrepancy of sample mean vs Dimension")
    plt.grid(True)
    plt.savefig("figures/samples_gaussian/discrepancy_mean.png")

    # Plot variance discrepancy vs. D with error bars
    plt.figure(figsize=(10, 5))
    plt.errorbar(dimensions, var_discrepancy_avg, yerr=var_discrepancy_std,
                 marker='o', color='r')
    plt.xlabel("Dimension")
    plt.ylabel("Discrepancy")
    plt.title("Discrepancy of sample variance vs Dimension")
    plt.grid(True)
    plt.savefig("figures/samples_gaussian/discrepancy_var.png")
