import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import MidpointIntegral
from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.exampleProblems import Camel
from treeQuadrature.container import Container

D = 5  # try 2, 5, 8, 10 various dimensions
n_repeats = 5  # Number of times to repeat the experiment

# Assume Camel, Container, and integ are defined and initialized
problem = Camel(D)

# Define a range of base_N values to test
base_N_values = range(6000, 15000, 1000)
P_values = range(20, 80, 10)

# DataFrame to store the results
results = []

for P in P_values:
    for base_N in base_N_values:
        print(f'testing base_N = {base_N}, P = {P}')
        avg_sample_size_repeats = []
        avg_std_small_containers_repeats = []
        for _ in range(n_repeats):
            integ = SimpleIntegrator(base_N, P, MinSseSplit(), MidpointIntegral())
            X = integ.sampler.rvs(base_N, problem)
            y = problem.integrand(X)
            root = Container(X, y, mins=problem.lows, maxs=problem.highs)

    # Construct the tree of containers
    tree = integ.construct_tree(root)
    leaf_nodes = tree.get_leaf_nodes()
    containers = [node.container for node in leaf_nodes]

            total_samples = sum(container.N for container in containers)
            avg_sample_size = total_samples / len(containers) if containers else 0

            small_containers = [container for container in containers if container.N == 1]
            if small_containers:
                samples_from_small_containers = np.vstack([container.rvs(20) 
                                                           for container in small_containers])
                avg_std_small_containers = np.std(problem.integrand(samples_from_small_containers))
            else:
                avg_std_small_containers = np.nan

            avg_sample_size_repeats.append(avg_sample_size)
            avg_std_small_containers_repeats.append(avg_std_small_containers)

        # Store median results
        results.append({
            'P': P,
            'base_N': base_N,
            'avg_sample_size': np.median(avg_sample_size_repeats),
            'avg_std_small_containers': np.median(avg_std_small_containers_repeats)
        })

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Create pivot tables for plotting
avg_sample_sizes = df_results.pivot(index='P', columns='base_N', 
                                    values='avg_sample_size').values
avg_std_smalls = df_results.pivot(index='P', columns='base_N', 
                                  values='avg_std_small_containers').values

# Plotting average sample size as heatmap
plt.figure(figsize=(10, 6))
plt.imshow(avg_sample_sizes, aspect='auto', cmap='YlGnBu', origin='lower')
plt.colorbar(label='Average Sample Size')
plt.xticks(np.arange(len(base_N_values)), base_N_values)
plt.yticks(np.arange(len(P_values)), P_values)
plt.xlabel('Base_N (Initial Sample Size)')
plt.ylabel('P (Max Samples per Container)')
plt.title(f'Average Sample Size per Container for {str(problem)}')
plt.show()

# Plotting std of integrand in small containers as heatmap
plt.figure(figsize=(10, 6))
plt.imshow(avg_std_smalls, aspect='auto', cmap='YlOrRd', origin='lower')
plt.colorbar(label='Std of Integrand')
plt.xticks(np.arange(len(base_N_values)), base_N_values)
plt.yticks(np.arange(len(P_values)), P_values)
plt.xlabel('Base_N (Initial Sample Size)')
plt.ylabel('P (Max Samples per Container)')
plt.title(f'Std of Integrand in Small Containers (N = 2) \n for {str(problem)} ')

# Mark invalid entries with a specific color 
for (i, j), val in np.ndenumerate(avg_std_smalls):
    if np.isnan(val):
        plt.text(j, i, 'X', ha='center', va='center', color='red', fontsize=12)

plt.show()