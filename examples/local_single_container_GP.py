import numpy as np

import matplotlib.pyplot as plt

from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import RbfIntegral
from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.exampleProblems import SimpleGaussian
from treeQuadrature.container import Container

Ds = range(1, 14)
N = 8_000
P = 40
num_runs = 10

integ = SimpleIntegrator(N, P, MinSseSplit(), RbfIntegral())

lengths = np.zeros((num_runs, len(Ds)))

for i in range(num_runs):
    for j, D in enumerate(Ds):
        print(f'Run {i+1}, D={D}')
        problem = SimpleGaussian(D)
        X = integ.sampler.rvs(N, problem)
        y = problem.integrand(X)
        root = Container(X, y, mins=problem.lows, maxs=problem.highs)

        containers = integ.construct_tree(root)

        n_cont = len(containers)
        lengths[i, j] = n_cont

# Calculate mean and standard deviation across the runs
mean_lengths = np.mean(lengths, axis=0)
std_lengths = np.std(lengths, axis=0)

# Plotting
plt.plot(Ds, mean_lengths, label='Mean Number of Containers')
plt.fill_between(Ds, mean_lengths - std_lengths, mean_lengths + std_lengths, alpha=0.2, label='Std Dev')
plt.title('Number of Containers vs Dimension of Problem')
plt.xlabel('Dimension')
plt.ylabel('Number of Containers')
plt.legend()

# Display the plot
plt.show()