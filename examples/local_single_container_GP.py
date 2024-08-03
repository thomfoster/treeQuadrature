import numpy as np
import matplotlib.pyplot as plt

import time

from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import RbfIntegral
from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.exampleProblems import Camel
from treeQuadrature.container import Container

Ds = range(1, 15)
N = 5_000
P = 40
num_runs = 5

integ = SimpleIntegrator(N, P, MinSseSplit(), RbfIntegral())

lengths = np.zeros((num_runs, len(Ds)))
gp_times = np.zeros((num_runs, len(Ds)))
avg_samples = np.zeros((num_runs, len(Ds)))
min_samples = np.zeros((num_runs, len(Ds)))
max_samples = np.zeros((num_runs, len(Ds)))

log_base = np.exp(1)

for i in range(num_runs):
    for j, D in enumerate(Ds):
        base_N = int(N * np.log(D+log_base-1) / np.log(log_base))
        print(f'Run {i+1}, D={D}, base_N={base_N}')
        problem = Camel(D)
        X = integ.sampler.rvs(base_N, problem)
        y = problem.integrand(X)
        root = Container(X, y, mins=problem.lows, maxs=problem.highs)

        containers = integ.construct_tree(root)

        container_samples = [container.N for container in containers]
        avg_samples[i, j] = np.mean(container_samples)
        min_samples[i, j] = np.min(container_samples)
        max_samples[i, j] = np.max(container_samples)

        largest_container = max(containers, key=lambda container: container.N)

        start_time = time.time()
        integ.integral.containerIntegral(largest_container, problem.integrand)
        end_time = time.time()

        n_cont = len(containers)
        lengths[i, j] = n_cont
        gp_times[i, j] = end_time - start_time

# Calculate mean and standard deviation across the runs
mean_lengths = np.mean(lengths, axis=0)
std_lengths = np.std(lengths, axis=0)

mean_times = np.mean(gp_times, axis=0)
std_times = np.std(gp_times, axis=0)

# Calculate mean, min, max for the number of samples in containers
mean_avg_samples = np.mean(avg_samples, axis=0)
min_min_samples = np.min(min_samples, axis=0)
max_max_samples = np.max(max_samples, axis=0)

### plot number of containers
plt.plot(Ds, mean_lengths, label='Mean Number of Containers')
plt.fill_between(Ds, mean_lengths - std_lengths, mean_lengths + std_lengths, alpha=0.2, label='Std Dev')
plt.title('Number of Containers vs Dimension of Problem')
plt.xlabel('Dimension')
plt.ylabel('Number of Containers')
plt.legend()

plt.savefig('figures/container_nums_vs_dimension.png')
plt.close()

### plot samples in containers
plt.plot(Ds, mean_avg_samples, label='Average Samples per Container')
plt.plot(Ds, min_min_samples, label='Min Samples per Container', linestyle='--')
plt.plot(Ds, max_max_samples, label='Max Samples per Container', linestyle='--')
plt.fill_between(Ds, min_min_samples, max_max_samples, color='gray', alpha=0.1)
plt.title('Samples per Container vs Dimension of Problem')
plt.xlabel('Dimension')
plt.ylabel('Number of Samples')
plt.legend()

plt.savefig('figures/container_samples_vs_dimension.png')
plt.close()

### Plot time to fit GP
plt.plot(Ds, mean_times, label='Mean Time')
plt.fill_between(Ds, mean_times - std_times, mean_times + std_times, alpha=0.2, label='Std Dev')
plt.title('Time to fit GP in one container vs Dimension of Problem')
plt.xlabel('Dimension')
plt.ylabel('Time (s)')
plt.legend()

plt.savefig('figures/gp_times_vs_dimension.png')
plt.close()