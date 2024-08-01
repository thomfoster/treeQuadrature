import matplotlib.pyplot as plt

from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import RbfIntegral
from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.exampleProblems import Camel
from treeQuadrature.container import Container

D = 13

integ = SimpleIntegrator(1000, 40, MinSseSplit(), RbfIntegral())

# Assume Camel, Container, and integ are defined and initialized
problem = Camel(D)

# Define a range of base_N values to test
base_N_values = range(5000, 15000, 1000)  # Adjust the range and step as needed

# Lists to store base_N and the corresponding average sample size
base_N_list = []
avg_sample_size_list = []

for base_N in base_N_values:
    # Draw initial samples and create the root container
    X = integ.sampler.rvs(base_N, problem)
    y = problem.integrand(X)
    root = Container(X, y, mins=problem.lows, maxs=problem.highs)

    # Construct the tree of containers
    containers = integ.construct_tree(root)

    # Calculate the average sample size per container
    total_samples = sum(container.N for container in containers)
    avg_sample_size = total_samples / len(containers) if containers else 0

    # Store the results
    base_N_list.append(base_N)
    avg_sample_size_list.append(avg_sample_size)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(base_N_list, avg_sample_size_list, marker='o')
plt.xlabel('Base_N (Initial Sample Size)')
plt.ylabel('Average Sample Size per Container')
plt.title('Average Sample Size vs Base_N')
plt.grid(True)
plt.show()