import numpy as np
import matplotlib.pyplot as plt

from treeQuadrature.splits import MinSseSplit
from treeQuadrature.container_integrators import KernelIntegral
from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.example_problems import Camel
from treeQuadrature.container import Container

D = 13

integ = TreeIntegrator(1000, integral=KernelIntegral())

# Assume `Camel`, `Container`, and `integ` are defined and initialized
problem = Camel(D)
base_N = 14000  # Number of initial samples, adjust as needed

# Draw initial samples and create the root container
X = integ.sampler.rvs(base_N, problem)
y = problem.integrand(X)
root = Container(X, y, mins=problem.lows, maxs=problem.highs)

# Construct the tree of containers
containers = integ.construct_tree(root)

# Lists to store the number of samples and standard deviations
num_samples_list = []
std_dev_list = []

# Iterate over each container
for container in containers:
    # Draw samples from the container
    samples = container.rvs(20)

    # Compute the function values for these samples
    function_values = problem.integrand(samples)

    # Calculate the standard deviation of the function values
    std_dev = np.std(function_values)

    # Store the number of samples and the standard deviation
    num_samples_list.append(container.N)
    std_dev_list.append(std_dev)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_samples_list, std_dev_list, marker='o')
plt.xlabel('Number of Samples')
plt.ylabel('Function Std Dev')
plt.title('Function Std Dev vs Number of Samples')
plt.grid(True)
plt.show()