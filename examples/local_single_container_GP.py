import time

import matplotlib.pyplot as plt

from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import RbfIntegral
from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.exampleProblems import SimpleGaussian
from treeQuadrature.container import Container

Ds = range(1, 14)

N = 8_000
P = 40


integ = SimpleIntegrator(N, P, MinSseSplit(), RbfIntegral())

lengths = []

for D in Ds:
    print(f'D={D}')
    problem = SimpleGaussian(D)
    X = integ.sampler.rvs(N, problem)
    y = problem.integrand(X)
    root = Container(X, y, mins=problem.lows, maxs=problem.highs)

    containers = integ.construct_tree(root)

    n_cont = len(containers)

    lengths.append(n_cont)

plt.plot(Ds, lengths)
plt.title('Number of containers vs dimension of problem')
plt.xlabel('Dimension')
plt.ylabel('Number')

# Display the plot
plt.show()