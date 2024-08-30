from treeQuadrature.integrators import BayesMcIntegrator, TreeIntegrator
from treeQuadrature.container_integrators import AdaptiveRbfIntegral
from treeQuadrature.trees import SimpleTree
from treeQuadrature.splits import KdSplit
from treeQuadrature.example_problems import SimpleGaussian
from treeQuadrature.visualisation import plot_containers

import numpy as np
import matplotlib.pyplot as plt

problem = SimpleGaussian(D=3)

treeInteg = TreeIntegrator(8_000, tree=SimpleTree(split=KdSplit()), integral=AdaptiveRbfIntegral())
                             
bmcInteg = BayesMcIntegrator(500)

if __name__ == '__main__':
    # result = bmcInteg(problem_simple_Gaussian, return_std=True)
    # print(result)

    result = treeInteg(problem, return_containers=True, return_std=True)
    plt.hist(result['stds'])
    plt.show()
    print(f'total std = {np.sqrt(np.sum(np.array(result['stds']) ** 2))}')
    containers = result['containers']
    print(f'Number of containers: {len(containers)}')
    contributions = result['contributions']

    error = 100 * np.abs(result['estimate'] - problem.answer) / problem.answer
    print(f'Relative Error: {error:.2f} %')
    # plot_containers(containers, contributions, 
    #             xlim=[-1.0, 1.0], ylim=[-1.0, 1.0],
    #             integrand=problem.integrand, plot_samples=True, 
    #             dimensions=[0, 1])