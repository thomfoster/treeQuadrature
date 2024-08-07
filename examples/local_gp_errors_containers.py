from treeQuadrature.exampleProblems import QuadraticProblem, ExponentialProductProblem
from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral, SmcIntegral
from treeQuadrature.splits import MinSseSplit

from treeQuadrature import Container

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Monte Carlo sample size in each container
    ns = np.arange(15, 45, 15)

    rbfIntegral = AdaptiveRbfIntegral(max_redraw=0,
                                      n_splits=5)
    rmeanIntegral = SmcIntegral(n=20)

    split = MinSseSplit()

    integ = SimpleIntegrator(5_000, 80, split, rbfIntegral)
    integ.name = 'TQ with RBF, fitting to mean'

    results = {}

    # problem = QuadraticProblem(D=2)
    problem = ExponentialProductProblem(D=2)

    X = integ.sampler.rvs(integ.base_N, problem)
    y = problem.integrand(X)
    root = Container(X, y, mins=problem.lows, maxs=problem.highs)
    containers = integ.construct_tree(problem, return_all = True)

    for container in containers:
        for n in ns:
            gp_results = rbfIntegral.containerIntegral(container, problem.integrand,
                                                        n)
            estimate = gp_results['integral']
            true_value = problem.exact_integral(container.mins, container.maxs)
            error = estimate - true_value
            
            if container not in results:
                results[container] = {'performance': [], 'error': []}
        
            results[container]['performance'].append(gp_results['performance'])
            results[container]['error'].append(error)

    
    plt.figure()
    for container, data in results.items():
        plt.plot(data['performance'], data['error'], marker='o')
    plt.title(f'R^2 score vs Error for Container {container}')
    plt.xlabel('R^2 score')
    plt.ylabel('Error')
    plt.grid(True)
    plt.savefig(f'gp_errors_containers_{str(problem)}.png')
    plt.close()