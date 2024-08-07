from treeQuadrature.exampleProblems import QuadraticProblem, ExponentialProductProblem, Problem
from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral, SmcIntegral
from treeQuadrature.splits import MinSseSplit

from treeQuadrature import Container

import numpy as np
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Monte Carlo sample size in each container
    ns = np.arange(15, 45, 15)

    # number of containers to plot 
    n_containers = 10

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
    containers = integ.construct_tree(root)

    def compute_integral(container: Container, problem: Problem, n: int):
        gp_results = rbfIntegral.containerIntegral(container, 
                                                   problem.integrand, n)
        estimate = gp_results['integral']
        true_value = problem.exact_integral(container.mins, container.maxs)
        error = estimate - true_value
        return container, gp_results['performance'], error

    for n in ns:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(compute_integral, container, 
                                       problem, n) for container in containers]
            
            for future in concurrent.futures.as_completed(futures):
                container, performance, error = future.result()
                
                if container not in results:
                    results[container] = {'performance': [], 'error': []}
            
                results[container]['performance'].append(performance)
                results[container]['error'].append(error)

    selected_containers = sorted(results.keys(), key=lambda x: x.volume, 
                                 reverse=True)[:n_containers]

    plt.figure()
    for container in selected_containers:
        data = results[container]
        plt.plot(data['performance'], data['error'], marker='o', label=f'Container {id(container)}')

    plt.title(f'R^2 score vs Error for Largest Volume Containers')
    plt.xlabel('R^2 score')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'gp_errors_containers_volume_{str(problem)}.png')
    plt.close()