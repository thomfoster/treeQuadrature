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
    ns = np.arange(15, 205, 15)

    # number of containes to plot 
    n_containers = 6

    rbfIntegral = AdaptiveRbfIntegral(max_redraw=0, n_splits=5, 
                                      return_std=True)
    rmeanIntegral = SmcIntegral(n=20)

    split = MinSseSplit()

    integ = SimpleIntegrator(10_000, 40, split, rbfIntegral)
    integ.name = 'TQ with RBF, fitting to mean'

    results = {}

    # problem = QuadraticProblem(D=12)
    problem = ExponentialProductProblem(D=12)

    X = integ.sampler.rvs(integ.base_N, problem)
    y = problem.integrand(X)
    root = Container(X, y, mins=problem.lows, maxs=problem.highs)
    containers = integ.construct_tree(root)
    print(f'found {len(containers)} containers')

    def compute_integral(container: Container, problem: Problem, n: int):
        gp_results = rbfIntegral.containerIntegral(container, 
                                                   problem.integrand, n)
        estimate = gp_results['integral']
        true_value = problem.exact_integral(container.mins, container.maxs)
        error = estimate - true_value
        return container, gp_results['performance'], error, gp_results['std']

    # select the largest containers
    selected_containers = sorted(containers, 
                                 key=lambda c: c.volume, 
                                 reverse=True)[:n_containers]

    for n in ns:
        print(f'testing {n} samples')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(compute_integral, container, 
                                       problem, n) for container in selected_containers]
            
            for future in concurrent.futures.as_completed(futures):
                container, performance, error, std = future.result()
                
                if container not in results:
                    results[container] = {'performance': [], 'error': [], 'std': []}
            
                results[container]['performance'].append(performance)
                results[container]['error'].append(error)
                results[container]['std'].append(std)

    # Plot R^2 score vs Error
    plt.figure()
    for container in selected_containers:
        plt.plot(results[container]['performance'], results[container]['error'], marker='o')

    plt.title(f'R^2 score vs Error for {n_containers} Largest Volume Containers')
    plt.xlabel('R^2 score', fontsize=17)
    plt.ylabel('Error', fontsize=17)
    plt.grid(True)
    plt.tight_layout() 
    plt.savefig(f'figures/r2_errors_containers_{str(problem)}.png')
    plt.close()

    # Plot Posterior std vs Error
    plt.figure()
    for container in selected_containers:
        plt.plot(results[container]['std'], results[container]['error'], marker='o')

    plt.title(f'Posterior std vs Error for {n_containers} Largest Volume Containers')
    plt.xlabel('Posterior Std', fontsize=17)
    plt.ylabel('Error', fontsize=17)
    plt.grid(True)
    plt.tight_layout()  
    plt.savefig(f'figures/gp_std_errors_containers_{str(problem)}.png')
    plt.close()