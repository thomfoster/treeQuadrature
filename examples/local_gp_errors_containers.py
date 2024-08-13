from treeQuadrature.exampleProblems import QuadraticProblem, ExponentialProductProblem, Problem
from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral, RandomIntegral, PolyIntegral, ContainerIntegral
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.samplers import ImportanceSampler

from treeQuadrature import Container

import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Monte Carlo sample size in each container
    ns = np.arange(10, 105, 10)
    # dimensions of the problem
    Ds = np.arange(2, 14, 2)
    Ds = np.append(Ds, 3)

    # number of containes to plot 
    n_containers = 6

    def mse(y_true, y_pred, sigma):
        return mean_squared_error(y_true, y_pred)
    
    def predictive_ll(y_true, y_pred, sigma):
        return np.sum(norm.logpdf(y_true, loc=y_pred, scale=sigma))
    
    def picp(y_true, y_pred, sigma):
        """Prediction Interval Coverage Probability"""
        lower_bound = y_pred - 1.96 * sigma
        upper_bound = y_pred + 1.96 * sigma
        return np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    
    def nlpd(y_true, y_pred, sigma):
        """Negative Log Predictive Density"""
        return -np.mean(norm.logpdf(y_true, loc=y_pred, scale=sigma))

    rbfIntegral = AdaptiveRbfIntegral(max_redraw=0, n_splits=5, scoring=nlpd)
    rmeanIntegral = RandomIntegral(n=20)

    split = MinSseSplit()
    sampler = ImportanceSampler()

    integ = SimpleIntegrator(20_000, 40, split, rbfIntegral, 
                             sampler=sampler)
    integ.name = 'TQ with RBF, fitting to mean'

    def compute_integral(integral: ContainerIntegral, container: Container, problem: Problem):
        gp_results = integral.containerIntegral(container, 
                                                problem.integrand, return_std=False)
        estimate = gp_results['integral']
        true_value = problem.exact_integral(container.mins, container.maxs)
        error = estimate - true_value
        lml = integral.gp.gp.log_marginal_likelihood()
        return container, lml, error
        # return container, gp_results['performance'], error

    problems = []
    for D in Ds:
        problems.append(QuadraticProblem(D))
        problems.append(ExponentialProductProblem(D))

    for problem in problems:
        print(f'testing {str(problem)}')

        results = {}

        X = integ.sampler.rvs(integ.base_N, problem)
        y = problem.integrand(X)
        root = Container(X, y, mins=problem.lows, maxs=problem.highs)
        containers = integ.construct_tree(root)
        print(f'found {len(containers)} containers')

        # select the largest containers
        selected_containers = sorted(containers, 
                                    key=lambda c: c.volume, 
                                    reverse=False)[:n_containers]

        for n in ns:
            print(f'testing {n} samples')
            integral = PolyIntegral(degrees=[2, 3], n_samples=n, max_redraw=0)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(compute_integral, integral, container, 
                                        problem) for container in selected_containers]
                
                for future in concurrent.futures.as_completed(futures):
                    # container, performance, error, std = future.result()
                    container, performance, error = future.result()
                    
                    if container not in results:
                        # results[container] = {'performance': [], 'error': [], 'std': [], 'n': []}
                        results[container] = {'performance': [], 'error': [], 'n': []}
                
                    results[container]['performance'].append(performance)
                    results[container]['error'].append(error)
                    # results[container]['std'].append(std)
                    results[container]['n'].append(n)

        # Plot R^2 score vs Error
        plt.figure()
        for container in selected_containers:
            plt.plot(results[container]['performance'], results[container]['error'], marker='o')

        plt.title(f'Log marginal likelihood vs Error \n for {n_containers} containers with smallest volumes')
        plt.xlabel('LML', fontsize=17)
        plt.ylabel('Error', fontsize=17)
        plt.grid(True)
        plt.tight_layout() 
        plt.savefig(f'figures/lml_errors_small_containers_{str(problem)}.png')
        plt.close()

        # Plot Posterior std vs Error
        # plt.figure()
        # for container in selected_containers:
        #     plt.plot(results[container]['std'], results[container]['error'], marker='o')

        # plt.title(f'Posterior std vs Error for {n_containers} Largest Volume Containers')
        # plt.xlabel('Posterior Std', fontsize=17)
        # plt.ylabel('Error', fontsize=17)
        # plt.grid(True)
        # plt.tight_layout()  
        # plt.savefig(f'figures/gp_std_errors_containers_{str(problem)}.png')
        # plt.close()

        # Plot Error vs number of samples
        plt.figure()
        for container in selected_containers:
            plt.plot(results[container]['n'], results[container]['error'], marker='o')

        plt.title(f'Number of samples vs Error \n for {n_containers} containers with smallest volumes')
        plt.xlabel('Number of samples', fontsize=17)
        plt.ylabel('Error', fontsize=17)
        plt.grid(True)
        plt.tight_layout()  
        plt.savefig(f'figures/poly_gp/gp_n_errors_small_containers_{str(problem)}.png')
        plt.close()