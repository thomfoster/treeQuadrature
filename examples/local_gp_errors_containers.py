from treeQuadrature.exampleProblems import QuadraticProblem, ExponentialProductProblem, Problem
from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral, RandomIntegral, PolyIntegral, ContainerIntegral
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.samplers import ImportanceSampler

from treeQuadrature import Container

import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Monte Carlo sample size in each container
    ns = np.arange(20, 105, 10)
    # dimensions of the problem
    Ds = np.arange(2, 14, 2)
    Ds = np.append(Ds, 3)

    # number of containes to plot 
    n_containers = 6

    def mse(y_true, y_pred, sigma):
        return mean_squared_error(y_true, y_pred)

    def r2(y_true, y_pred, sigma):
        return r2_score(y_true, y_pred)
    
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

    rmeanIntegral = RandomIntegral(n_samples=20)

    split = MinSseSplit()
    sampler = ImportanceSampler()

    integ = SimpleIntegrator(20_000, 40, split, rmeanIntegral, 
                             sampler=sampler)

    problems = []
    for D in Ds:
        problems.append(QuadraticProblem(D))
        problems.append(ExponentialProductProblem(D))

    for problem in problems:
        print(f'testing {str(problem)}')

        results = {}

        X = sampler.rvs(integ.base_N, problem)
        y = problem.integrand(X)
        root = Container(X, y, mins=problem.lows, maxs=problem.highs)
        containers = integ.construct_tree(root)
        print(f'found {len(containers)} containers')

        # select the largest containers
        selected_containers = sorted(containers, 
                                    key=lambda c: c.volume, 
                                    reverse=True)[:n_containers]

        for n in ns:
            print(f'testing {n} samples')
            # integral = PolyIntegral(degrees=[2, 3], n_samples=n, max_redraw=0)
            integral = AdaptiveRbfIntegral(max_redraw=0, n_splits=5, scoring=nlpd, 
                                              min_n_samples=n)
            for container in selected_containers:
                gp_results = integral.containerIntegral(container, 
                                                problem.integrand, return_std=False)
                estimate = gp_results['integral']
                true_value = problem.exact_integral(container.mins, container.maxs)
                error = estimate - true_value
                lml = integral.gp.gp.log_marginal_likelihood()

                if container not in results:
                    # results[container] = {'performance': [], 'error': [], 'std': [], 'n': []}
                    results[container] = {'performance': [], 'error': [], 'n': []}
            
                results[container]['performance'].append(gp_results['performance'])
                results[container]['error'].append(error)
                # results[container]['std'].append(gp_results['std'])
                results[container]['n'].append(n)

        # Plot R^2 score vs Error
        plt.figure()
        for container in selected_containers:
            plt.plot(results[container]['performance'], results[container]['error'], marker='o')

        plt.title(f'Negative Log Predictive Density \n for {n_containers} containers with largest volumes')
        plt.xlabel('nlpd', fontsize=17)
        plt.ylabel('Error', fontsize=17)
        plt.grid(True)
        plt.tight_layout() 
        plt.savefig(f'figures/nlpd_errors_containers_{str(problem)}.png')
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
        # plt.figure()
        # for container in selected_containers:
        #     plt.plot(results[container]['n'], results[container]['error'], marker='o')

        # plt.title(f'Number of samples vs Error \n for {n_containers} containers with largest volumes')
        # plt.xlabel('Number of samples', fontsize=17)
        # plt.ylabel('Error', fontsize=17)
        # plt.grid(True)
        # plt.tight_layout()  
        # plt.savefig(f'figures/gp_n_errors_containers_{str(problem)}.png')
        # plt.close()