from treeQuadrature.exampleProblems import QuadraticProblem, ExponentialProductProblem, Gaussian
from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.containerIntegration import RbfIntegral, SmcIntegral
from treeQuadrature.splits import MinSseSplit

from treeQuadrature.visualisation import plotContainers, plotIntegrand

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    rbfIntegral = RbfIntegral(max_redraw=4, threshold=0.5, n_splits=5)
    rmeanIntegral = SmcIntegral(n=20)

    split = MinSseSplit()

    integ = SimpleIntegrator(10_000, 50, split, rbfIntegral)
    integ.name = 'TQ with RBF, fitting to mean'

    Ds = np.arange(2, 12, 2)
    for D in Ds:
        # problem = Gaussian(D=D, lows=-1.0, highs=1.0, Sigma=1/200)
        # problem = ExponentialProductProblem(D=D)
        problem = QuadraticProblem(D=D)

        if problem.D == 2:
            plotIntegrand(problem.integrand, D, xlim=[problem.lows[0], problem.highs[0]], 
                          ylim=[problem.lows[1], problem.highs[1]])

        print(f'Analytic solution {problem.answer}')

        results, containers = integ(problem, return_all = True)

        contributions = [result['integral'] for result in results]
        print(f'estimated value difference {np.sum(contributions) - problem.answer}')
        length_scales = [result['hyper_params']['length'] for result in results]

        true_values = [problem.exact_integral(cont.mins, cont.maxs) for cont in containers]
        print(f'true value difference {np.sum(true_values) - problem.answer}')

        # signed_errors = np.array(contributions) - np.array(true_values)

        # performances = [result['performance'] for result in results]
        # container_volumes = [cont.volume for cont in containers]

        # if problem.D == 2:
        #     plotContainers(containers, signed_errors, title='Plot of Signed Absolute Errors', 
        #                 xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], plot_samples=True, colors='coolwarm', 
        #                 c_bar_labels='Errors', integrand=problem.integrand)
        
        #     plotContainers(containers, length_scales, title='Plot of length scales of RBF kernel', 
        #                 xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], plot_samples=False,
        #                 c_bar_labels='Length scale', integrand=problem.integrand)
        
        # plt.scatter(performances, signed_errors)
        # plt.xlabel('R2 score of GP')
        # plt.ylabel('Signed error')
        # plt.tight_layout()
        # plt.savefig(f'figures/gp_errors_{str(problem)}')
        # plt.close()
        # print(f'figure saved to figures/gp_errors_{str(problem)}')

        # plt.scatter(container_volumes, signed_errors)
        # plt.xlabel('Volume')
        # plt.ylabel('Signed error')
        # plt.tight_layout()
        # plt.savefig(f'figures/errors_vs_volumes_{str(problem)}')
        # plt.close()
        # print(f'figure saved to figures/errors_vs_volumes_{str(problem)}')