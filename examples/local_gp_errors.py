from treeQuadrature.exampleProblems import QuadraticProblem, ExponentialProductProblem
from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.containerIntegration import RbfIntegral, SmcIntegral
from treeQuadrature.splits import MinSseSplit

from treeQuadrature.visualisation import plotContainers
from treeQuadrature import Container

import numpy as np
import matplotlib.pyplot as plt

# problem = QuadraticProblem(D=10)
# problem = Gaussian(D=3, lows=-1.0, highs=1.0, Sigma=1/200)
problem = ExponentialProductProblem(D=2)

rbfIntegral = RbfIntegral(max_redraw=4, threshold=0.5, n_splits=3)
rmeanIntegral = SmcIntegral(n=20)

split = MinSseSplit()

integ = SimpleIntegrator(10_000, 50, split, rbfIntegral)
integ.name = 'TQ with RBF, fitting to mean'

if __name__ == '__main__':
    print(f'Analytic solution {problem.answer}')

    results, containers = integ(problem, return_all = True)

    contributions = [result['integral'] for result in results]
    print(f'estimated value difference {np.sum(contributions) - problem.answer}')
    length_scales = [result['hyper_params']['length'] for result in results]

    true_values = [problem.exact_integral(cont.mins, cont.maxs) for cont in containers]
    print(f'true value difference {np.sum(true_values) - problem.answer}')

    signed_errors = np.array(contributions) - np.array(true_values)

    performances = [result['performance'] for result in results]

    if problem.D == 2:
        plotContainers(containers, signed_errors, title='Plot of Signed Absolute Errors', 
                    xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], plot_samples=True, colors='coolwarm', 
                    c_bar_labels='Errors', integrand=problem.integrand)
    
        plotContainers(containers, length_scales, title='Plot of length scales of RBF kernel', 
                    xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], plot_samples=False,
                    c_bar_labels='Length scale', integrand=problem.integrand)
    
    plt.scatter(performances, signed_errors)
    plt.xlabel('R2 score of GP')
    plt.ylabel('Signed error')
    plt.title('Errors of containers')
    plt.show()