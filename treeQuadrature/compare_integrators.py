from .exampleProblems import Problem
from .integrators import Integrator
from .visualisation import plotContainers

import numpy as np
import time
from typing import List
import warnings
from traceback import print_exc

def compare_integrators(integrators: List[Integrator], plot: bool, 
                        xlim: List[float], ylim: List[float],
                        problem: Problem, verbose: bool=True) -> None:
    """
    Compare different integrators on a given problem.

    Parameters
    ----------
    integrators : List[Integrator]
        A list of integrator instances to be compared.
    plot : bool
        Whether to plot the contributions of the integrators.
    xlim, ylim : List[float]
        The limits for the plot containers.
    problem : Problem
        The problem instance containing the integrand and true answer.
    verbose : bool
        if true, print the stages of the test
    """
    print(f'true value: {problem.answer}')

    for i, integrator in enumerate(integrators):
        if not hasattr(integrator, 'name'):
            integrator.name = f'integrators[{i}]'
        if verbose:
            print(f'testing {integrator.name}')

        start_time = time.time()
        # perform integration
        try:
            # Perform integration
            result = integrator(problem, return_N=True, 
                                return_containers=True)
        except Exception as e:
            print(f'Error during integration with {integrator.name}: {e}')
            print_exc()
            continue

        end_time = time.time()
        estimate = result['estimate']
        n_evals = result['n_evals']
        if problem.answer != 0:
            error = 100 * np.abs(estimate - problem.answer) / problem.answer
            error_name = 'Relative error'
        else: 
            error = np.abs(estimate - problem.answer)
            error_name = 'Absolute error'

        print(f'-------- {integrator.name} --------')
        print(f'Estimated value: {estimate}')
        print(f'{error_name}: {error:.2f} %')
        print(f'Number of evaluations: {n_evals}')
        print(f'Time taken: {end_time - start_time:.2f} s')


        # plot contributions
        if plot and 'containers' in result and 'contributions' in result:
            title = 'Integral estimate using ' + integrator.name
            containers = result['containers']
            contributions = result['contributions']
            plotContainers(containers, contributions, 
                           xlim=xlim, ylim=ylim,
                           integrand=problem.integrand, 
                           title=title, plot_samples=True)
        
        else: 
            warnings.warn('result of integrator has no containers to plot', 
                          UserWarning)