from .exampleProblems import Problem
from .integrators import Integrator
from .visualisation import plotContainers

import warnings, time

from inspect import signature
import numpy as np
from typing import List, Optional
from traceback import print_exc

def compare_integrators(integrators: List[Integrator], problem: Problem, 
                        plot: bool=False, verbose: bool=False, 
                        xlim: Optional[List[float]]=None, 
                        ylim: Optional[List[float]]=None,
                        dimensions: Optional[List[float]]=None) -> None:
    """
    Compare different integrators on a given problem.

    Parameters
    ----------
    integrators : List[Integrator]
        A list of integrator instances to be compared.
    problem : Problem
        The problem instance containing the integrand and true answer.
    verbose : bool, optional
        if true, print the stages of the test
        Default: False
    plot : bool, optional
        Whether to plot the contributions of the integrators.
        Default is False
    xlim, ylim : List[float], optional
        The limits for the plot containers.
        will not be used when plot=False
    dimensions : List[Float], optional
        which dimensions to plot for higher dimensional problems
    """
    print(f'true value: {problem.answer}')

    for i, integrator in enumerate(integrators):
        if not hasattr(integrator, 'name'):
            integrator.name = f'integrators[{i}]'
        if verbose:
            print(f'testing {integrator.name}')

        start_time = time.time()
        # perform integration
        parameters = signature(integrator).parameters
        try:
            # Perform integration
            if 'verbose' in parameters and 'return_containers' in parameters:
                result = integrator(problem, return_N=True, 
                                return_containers=True, 
                                verbose=verbose) 
            elif 'return_containers' in parameters:
                result = integrator(problem, return_N=True, 
                                return_containers=True)
            elif 'verbose' in parameters:
                result = integrator(problem, return_N=True, 
                                verbose=verbose)
            else:
                result = integrator(problem, return_N=True)

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
        if plot:
            if xlim is None or ylim is None:
                raise ValueError(
                    'xlim and ylim must be provided for plotting'
                    )

            if 'containers' in result and 'contributions' in result:
                title = 'Integral estimate using ' + integrator.name
                containers = result['containers']
                print(f'Number of containers: {len(containers)}')
                contributions = result['contributions']
                plotContainers(containers, contributions, 
                            xlim=xlim, ylim=ylim,
                            integrand=problem.integrand, 
                            title=title, plot_samples=True, 
                            dimensions=dimensions)
            else: 
                warnings.warn('result of integrator has no containers to plot', 
                          UserWarning)
        
        print(f'----------------------------------')