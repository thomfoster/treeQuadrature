from .exampleProblems import Problem
from .integrators import Integrator
from .visualisation import plotContainers

import warnings, time, signal, csv

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
        integrator_name = getattr(integrator, 'name', f'integrator[{i}]')

        if verbose:
            print(f'testing {integrator_name}')

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
            print(f'Error during integration with {integrator_name}: {e}')
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

        print(f'-------- {integrator_name} --------')
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
                title = 'Integral estimate using ' + integrator_name
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


class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException

def test_integrators(integrators: List[Integrator], 
                     problems: List[Problem], 
                     output_file: str='results.csv', 
                     max_time: float=60.0, 
                     verbose: int=1, 
                     seed: int=2024) -> None:
    """
    Test different integrators on a list of problems 
    and save the results to a CSV file.
    give integrators attribute `name` 
    for clear outputs. 

    Parameters
    ----------
    integrators : List[Integrator]
        A list of integrator instances to be tested.
    problems : List[Problem]
        A list of problem instances containing 
        the integrand and true answer.
    output_file : str, optional
        The file path to save the results as a CSV.
        Default is 'results.csv'.
    max_time : float, optional
        Maximum allowed time (in seconds) for each integration.
        Default is 60.0 seconds.
    verbose : int, optional
        if 1, print problems and integrator stages
        otherwise, print nothing.
        Default is 1
    seed : int, optional
        specify the randomness seed 
        for reproducibility 
    """

    np.random.seed(seed)

    results = []

    for j, problem in enumerate(problems):
        problem_name = str(problem)
        if verbose >= 1:
            print(f'testing Probelm: {problem_name}')

        for i, integrator in enumerate(integrators):
            integrator_name = getattr(integrator, 'name', f'integrator[{i}]')

            if verbose >= 1:
                print(f'testing Integrator: {integrator_name}')

            start_time = time.time()
            parameters = signature(integrator).parameters
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(int(max_time))

                if 'return_containers' in parameters:
                    result = integrator(problem, return_N=True, return_containers=True)
                else:
                    result = integrator(problem, return_N=True)

                signal.alarm(0)  # Disable the alarm

                end_time = time.time()
                time_taken = end_time - start_time

            except TimeoutException:
                print(
                    f'Time limit exceeded for {integrator_name} on {problem_name}, '
                    'increase max_time or change the problem/integrator'
                    )
                results.append({
                    'integrator': integrator_name,
                    'problem': problem_name,
                    'estimate': None,
                    'error_type': 'Timeout',
                    'error': None,
                    'n_evals': None,
                    'time_taken': 'Exceeded max_time'
                })
                continue
            except Exception as e:
                print(f'Error during integration with {integrator_name} on {problem_name}: {e}')
                print_exc()
                continue

            estimate = result['estimate']
            n_evals = result['n_evals']
            if problem.answer != 0:
                error = 100 * np.abs(estimate - problem.answer) / problem.answer
                error_name = 'Relative error'
            else: 
                error = np.abs(estimate - problem.answer)
                error_name = 'Absolute error'

            results.append({
                'integrator': integrator_name,
                'problem': problem_name,
                'estimate': estimate,
                'error_type': error_name,
                'error': error,
                'n_evals': n_evals,
                'time_taken': time_taken
            })
    
    # Save results to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'integrator', 'problem', 'estimate', 'error_type', 
            'error', 'n_evals', 'time_taken'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f'Results saved to {output_file}')