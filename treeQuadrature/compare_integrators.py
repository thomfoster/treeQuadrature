from .exampleProblems import Problem
from .integrators import Integrator
from .visualisation import plotContainers

import warnings, time, csv, concurrent.futures

from inspect import signature
import numpy as np
from typing import List, Optional, Dict, Any
from traceback import print_exc
import os

def compare_integrators(integrators: List['Integrator'], problem: 'Problem', 
                        plot: bool=False, verbose: bool=False, 
                        xlim: Optional[List[float]]=None, 
                        ylim: Optional[List[float]]=None,
                        dimensions: Optional[List[float]]=None,
                        n_repeat: int=1) -> None:
    """
    Compare different integrators on a given problem.
    Give integrators attribute `name` 
    for clear outputs. 

    It will print for each integrator: 
    - Estimated integral value
    - The signed relative error  (estimate - answer) / answer
        (unless problem.answer is 0, 
        in which case signed absolute error will be used)
    - Number of integrand evaluations
    - Time taken in seconds
    if integrator uses containers, 
    - number of containers used
    - average number of samples per container
    - min number of samples per container
    - max number of samples per container

    Parameters
    ----------
    integrators : List[Integrator]
        A list of integrator instances to be compared.
    problem : Problem
        The problem instance containing the integrand and true answer.
    plot : bool, optional
        Whether to plot the contributions of the integrators.
        Default is False.
    verbose : bool, optional
        If true, print the stages of the test.
        Default is False.
    xlim, ylim : List[float], optional
        The limits for the plot containers.
        Will not be used when plot=False.
    dimensions : List[Float], optional
        Which dimensions to plot for higher dimensional problems.
    n_repeat : int, optional
        Number of times to repeat the integration and average the results.
        Default is 1.
    """
    print(f'True value: {problem.answer}')
    D = problem.D

    for i, integrator in enumerate(integrators):
        integrator_name = getattr(integrator, 'name', f'integrator[{i}]')

        if verbose:
            print(f'Testing {integrator_name}')

        estimates = []
        n_evals_list = []
        times = []

        for _ in range(n_repeat):
            start_time = time.time()
            try:
                # Perform integration
                parameters = signature(integrator).parameters
                if 'verbose' in parameters and 'return_containers' in parameters:
                    result = integrator(problem, return_N=True, return_containers=True, verbose=verbose)
                elif 'return_containers' in parameters:
                    result = integrator(problem, return_N=True, return_containers=True)
                elif 'verbose' in parameters:
                    result = integrator(problem, return_N=True, verbose=verbose)
                else:
                    result = integrator(problem, return_N=True)
            except Exception as e:
                print(f'Error during integration with {integrator_name}: {e}')
                print_exc()
                continue

            end_time = time.time()
            estimate = result['estimate']
            n_evals = result['n_evals']

            estimates.append(estimate)
            n_evals_list.append(n_evals)
            times.append(end_time - start_time)

        avg_estimate = np.mean(estimates)
        std_estimate = np.std(estimates)
        avg_n_evals = np.mean(n_evals_list)
        std_n_evals = np.std(n_evals_list)
        avg_time = np.mean(times)
        std_time = np.std(times)

        if problem.answer != 0:
            errors = 100 * (np.array(estimates) - problem.answer) / problem.answer
            avg_error = np.mean(errors)
            std_error = np.std(errors)
            error_name = 'Signed Relative error'
        else:
            errors = np.array(estimates) - problem.answer
            avg_error = np.mean(errors)
            std_error = np.std(errors)
            error_name = 'Signed Absolute error'

        print(f'-------- {integrator_name} --------')
        print(f'Estimated value: {avg_estimate:.4f} ± {std_estimate:.4f}')
        print(f'{error_name}: {avg_error:.2f} % ± {std_error:.2f} %')
        print(f'Number of evaluations: {avg_n_evals:.2f} ± {std_n_evals:.2f}')
        print(f'Time taken: {avg_time:.2f} s ± {std_time:.2f} s')

        if 'containers' in result and 'contributions' in result:
            title = 'Integral estimate using ' + integrator_name
            containers = result['containers']
            print(f'Number of containers: {len(containers)}')
            n_samples = [cont.N for cont in containers]
            print(f'Average samples/container: {np.mean(n_samples)}')
            print(f'Minimum samples in containers: {np.min(n_samples)}')
            print(f'Maximum samples in containers: {np.max(n_samples)}')
            contributions = result['contributions']
            if plot:
                if xlim is None or ylim is None:
                    raise ValueError('xlim and ylim must be provided for plotting')
                plotContainers(containers, contributions, 
                               xlim=xlim, ylim=ylim,
                               integrand=problem.integrand, 
                               title=title, plot_samples=True, 
                               dimensions=dimensions)
        elif plot: 
            warnings.warn('Result of integrator has no containers to plot', 
                          UserWarning)
        
        print(f'----------------------------------')


## add protection to code interruption
def load_existing_results(output_file: str) -> List[Dict[str, Any]]:
    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = (row['integrator'], row['problem'])
                if row['estimate'] != 'None':
                    existing_results[key] = row

    return existing_results

def test_integrators(integrators: List[Integrator], 
                     problems: List[Problem], 
                     output_file: str='results.csv', 
                     max_time: float=60.0, 
                     verbose: int=1, 
                     seed: int=2024, 
                     n_repeat: int=1) -> None:
    """
    Test different integrators on a list of problems 
    and save the results to a CSV file.
    Give integrators attribute `name` 
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
        if 1, print the problem and integrator being tested; 
        if 2, print details of each integrator as well; 
        if 0, print nothing.
        Default is 1
    seed : int, optional
        specify the randomness seed 
        for reproducibility 
    n_repeat : int, optional
        Number of times to repeat the integration and 
        average the results.
        Default is 1
    """

    np.random.seed(seed)

    existing_results = load_existing_results(output_file)

    results = []

    for problem in problems:
        problem_name = str(problem)

        if verbose >= 1:
            print(f'testing Probelm: {problem_name}')

        for i, integrator in enumerate(integrators):
            integrator_name = getattr(integrator, 'name', f'integrator[{i}]')

            # Check if the result already exists and is valid
            key = (integrator_name, problem_name)
            if key in existing_results and existing_results[key]['estimate'] != '':
                if verbose >= 1:
                    print(f'Skipping {integrator_name} for {problem_name}: already completed.')
                results.append(existing_results[key])
                continue

            if verbose >= 1:
                print(f'testing Integrator: {integrator_name}')

            estimates = []
            n_evals_list = []
            total_time_taken = 0

            for repeat in range(n_repeat):
                np.random.seed(seed + repeat)
                start_time = time.time()
                parameters = signature(integrator).parameters

                def integrator_wrapper():
                    if 'verbose' in parameters and verbose >= 2:
                        return integrator(problem, return_N=True, verbose=True)
                    else:
                        return integrator(problem, return_N=True)
                    
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(integrator_wrapper)
                    try:
                        result = future.result(timeout=max_time)
                        end_time = time.time()
                        time_taken = end_time - start_time
                        total_time_taken += time_taken
                    except concurrent.futures.TimeoutError:
                        print(
                            f'Time limit exceeded for {integrator_name} on {problem_name}, '
                            'increase max_time or change the problem/integrator'
                            )
                        results.append({
                            'integrator': integrator_name,
                            'problem': problem_name,
                            'true_value': problem.answer,
                            'estimate': None,
                            'estimate_std': None,
                            'error_type': 'Timeout',
                            'error': None,
                            'error_std': None,
                            'n_evals': None,
                            'n_evals_std': None,
                            'time_taken': f'Exceeded {max_time}s',
                            'errors': None
                        })
                        break
                    except Exception as e:
                        print(f'Error during integration with {integrator_name} on {problem_name}: {e}')
                        results.append({
                            'integrator': integrator_name,
                            'problem': problem_name,
                            'true_value': problem.answer,
                            'estimate': None,
                            'estimate_std': None,
                            'error_type': e,
                            'error': None,
                            'error_std': None,
                            'n_evals': None,
                            'n_evals_std': None,
                            'time_taken': None,
                            'errors': None
                        })
                        print_exc()
                        break

                estimate = result['estimate']
                n_evals = result['n_evals']
                estimates.append(estimate)
                n_evals_list.append(n_evals)

            if len(estimates) == n_repeat:
                estimates = np.array(estimates)
                avg_estimate = np.mean(estimates)
                avg_n_evals = np.mean(n_evals_list)
                avg_time_taken = total_time_taken / n_repeat

                if problem.answer != 0:
                    errors = 100 * (estimates - problem.answer) / problem.answer
                    avg_error = f'{np.mean(errors):.4f} %'
                    error_std = f'{np.std(errors):.4f} %'
                    error_name = 'Signed Relative error'
                else: 
                    errors = estimates - problem.answer
                    avg_error = np.mean(errors)
                    error_std = np.std(errors)
                    error_name = 'Signed Absolute error'

                results.append({
                    'integrator': integrator_name,
                    'problem': problem_name,
                    'true_value': problem.answer,
                    'estimate': avg_estimate,
                    'estimate_std': np.std(estimates),
                    'error_type': error_name,
                    'error': avg_error,
                    'error_std': error_std,
                    'n_evals': avg_n_evals,
                    'n_evals_std': np.std(n_evals_list),
                    'time_taken': avg_time_taken, 
                    'errors': errors
                })
    
            # Save for each integrator and each problem
            with open(output_file, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'integrator', 'problem', 'true_value', 'estimate', 'estimate_std', 'error_type', 
                    'error', 'error_std', 'n_evals', 'n_evals_std', 'time_taken', 'errors'])
                writer.writeheader()
                for result in results:
                    writer.writerow(result)

    print(f'Results saved to {output_file}')