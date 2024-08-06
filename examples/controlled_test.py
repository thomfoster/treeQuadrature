from treeQuadrature.exampleProblems import Problem
from treeQuadrature.integrators import Integrator
from treeQuadrature.compare_integrators import load_existing_results
from treeQuadrature.integrators import LimitedSampleIntegrator, VegasIntegrator, SmcIntegrator

import time, csv, concurrent.futures, os

from inspect import signature
import numpy as np
from typing import List
from traceback import print_exc

def load_existing_results(output_file: str) -> dict:
    if not os.path.exists(output_file):
        return {}
    with open(output_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        return {(row['integrator'], row['problem']): row for row in reader}

def write_results(output_file: str, results: List[dict], write_header: bool):
    with open(output_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'integrator', 'problem', 'true_value', 'estimate', 'estimate_std', 'error_type', 
            'error', 'error_std', 'n_evals', 'n_evals_std', 'time_taken', 'errors'])
        if write_header:
            writer.writeheader()
        for result in results:
            writer.writerow(result)

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
    new_results = []
    n_eval = None
    existing_keys = set(existing_results.keys())

    is_first_run = not os.path.exists(output_file)

    for problem in problems:
        problem_name = str(problem)

        # extract n_eval from existing results
        if existing_results:
            for key, value in existing_results.items():
                if value['integrator'] == integrators[0].name and value['n_evals'] is not None and (
                    value['problem'] == problem_name
                ):
                    n_eval = int(float(value['n_evals']))
                    break

        if verbose >= 1:
            print(f'testing Problem: {problem_name}')

        for i, integrator in enumerate(integrators):
            integrator_name = getattr(integrator, 'name', f'integrator[{i}]')

            # Check if the result already exists and is valid
            key = (integrator_name, problem_name)
            if key in existing_results and existing_results[key]['estimate'] != '':
                if verbose >= 1:
                    print(f'Skipping {integrator_name} for {problem_name}: already completed.')
                results.append(existing_results[key])
                continue

            if i > 0 and n_eval is not None:
                if isinstance(integrator, VegasIntegrator):
                    n_iter = integrator.NITN
                    n = int(n_eval / n_iter)
                    integrator.N = n
                elif isinstance(integrator, SmcIntegrator):
                    integrator.N = n_eval
                elif isinstance(integrator, LimitedSampleIntegrator):
                    integrator.N = int(n_eval / (integrator.integral.n + 1))
            elif i > 0:
                raise Exception('first integrator did not set n_eval')

            if verbose >= 1:
                print(f'testing Integrator: {integrator_name}')

            estimates = []
            n_evals_list = []
            total_time_taken = 0

            for repeat in range(n_repeat):
                np.random.seed(seed + repeat)
                start_time = time.time()
                parameters = signature(integrator).parameters
                break_integrator = False

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
                        new_result = {
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
                            'time_taken': f'Exceeded max_time {max_time}',
                            'errors': None
                        }
                        if key not in existing_keys:
                            write_results(output_file, [new_result], is_first_run)
                            is_first_run = False
                            existing_keys.add(key)
                        break_integrator = True
                        break
                    except Exception as e:
                        print(f'Error during integration with {integrator_name} on {problem_name}: {e}')
                        new_result = {
                            'integrator': integrator_name,
                            'problem': problem_name,
                            'true_value': problem.answer,
                            'estimate': None,
                            'estimate_std': None,
                            'error_type': str(e),
                            'error': None,
                            'error_std': None,
                            'n_evals': None,
                            'n_evals_std': None,
                            'time_taken': None,
                            'errors': None
                        }
                        if key not in existing_keys:
                            write_results(output_file, [new_result], is_first_run)
                            is_first_run = False
                            existing_keys.add(key)
                        print_exc()
                        break_integrator = True
                        break

                estimate = result['estimate']
                n_evals = result['n_evals']
                estimates.append(estimate)
                n_evals_list.append(n_evals)

            if break_integrator:
                continue

            if len(estimates) == n_repeat:
                estimates = np.array(estimates)
                avg_estimate = np.mean(estimates)
                avg_n_evals = int(np.mean(n_evals_list))
                avg_time_taken = total_time_taken / n_repeat

                # Store the average number of evaluations for the first integrator
                if i == 0:
                    n_eval = avg_n_evals
                    
                if problem.answer != 0:
                    errors = 100 * (estimates - problem.answer) / problem.answer
                    avg_error = f'{np.median(errors):.4f} %'
                    error_std = f'{np.std(errors)::.4f} %'
                    error_name = 'Signed Relative error'
                else: 
                    errors = estimates - problem.answer
                    avg_error = np.mean(errors)
                    error_std = np.std(errors)
                    error_name = 'Signed Absolute error'

                new_result = {
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
                }
                if key not in existing_keys:
                    write_results(output_file, [new_result], is_first_run)
                    is_first_run = False
                    existing_keys.add(key)

    print(f'Results saved to {output_file}')