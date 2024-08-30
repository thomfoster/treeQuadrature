from .exampleProblems import Problem
from .integrators import TreeIntegrator, Integrator
from .containerIntegration import ContainerIntegral
from .visualisation import plotContainers
from .container import Container

import warnings, time, csv, os, multiprocessing, itertools

from inspect import signature
import numpy as np
from typing import List, Optional, Any
from traceback import print_exc


def integrator_wrapper(integrator, problem, verbose, result_queue, specific_kwargs={}):
    parameters = signature(integrator).parameters
    try:
        if verbose >= 2 and 'verbose' in parameters:
            result = integrator(problem, return_N=True, verbose=True, **specific_kwargs)
        else:
            result = integrator(problem, return_N=True, **specific_kwargs)
        result_queue.put({'result': result})
    except Exception as e:
        result_queue.put({'exception': e})

def compare_integrators(integrators: List[Integrator], problem: Problem, 
                        plot: bool=False, 
                        verbose: int=1, 
                        xlim: Optional[List[float]]=None, 
                        ylim: Optional[List[float]]=None,
                        dimensions: Optional[List[float]]=None,
                        n_repeat: int=1, integrator_specific_kwargs: Optional[dict]=None, 
                        **kwargs: Any) -> None:
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
    plot_samples : bool, optional
        Whether to plot samples on the figure produced
        Default is True.
        Will be ignored if plot = False
    verbose : int, optional
        If 0, print no message;
        if 1, print the test runrs;
        if 2, print the messages within integrators
        Default is 1.
    xlim, ylim : List[float], optional
        The limits for the plot containers.
        Will not be used when plot=False.
    dimensions : List[Float], optional
        Which dimensions to plot for higher dimensional problems.
    n_repeat : int, optional
        Number of times to repeat the integration and average the results.
        Default is 1.
    integrator_specific_kwargs : dict, optional
        A dictionary where the keys are names of integrator and the values are
        dictionaries of specific arguments to be passed to those integrators.
        Default is None.
    **kwargs : Any
        kwargs that should be used by integrator.__call__ method 
        or the plotContainers method
    """
    if integrator_specific_kwargs is None:
        integrator_specific_kwargs = {}

    for i, integrator in enumerate(integrators):
        integrator_name = getattr(integrator, 'name', f'integrator[{i}]')

        if verbose >= 1:
            print(f'Testing {integrator_name}')

        estimates = []
        n_evals_list = []
        times = []

        integrator_params = signature(integrator.__call__).parameters
        applicable_kwargs = {k: v for k, v in kwargs.items() if k in integrator_params}

        if hasattr(integrator, 'tree'):
            construct_tree_params = signature(integrator.tree.construct_tree).parameters
            applicable_kwargs.update({k: v for k, v in kwargs.items() if k in construct_tree_params})

        if hasattr(integrator, 'integral'):
            container_params = signature(integrator.integral.containerIntegral).parameters
            applicable_kwargs.update({k: v for k, v in kwargs.items() if k in container_params})

        # Prepare common arguments
        integration_args = {'return_N': True}

        if 'return_containers' in integrator_params:
            integration_args['return_containers'] = True

        if 'verbose' in integrator_params:
            integration_args['verbose'] = verbose >= 2

        # Merge common arguments with applicable kwargs
        integration_args.update(applicable_kwargs)
        integration_args.update(integrator_specific_kwargs.get(integrator_name, {}))

        for i in range(n_repeat):
            if verbose >= 1:
                print(f'Run {i}')
            start_time = time.time()
            try:
                # Perform integration
                result = integrator(problem, **integration_args)
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

        if len(estimates) == 0:
            raise Exception('no run succeeded')

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
        print(f'True answer of {str(problem)}: {problem.answer}')
        print(f'Estimated value: {avg_estimate:.4f} ± {std_estimate:.4f}')
        print(f'{error_name}: {avg_error:.2f} % ± {std_error:.2f} %')
        print(f'Number of evaluations: {avg_n_evals:.2f} ± {std_n_evals:.2f}')
        print(f'Time taken: {avg_time:.2f} s ± {std_time:.2f} s')

        if 'containers' in result and 'contributions' in result:
            default_title = 'Integral estimate using ' + integrator_name
            containers = result['containers']
            print(f'Number of containers: {len(containers)}')
            n_samples = [cont.N for cont in containers]
            print(f'Average samples/container: {np.mean(n_samples)}')
            print(f'Minimum samples in containers: {np.min(n_samples)}')
            print(f'Maximum samples in containers: {np.max(n_samples)}')
            contributions = result['contributions']
            if plot:
                if xlim is None:
                    raise ValueError('xlim must be provided for plotting')
                plot_params = signature(plotContainers).parameters
                applicable_kwargs = {k: v for k, v in kwargs.items() if k in plot_params}
                title = applicable_kwargs.pop('title', default_title)
                plotContainers(containers, contributions, 
                               xlim=xlim, ylim=ylim,
                               integrand=problem.integrand, 
                               title=title, dimensions=dimensions, 
                               **applicable_kwargs)
        elif plot: 
            warnings.warn('Result of integrator has no containers to plot', 
                          UserWarning)
        
        print(f'----------------------------------')


## protection to code interruption
def load_existing_results(output_file: str) -> dict:
    if not os.path.exists(output_file):
        return {}
    with open(output_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        return {(row['integrator'], row['problem']): row for row in reader}
    
def write_results(output_file: str, results: List[dict], write_header: bool, fieldnames: List[str], 
                  mode: str='a'):
    with open(output_file, mode=mode, newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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
                     n_repeat: int=1, 
                     integrator_specific_kwargs: Optional[dict] = None, 
                     retest_integrators: List[str]=[]) -> None:
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
    retest_integrators: List[str], optional
        Names of integrators that needs to be retested.
    """

    np.random.seed(seed)

    existing_results = load_existing_results(output_file)

    if existing_results:
        is_first_run = False
    else:
        is_first_run = True

    results = []

    fieldnames = [
            'integrator', 'problem', 'true_value', 'estimate', 'estimate_std', 'error_type', 
            'error', 'error_std', 'n_evals', 'n_evals_std', 'time_taken', 'errors']

    for problem in problems:
        problem_name = str(problem)

        if verbose >= 1:
            print(f'testing Probelm: {problem_name}')

        for i, integrator in enumerate(integrators):
            integrator_name = getattr(integrator, 'name', f'integrator[{i}]')

            # Check if the result already exists and is valid
            key = (integrator_name, problem_name)
            if key in existing_results and existing_results[key]['estimate'] != '' and (
                existing_results[key]['integrator'] not in retest_integrators
            ):
                if verbose >= 1:
                    print(f'Skipping {integrator_name} for {problem_name}: already completed.')
                results.append(existing_results[key])
                continue

            if verbose >= 1:
                print(f'testing Integrator: {integrator_name}')

            estimates = []
            n_evals_list = []
            total_time_taken = 0

            specific_kwargs = integrator_specific_kwargs.get(integrator.name, {}).copy()

            if 'integrand' in specific_kwargs:
                specific_kwargs['integrand'] = problem.integrand

            for repeat in range(n_repeat):
                np.random.seed(seed + repeat)
                start_time = time.time()
                
                result_queue = multiprocessing.Queue()

                process = multiprocessing.Process(
                    target=integrator_wrapper, 
                    args=(integrator, problem, verbose, result_queue, specific_kwargs)
                )
                process.start()
                process.join(timeout=max_time)

                if process.is_alive():
                    print(f'Time limit exceeded for {integrator_name} on {problem_name}, '
                          'increase max_time or change the problem/integrator')
                    process.terminate()
                    process.join()  # Ensure process is fully terminated
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
                        'time_taken': f'Exceeded {max_time}s',
                        'errors': None
                    }
                    break
                else:
                    result_dict = result_queue.get()
                    if 'exception' in result_dict:
                        print(result_dict['exception'])
                        new_result = {
                            'integrator': integrator_name,
                            'problem': problem_name,
                            'true_value': problem.answer,
                            'estimate': None,
                            'estimate_std': None,
                            'error_type': result_dict['exception'],
                            'error': None,
                            'error_std': None,
                            'n_evals': None,
                            'n_evals_std': None,
                            'time_taken': None, 
                            'errors': None
                        }
                        break
                    else:
                        result = result_dict['result']
                        end_time = time.time()
                        time_taken = end_time - start_time
                        total_time_taken += time_taken

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
    
            # Update the existing results
            existing_results[key] = new_result

            # Write results incrementally to ensure recovery
            write_results(output_file, [new_result], 
                            is_first_run, fieldnames)
            is_first_run = False

    write_results(output_file, list(existing_results.values()), True, fieldnames, mode='w')

    print(f'Results saved to {output_file}')



def _test_integrals_single_problem(problem, integrals: ContainerIntegral, 
                   integrator: TreeIntegrator, verbose: int):
    """Test container integrators on the same tree"""
    return_values = [[] for _ in range(len(integrals))]

    # construct tree
    X, y = integrator.sampler.rvs(integrator.base_N, problem.lows, 
                                    problem.highs,
                                    problem.integrand)
        
    root = Container(X, y, mins=problem.lows, maxs=problem.highs)
    finished_containers = integrator.construct_tree(root)
    
    for i, integral in enumerate(integrals):
        integrator.integral = integral

        if verbose >= 2:
            print(f"integrating {integrals[i].name}")
        start_time = time.time()
        results, containers = integrator.integrate_containers(finished_containers, 
                                                              problem)
        end_time = time.time()
        if verbose >= 2:
            print(f"completed, took {end_time - start_time}s")

        N = sum([cont.N for cont in containers])
        contributions = [result['integral'] for result in results]
        estimate = np.sum(contributions)
        return_values[i] = {'estimate' : estimate, 
                            'n_evals' : N,
                            'time' : end_time - start_time}
        
    
    return return_values


def test_container_integrals(problems: List[Problem],  integrals: ContainerIntegral, 
                             integrator: TreeIntegrator, output_file: str, n_repeat : int, 
                             verbose: int=1) -> None:
    """
    Test different container integrals on a list of problems 
    and save the results to a CSV file.

    Parameters
    ----------
    problems : List[Problem]
        A list of problem instances containing 
        the integrand and true answer.
    integrals : ContainerIntegral
        A list of container integrals to be tested.
    integrator : TreeIntegrator
        The integrator instance used to perform the integrations.
    output_file : str
        The file path to save the results as a CSV.
    n_repeat : int
        Number of times to repeat the integration and average the results.
    verbose : int, optional
        Level of verbosity (default is 1):
            0 - print no messages.
            1 - print basic progress messages.
            2 - print detailed progress messages, including each repetition.
    """
    existing_results = load_existing_results(output_file)
    n_integrals = len(integrals)
    integral_names = [integral.name for integral in integrals]

    if existing_results:
        is_first_run = False
    else:
        is_first_run = True
    
    final_results = {}

    fieldnames = [
            'integrator', 'problem', 'true_value', 'estimate', 'estimate_std', 'error_type', 
            'error', 'error_std', 'n_evals', 'n_evals_std', 'time_taken', 'errors']

    for problem in problems:
        problem_name = str(problem)
        if verbose >= 1:
            print(f'testing Probelm: {problem_name}')

        key = (integral_names[0], problem_name)
        if key in existing_results and existing_results[key]['estimate'] != '':
            if verbose >= 1:
                print(f'Skipping {problem_name}: already completed.')
            final_results[key] = existing_results[key]
            continue
        
        estimates = [[] for _ in range(n_integrals)]
        n_evals_list = [[] for _ in range(n_integrals)]
        time_list = [[] for _ in range(n_integrals)]

        for repeat in range(n_repeat):
            if verbose >= 2:
                print(f"Repeat {repeat}")
            try: 
                results = _test_integrals_single_problem(problem, integrals, integrator, verbose)
            except Exception as e:
                print(f'Error during integration on {problem_name}: {e}')
                print_exc()
                break

            for j in range(n_integrals):
                estimates[j].append(results[j]['estimate'])
                n_evals_list[j].append(results[j]['n_evals'])
                time_list[j].append(results[j]['time'])
            

        if len(estimates[0]) == n_repeat:
            new_results = []
            for i in range(n_integrals):
                integral_estimates = np.array(estimates[i])
                avg_estimate = np.mean(integral_estimates)
                avg_n_evals = int(np.mean(n_evals_list[i]))
                    
                errors = 100 * (
                    integral_estimates - problem.answer) / problem.answer
                avg_error = f'{np.median(errors):.4f} %'
                error_std = f'{np.std(errors):.4f} %'
                error_name = 'Signed Relative error'

                new_results.append({
                    'integrator': integral_names[i],
                    'problem': problem_name,
                    'true_value': problem.answer,
                    'estimate': avg_estimate,
                    'estimate_std': np.std(estimates),
                    'error_type': error_name,
                    'error': avg_error,
                    'error_std': error_std,
                    'n_evals': avg_n_evals,
                    'n_evals_std': np.std(n_evals_list),
                    'time_taken' : np.mean(time_list[i]),
                    'errors': errors
                })
            
            # Update the existing results
            for i in range(n_integrals):
                final_results[(integral_names[i], problem_name)] = new_results[i]

            # Write results incrementally to ensure recovery
            write_results(output_file, new_results, 
                            is_first_run, fieldnames)
        is_first_run = False
        
    write_results(output_file, list(final_results.values()), True, fieldnames, mode='w')
    print(f'Results saved to {output_file}')



def test_integrator_performance_with_params(integrator: Integrator, 
                                            problem: Problem, 
                                            param_grid: dict, 
                                            output_file: str='results.csv', 
                                            max_time: float=60.0, 
                                            verbose: int=1, 
                                            seed: int=2024, 
                                            n_repeat: int=10, 
                                            **kwargs: Any) -> None:
    """
    Test the performance of a single integrator on a problem 
    with varying parameter values.

    Parameters
    ----------
    integrator : Integrator
        The integrator instance to be tested.
    problem : Problem
        The problem instance containing the integrand and true answer.
    param_grid : dict
        Dictionary where keys are parameter names and values are lists of parameter values to test.
        Example: {'base_N': [1000, 5000], 'P': [40, 60]}
    output_file : str, optional
        The file path to save the results as a CSV. Default is 'results.csv'.
    max_time : float, optional
        Maximum allowed time (in seconds) for each integration. Default is 60.0 seconds.
    verbose : int, optional
        if 1, print the problem and integrator being tested; 
        if 2, print details of each integrator as well; 
        if 0, print nothing. Default is 1.
    seed : int, optional
        specify the randomness seed for reproducibility. Default is 2024.
    n_repeat : int, optional
        Number of times to repeat the integration and average the results. Default is 1.
    kwargs : Any
        arguments required for integrator.__call__ method
    """

    np.random.seed(seed)

    existing_results = load_existing_results(output_file)

    if existing_results:
        is_first_run = False
    else:
        is_first_run = True

    results = []

    fieldnames = [
            'base_N', 'P', 'problem', 'true_value', 'estimate', 'estimate_std', 'error_type', 
            'error', 'error_std', 'n_evals', 'n_evals_std', 'time_taken', 'errors']

    # Check if the integrator has the parameters specified in param_grid
    for param in param_grid:
        if not hasattr(integrator, param):
            raise ValueError(f"Integrator does not have a attribute '{param}' specified in param_grid")

    # Generate combinations of parameter values
    param_names = list(param_grid.keys())
    param_combinations = [dict(zip(param_names, values)) 
                          for values in itertools.product(*param_grid.values())]

    for params in param_combinations:
        key = tuple(params.items())

        if key in existing_results and existing_results[key]['estimate'] != '':
            if verbose >= 1:
                print(f'Skipping combination {params}: already completed.')
            results.append(existing_results[key])
            continue
        
        if verbose >= 1:
            print(f'Testing integrator with parameters {params}')

        integrator.base_N = params['base_N']
        integrator.P = params['P']

        estimates = []
        n_evals_list = []
        total_time_taken = 0

        for repeat in range(n_repeat):
            np.random.seed(seed + repeat)
            start_time = time.time()
            
            result_queue = multiprocessing.Queue()

            process = multiprocessing.Process(
                target=integrator_wrapper, 
                args=(integrator, problem, verbose, result_queue),
                kwargs=kwargs
            )
            process.start()
            process.join(timeout=max_time)

            if process.is_alive():
                print(f'Time limit exceeded for {integrator.name} with {params}, '
                      'increase max_time or change the problem/integrator')
                process.terminate()
                process.join()  # Ensure process is fully terminated
                new_result = {
                    'base_N': params.get('base_N', None),
                    'P': params.get('P', None),
                    'problem': str(problem),
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
                }
                break
            else:
                result_dict = result_queue.get()
                if 'exception' in result_dict:
                    print(result_dict['exception'])
                    new_result = {
                        'base_N': params.get('base_N', None),
                        'P': params.get('P', None),
                        'problem': str(problem),
                        'true_value': problem.answer,
                        'estimate': None,
                        'estimate_std': None,
                        'error_type': result_dict['exception'],
                        'error': None,
                        'error_std': None,
                        'n_evals': None,
                        'n_evals_std': None,
                        'time_taken': None, 
                        'errors': None
                    }
                    break
                else:
                    result = result_dict['result']
                    end_time = time.time()
                    time_taken = end_time - start_time
                    total_time_taken += time_taken

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

            new_result = {
                'base_N': params.get('base_N', None),
                'P': params.get('P', None),
                'problem': str(problem),
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

        # Update the existing results
        existing_results[key] = new_result

        # Write results incrementally to ensure recovery
        write_results(output_file, [new_result], is_first_run, fieldnames)
        is_first_run = False

    write_results(output_file, list(existing_results.values()), True, fieldnames, mode='w')

    print(f'Results saved to {output_file}')