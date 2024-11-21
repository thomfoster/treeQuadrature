from treeQuadrature.example_problems import Gaussian, Problem
from treeQuadrature.container_integrators import ProgressiveRandomIntegral
from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.samplers import McmcSampler
from treeQuadrature import Container
from gaussian_container_errors import plot_errors_vs_indicators

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable


def iterative_sampling_and_plot(
    containers: List[Container],
    errors: np.ndarray,
    true_values: np.ndarray,
    integrator: ProgressiveRandomIntegral,
    problem: Problem,
    additional_samples: int,
    K: int=5,
    iterations: int=10,
    negative: bool=False,
    relative: bool=False
):
    """
    Selects top K containers with the largest errors,
    iteratively draws samples,
    and plots error vs. sample size for each container.

    Parameters:
    ----------
    containers : List[Container]
        List of container objects.
    errors : np.ndarray
        Initial error values for each container.
    integrator : ProgressiveRandomIntegral
        The integrator object to perform sampling and calculate integrals.
    problem : Problem
        the integration problem being solved
    additional_samples : int
        Number of additional samples to draw in each iteration.
    K : int, optional
        Number of containers to select based on the largest errors. Default is 5.
    iterations : int, optional
        Number of iterations to perform additional sampling. Default is 10.
    negative : bool, optional
        If true, only focus on the under-estimations
    relative : bool, optional
        If true, record the relative errors
    """

    # select containers with largest errors
    if negative and relative:
        scores = - errors / true_answers
        scores[scores < 0] = 0
    elif negative:
        scores = - errors
    elif relative:
        scores = np.abs(errors) / true_answers
    else:
        scores = np.abs(errors)

    top_indices = np.argsort(-scores)[:K]
    top_containers = [containers[i] for i in top_indices]
    true_top_values = [true_values[i] for i in top_indices]

    # Initialise data storage for plotting
    error_history = {i: [] for i in range(K)}
    sample_size_history = {i: [] for i in range(K)}

    for i, idx in enumerate(top_indices):
        if relative:
            error_history[i].append(
                errors[idx] / true_answers[idx] * 100)
        else:
            error_history[i].append(errors[idx])
        sample_size_history[i].append(integrator.n_samples)

    # Perform iterative sampling for each selected container
    for i, container in enumerate(top_containers):
        true_value = true_top_values[i]
        sample_count = integrator.n_samples
        
        # Iteratively add samples and record error
        for _ in range(iterations):
            # Draw additional samples and update integral estimate
            result = integrator.containerIntegral(
                container, problem.integrand,
                additional_samples=additional_samples)
            estimated_integral = result["integral"]
            sample_count += additional_samples
            
            # Calculate and record the error
            if relative:
                current_error = (
                    estimated_integral - true_value
                    ) / true_value * 100
            else:
                current_error = estimated_integral - true_value
            error_history[i].append(current_error)
            sample_size_history[i].append(sample_count)
    
    # Plot error vs. sample size for each container
    plt.figure(figsize=(12, 8))
    for i in range(K):
        plt.plot(sample_size_history[i], error_history[i])

    if relative:
        error_name = 'Relative Errors (%)'
    else:
        error_name = 'Errors'

    title = error_name+' vs. Sample Size for Containers with largest errors'

    plt.xlabel('Sample Size')
    plt.ylabel(error_name)
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    D = 10
    N = 7_000 + 2500*D
    problem = Gaussian(D, lows=-1.0, highs=1.0, Sigma=1/200)
    random = ProgressiveRandomIntegral(n_samples=30)

    # draw samples and build tree
    integ = TreeIntegrator(N, integral=random,
                           sampler=McmcSampler(temperature=2.0))
    X, y = integ._draw_initial_samples(problem, False)
    root = integ._construct_root_container(
        X, y, problem, False)
    leaf_containers = integ._construct_tree(
            root, problem, False)
    print("Finished Tree Construction")

    results, containers = integ.integrate_containers(
        leaf_containers, problem, compute_std=True)
    print("Finished Monte Carlo Evaluation")
    evals_random = [result["integral"] for result in results]

    true_answers = []

    for cont in containers:
        true_answers.append(problem._integrate(cont.mins, cont.maxs))

    true_answers = np.array(true_answers)
    errors = np.array(evals_random) - true_answers
    errors_rel = errors / true_answers * 100
    estimated_errors = np.array([result["std"] for result in results])
    
    n_samples = sum([cont.N for cont in containers])
    print(f"True answer: {1.0}")
    print(f"scipy answer: {problem.answer}")
    print(f"scipy estimate: {sum(true_answers)}")
    print(f"estimate using mean: {sum(evals_random)}")
    print(f"Number of evaluations: {n_samples}")

    errors_rel_negative = errors_rel[errors_rel < 0]
    estimated_errors_negative = estimated_errors[errors_rel < 0]
    plot_errors_vs_indicators(errors_rel_negative, estimated_errors_negative, "estimated error",
                              title="Negative Container Errors (%) vs Monte Carlo Standard Deviation")

    plt.rcParams.update({'font.size': 14})
    iterative_sampling_and_plot(containers, errors, true_answers,
                                random, problem,
                                additional_samples=40,
                                K=50, negative=True, relative=True)
