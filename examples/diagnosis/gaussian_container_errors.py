from treeQuadrature.example_problems import Gaussian
from treeQuadrature.container_integrators import RandomIntegral
from treeQuadrature.integrators import DistributedTreeIntegrator, TreeIntegrator
from treeQuadrature.integrators.distributed_tree_integrator import max_side_length
from treeQuadrature.samplers import McmcSampler
from treeQuadrature.trees import SimpleTree
from treeQuadrature.splits import MinSseSplit, sse_score

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def _type_check(input, name):
    if isinstance(input, list):
        return np.array(input)
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError(
            f"{name} must either be a list or numpy.ndarray")

def plot_errors_vs_indicators(errors: np.ndarray, indicators: np.ndarray,
                              indicator_name: str,
                              title: Optional[str] = None,
                              error_name: Optional[str] = None,
                              negative: bool = False):
    """
    Plots errors against indicators (e.g., volumes) with different colors for
    positive, negative, and unbiased (near-zero) errors. Optionally, it can
    focus only on negative errors.

    Parameters
    ----------
    errors : np.ndarray or list
        Error values for each container.
    indicators : np.ndarray or list
        Some indicator for each container.
    indicator_name : str
        Name of the indicator displayed on the x-axis.
    title : str, optional
        Title of the plot. If not provided, defaults to 
        "Container Errors vs <indicator_name>".
    error_name : str, optional
        y-label of the plot. If not provided, defaults to 
        "Errors".
    focus_on_negative : bool, optional
        If True, ignores positive errors and only plots negative and
        unbiased (near-zero) errors. Defaults to False.
    """ 
    errors = _type_check(errors, "errors")
    indicators = _type_check(indicators, "indicators")
    if errors.shape[0] != indicators.shape[0]:
        raise ValueError(
            "The length of errors and indicators must be the same")

    # Set larger font size for all plot elements
    plt.rcParams.update({'font.size': 14})

    # Define threshold for unbiased errors based on the focus mode
    if negative:
        max_negative_error = np.min(errors[errors < 0])
        unbiased_threshold = 0.01 * abs(max_negative_error)
    else:
        max_error = np.max(np.abs(errors))
        unbiased_threshold = 0.01 * max_error

    # Separate errors into positive, negative, and unbiased categories
    pos_errors = errors > unbiased_threshold
    neg_errors = errors < -unbiased_threshold
    unbiased_errors = (errors >= -unbiased_threshold) & (errors <= unbiased_threshold)

    # Plot errors with different colors
    plt.figure(figsize=(10, 5))
    plt.scatter(indicators[unbiased_errors], errors[unbiased_errors],
                color='grey', alpha=0.7, label='Unbiased Errors')

    plt.scatter(indicators[neg_errors], errors[neg_errors], color='r', alpha=0.7,
                label='Negative Errors')
    if not negative:
        # Plot positive, negative, and unbiased errors
        plt.scatter(indicators[pos_errors], errors[pos_errors], color='b', alpha=0.7,
                    label='Positive Errors')

    # Set plot labels and title
    plt.xlabel(indicator_name)
    error_name = "Errors" if error_name is None else error_name
    plt.ylabel(error_name)
    title = f"Container Errors vs {indicator_name}" if title is None else title
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    D = 10
    N = 7_000 + 2500*D
    problem = Gaussian(D, lows=-1.0, highs=1.0, Sigma=1/200)
    random = RandomIntegral(1000)

    # draw samples and build tree
    # integ = DistributedTreeIntegrator(N, 20_000+D*10_000,
    #                                   sampler=McmcSampler(temperature=2.0),
    #                                   min_container_samples=20,
    #                                   max_container_samples=400)
    tree_sse = SimpleTree(MinSseSplit(scoring_function=sse_score))
    integ = TreeIntegrator(N, integral=random,
                           sampler=McmcSampler(temperature=2.0),
                           tree=tree_sse)
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
    dists_mode = []

    for cont in containers:
        true_answers.append(problem._integrate(cont.mins, cont.maxs))
        dists_mode.append(np.linalg.norm(cont.midpoint))

    true_answers = np.array(true_answers)
    errors = np.array(evals_random) - true_answers
    rel_errors = errors / true_answers * 100
    
    print(f"True answer: {1.0}")
    print(f"scipy answer: {problem.answer}")
    print(f"scipy estimate: {sum(true_answers)}")
    print(f"estimate using mean: {sum(evals_random)}")

    # Set larger font size for all plot elements
    plt.rcParams.update({'font.size': 14})

    # n_eval_samples = [result["sample_size"] for result in results]
    estimated_errors = [result["std"] for result in results]
    max_sides = [max_side_length(cont) for cont in containers]
    
    indicator_lists = [dists_mode, max_sides, estimated_errors]
    short_names = ["Distance to Mode", "Maximum Side Length", "Estimated std"]
    long_names = ["Distance of Midpoint to Mode",
                  "Length of the longest edge",
                  "Monte Carlo Standard Deviation"]

    for indicators, short_name, long_name in zip(
        indicator_lists, short_names, long_names
    ):
        # absolute errors
        plot_errors_vs_indicators(errors, indicators, short_name,
                               title=f"Container Errors vs {long_name}")
        # relative negative errors
        # plot_errors_vs_indicators(rel_errors, indicators, short_name,
        #                         title=f"Container Errors vs {long_name}",
        #                         error_name="Relative Error (%)",
        #                         negative=True)
