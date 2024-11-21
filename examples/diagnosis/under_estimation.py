from treeQuadrature.example_problems import Gaussian
from treeQuadrature.container_integrators import RandomIntegral
from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.samplers import McmcSampler

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Parameters
    dimensions = range(2, 11)
    temperatures = [1.0, 1.5, 2.0]  # Different temperatures to compare
    underestimation_threshold = -50  # Relative error threshold for under-estimation (in percentage)
    proportions_by_temp = {}  # To store proportions for each temperature

    for temp in temperatures:
        proportions = [] 
        for D in dimensions:
            # Set up the problem
            N = 7_000 + 2500 * D
            problem = Gaussian(D, lows=-1.0, highs=1.0, Sigma=1 / 200)
            random = RandomIntegral(100)

            integ = TreeIntegrator(
                N, integral=random, sampler=McmcSampler(temperature=temp)
            )

            # Draw samples and build tree
            X, y = integ._draw_initial_samples(problem, False)
            root = integ._construct_root_container(X, y, problem, False)
            leaf_containers = integ._construct_tree(root, problem, False)
            print(f"Finished Tree Construction for D={D}, Temp={temp}")

            # Perform Monte Carlo Evaluation
            results, containers = integ.integrate_containers(
                leaf_containers, problem, compute_std=True
            )
            print(f"Finished Monte Carlo Evaluation for D={D}, Temp={temp}")

            # Calculate true integrals and errors
            evals_random = np.array([result["integral"] for result in results])
            true_answers = np.array([
                problem._integrate(cont.mins, cont.maxs) for cont in containers])
            errors = evals_random - true_answers
            rel_errors = (errors / true_answers) * 100  # Relative errors in percentage

            # Calculate proportion of containers with under-estimation above 50%
            underestimation_count = np.sum(rel_errors < underestimation_threshold)
            proportion = underestimation_count / len(containers)
            proportions.append(proportion)
            print(
                f"Dimension {D}, Temp {temp}: "
                f"Proportion of under-estimated containers: {proportion:.2%}")

        # Store the proportions for the current temperature
        proportions_by_temp[temp] = proportions

    # Plot the results
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12, 8))
    for temp, proportions in proportions_by_temp.items():
        plt.plot(dimensions, proportions, marker='o', label=f'Temperature {temp}')
    plt.xlabel("Dimension")
    plt.ylabel("Proportion")
    plt.title("Proportion of Containers with Under-Estimation > 50% vs Dimension")
    plt.legend(title="Temperature")
    plt.grid(True)
    plt.show()