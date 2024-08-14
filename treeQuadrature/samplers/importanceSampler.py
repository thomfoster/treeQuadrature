from .sampler import Sampler
from ..exampleProblems import Problem

import numpy as np


class ImportanceSampler(Sampler):
    """
    Importance sampler based on the integrand function

    Attributes
    ----------
    n_init : int
        size of initial samples drawn to identify the function weights
    """
    def __init__(self, n_init=2000) -> None:
        self.n_init = n_init

    def rvs(self, n: int, problem: Problem) -> np.ndarray:
        """
        Generate importance sampling random samples.
        with importance propotional to |problem.integrand|

        Parameters
        ----------
        n : int 
            Number of samples.
        problem: Problem
            The integration problem being solved.

        Returns
        -------
        np.ndarray
            Samples from the importance sampling distribution.
        """
        # Determine the number of initial samples
        n_init_used = min(self.n_init, n)
        n_remaining = n - n_init_used

        # Step 1: Initial uniform sampling
        initial_samples = np.random.uniform(low=problem.lows, high=problem.highs, 
                                            size=(n_init_used, problem.D))
        integrand_values = problem.integrand(initial_samples).reshape(-1)
        weights = np.abs(integrand_values)
        probabilities = weights / weights.sum()

        # Step 2: Importance sampling
        if n_remaining > 0:
            indices = np.random.choice(np.arange(n_init_used), size=n_remaining, p=probabilities)
            importance_samples = initial_samples[indices]
            combined_samples = np.vstack((initial_samples, importance_samples))
        else:
            # If no remaining evaluations are allowed, 
            # just perform importance sampling within initial samples
            indices = np.random.choice(np.arange(n_init_used), size=n, p=probabilities)
            combined_samples = initial_samples[indices]

        # Ensure exactly n samples are returned
        return combined_samples[:n]
