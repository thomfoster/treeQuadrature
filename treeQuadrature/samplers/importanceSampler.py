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
        # Initial uniform sampling (to find weights)
        initial_samples = np.random.uniform(low=problem.lows, high=problem.highs, 
                                            size=(self.n_init, problem.D))
        integrand_values = problem.integrand(initial_samples).reshape(-1)
        weights = np.abs(integrand_values)
        probabilities = weights / weights.sum()

        # Importance sampling
        num_additional_samples = n - self.n_init
        indices = np.random.choice(np.arange(self.n_init), size=num_additional_samples, p=probabilities)
        importance_samples = initial_samples[indices]

        # Step 3: Combine initial samples with importance samples
        combined_samples = np.vstack((initial_samples, importance_samples))

        return combined_samples
