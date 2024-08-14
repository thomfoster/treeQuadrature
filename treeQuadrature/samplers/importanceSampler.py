from .sampler import Sampler
from ..exampleProblems import Problem

import numpy as np


class ImportanceSampler(Sampler):
    def rvs(self, n: int, problem: Problem) -> np.ndarray:
        """
        Importance sampling with a bias towards edges and corners.

        Parameters
        ----------
        n : int
            Number of samples.
        problem : Problem
            The integration problem being solved.
        
        Returns
        -------
        np.ndarray
            Samples from the distribution.
        """
        # Generate uniform random samples within the domain
        samples = np.random.uniform(problem.lows, problem.highs, (n, problem.D))
        
        # Calculate distances to the edges of the domain for each sample
        distances_to_edges = np.minimum(samples - problem.lows, problem.highs - samples)
        min_distance_to_edge = np.min(distances_to_edges, axis=1, keepdims=True)
        
        # Invert distances to give higher weight to points closer to the edges/corners
        edge_bias = 1 / (min_distance_to_edge + 1e-6)  # Avoid division by zero
        
        # Combine edge bias with the original density
        densities = problem.integrand(samples)
        biased_probabilities = densities * edge_bias
        biased_probabilities /= np.sum(biased_probabilities)  # Normalize to sum to 1
        
        # Draw samples with replacement using biased probabilities
        indices = np.random.choice(np.arange(n), size=n, replace=True, p=biased_probabilities.flatten())
        
        return samples[indices]