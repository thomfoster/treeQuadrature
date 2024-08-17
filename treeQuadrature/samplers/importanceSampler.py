from .sampler import Sampler

import numpy as np
from typing import Tuple


class ImportanceSampler(Sampler):
    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray, 
            f: callable,
            **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Importance sampling with a bias towards edges and corners.

        Parameters
        ----------
        n : int
            Number of samples.
        mins, maxs : np.ndarray
            1 dimensional arrays of the lower bounds
            and upper bounds
        f : function
            the integrand
        
        Returns
        -------
        np.ndarray
            Samples from the distribution.
        """
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"n must be an integer, got {n} of type "
                            f"type(n)")
        
        mins, maxs, D = Sampler.handle_mins_maxs(mins, maxs)

        # Generate uniform random samples within the domain
        samples = np.random.uniform(mins, maxs, (n, D))
        
        # Calculate distances to the edges of the domain for each sample
        distances_to_edges = np.minimum(samples - mins, maxs - samples)
        min_distance_to_edge = np.min(distances_to_edges, axis=1, keepdims=True)
        
        # Invert distances to give higher weight to points closer to the edges/corners
        edge_bias = 1 / (min_distance_to_edge + 1e-6)  # Avoid division by zero
        
        # Combine edge bias with the original density
        densities = f(samples)
        biased_probabilities = densities * edge_bias
        biased_probabilities /= np.sum(biased_probabilities)  # Normalize to sum to 1
        
        # Draw samples with replacement using biased probabilities
        indices = np.random.choice(np.arange(n), size=n, replace=True, p=biased_probabilities.flatten())

        xs = samples[indices]
        
        return xs, f(xs)