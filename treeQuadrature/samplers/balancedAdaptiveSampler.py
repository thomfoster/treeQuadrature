from typing import Tuple
import numpy as np

from .sampler import Sampler


class BalancedAdaptiveSampler(Sampler):
    def __init__(self, strata_per_dim: int = 5, refinement_threshold: float = 0.1, 
                 max_samples_per_stratum: int = 10):
        """
        Initialize the BalancedAdaptiveSampler.

        Parameters
        ----------
        strata_per_dim : int, optional
            Number of strata (subdivisions) per dimension.
        refinement_threshold : float, optional
            Threshold for refining a stratum based on the integrand's value.
        max_samples_per_stratum : int, optional
            Maximum number of samples allowed per stratum to prevent over-sampling.
        """
        self.strata_per_dim = strata_per_dim
        self.refinement_threshold = refinement_threshold
        self.max_samples_per_stratum = max_samples_per_stratum

    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray, 
            f: callable, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balanced adaptive sampling to ensure coverage of the entire domain and prevent over-sampling.

        Parameters
        ----------
        n : int
            Number of samples.
        mins : np.ndarray
            1-dimensional array of the lower bounds of the domain.
        maxs : np.ndarray
            1-dimensional array of the upper bounds of the domain.
        f : callable
            The integrand function to be sampled.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            xs : np.ndarray of shape (n, D)
                The sampled points.
            ys : np.ndarray of shape (n, )
                The integrand values at the sampled points.
        """
        D = len(mins)
        strata = [(mins, maxs)]
        samples = []
        values = []

        # Calculate the number of samples per stratum to ensure at least one sample per stratum
        samples_per_stratum = max(1, n // (self.strata_per_dim ** D))

        for _ in range(self.strata_per_dim):
            new_strata = []
            for low, high in strata:
                # Subdivide each stratum
                sub_strata = self.subdivide_stratum(low, high)
                for sub_low, sub_high in sub_strata:
                    # Sample within each sub-stratum with a cap on the maximum number of samples
                    sub_samples = np.random.uniform(sub_low, sub_high, 
                                                    (min(samples_per_stratum, self.max_samples_per_stratum), D))
                    sub_values = f(sub_samples)
                    # Include these samples in the final output
                    samples.append(sub_samples)
                    values.append(sub_values)
                    # Only refine if the average integrand value in this stratum exceeds the threshold
                    if np.mean(sub_values) > self.refinement_threshold:
                        new_strata.append((sub_low, sub_high))
            strata = new_strata

        xs = np.vstack(samples)
        ys = np.concatenate(values)
        
        # Trim or expand samples to exactly n if needed
        if xs.shape[0] > n:
            indices = np.random.choice(xs.shape[0], n, replace=False)
            xs = xs[indices]
            ys = ys[indices]
        elif xs.shape[0] < n:
            extra_samples = n - xs.shape[0]
            additional_samples = np.random.uniform(mins, maxs, (extra_samples, D))
            additional_values = f(additional_samples)
            xs = np.vstack([xs, additional_samples])
            ys = np.concatenate([ys, additional_values])
        
        return xs, ys
    
    def subdivide_stratum(self, low: np.ndarray, high: np.ndarray) -> list:
        """
        Subdivide a stratum into smaller sub-strata.

        Parameters
        ----------
        low, high : np.ndarray
            Lower and upper bounds of the stratum.

        Returns
        -------
        list of tuples
            Subdivided strata as a list of (low, high) tuples.
        """
        divisions = np.linspace(0, 1, self.strata_per_dim + 1)
        sub_strata = []
        for i in range(self.strata_per_dim):
            sub_low = low + divisions[i] * (high - low)
            sub_high = low + divisions[i + 1] * (high - low)
            sub_strata.append((sub_low, sub_high))
        return sub_strata