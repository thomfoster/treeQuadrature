from .sampler import Sampler

from typing import Tuple
import numpy as np
from scipy.stats import qmc


class AdaptiveImportanceSampler(Sampler):
    def __init__(self, strata_per_dim: int = 10, oversample_factor: int = 5):
        """
        Initialize the TwoStageSampler.

        Parameters
        ----------
        strata_per_dim : int, optional
            Number of strata (subdivisions) per dimension for initial adaptive sampling.
        oversample_factor : int, optional
            Factor by which to oversample initially to ensure good coverage of high-value regions.
        """
        self.strata_per_dim = strata_per_dim
        self.oversample_factor = oversample_factor

    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray, 
            f: callable, max_samples_per_region: int = 50, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-stage sampling that first adapts to the integrand's values and then applies importance sampling.

        Parameters
        ----------
        n : int
            Number of samples.
        mins, maxs : np.ndarray
            1-dimensional arrays of the lower bounds and upper bounds.
        f : callable
            The integrand function to be sampled.
        max_samples_per_region : int, optional
            Maximum number of samples allowed in any given region to prevent over-sampling.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            xs : np.ndarray of shape (n, D)
                The sampled points.
            ys : np.ndarray of shape (n, )
                The integrand values at the sampled points.
        """
        mins, maxs, D = Sampler.handle_mins_maxs(mins, maxs)

        # Stage 1: Adaptive Sampling
        total_strata = self.strata_per_dim ** D
        samples_per_stratum = max(1, (n * self.oversample_factor) // total_strata)

        xs_stage1 = []
        ys_stage1 = []

        # Divide the domain into strata and sample within each stratum
        strata = self.subdivide_domain(mins, maxs)
        for (low, high) in strata:
            stratum_samples = np.random.uniform(low, high, (samples_per_stratum, D))
            stratum_values = f(stratum_samples)

            xs_stage1.append(stratum_samples)
            ys_stage1.append(stratum_values)

        xs_stage1 = np.vstack(xs_stage1)
        ys_stage1 = np.concatenate(ys_stage1).reshape(-1)

        # Stage 2: Importance Sampling on the sampled points from Stage 1
        densities = np.abs(ys_stage1)
        probabilities = densities / np.sum(densities)  # Normalize to sum to 1

        # Ensure that there are enough non-zero probability samples
        non_zero_indices = np.nonzero(probabilities)[0]
        if len(non_zero_indices) < n:
            # If not enough, fallback to uniform sampling for remaining samples
            uniform_samples = np.random.uniform(mins, maxs, 
                                                (n - len(non_zero_indices), D))
            xs = np.vstack([xs_stage1[non_zero_indices], uniform_samples])
            ys = np.concatenate([ys_stage1[non_zero_indices], f(uniform_samples)])
        else:
            # Perform importance sampling
            indices = np.random.choice(non_zero_indices, size=n, replace=False, 
                                       p=probabilities[non_zero_indices])
            xs = xs_stage1[indices]
            ys = ys_stage1[indices]

        # Enforce the max_samples_per_region constraint
        final_xs = []
        final_ys = []
        seen_samples = {}

        for x, y in zip(xs, ys):
            rounded_x = tuple(np.round(x, decimals=6))
            if rounded_x not in seen_samples:
                seen_samples[rounded_x] = 0

            if seen_samples[rounded_x] < max_samples_per_region:
                final_xs.append(x)
                final_ys.append(y)
                seen_samples[rounded_x] += 1

            if len(final_xs) >= n:
                break

        final_xs = np.array(final_xs)[:n]
        final_ys = np.array(final_ys)[:n]

        return final_xs, final_ys

    def subdivide_domain(self, mins: np.ndarray, maxs: np.ndarray) -> list:
        """
        Subdivide the domain into smaller strata.

        Parameters
        ----------
        mins : np.ndarray
            Lower bounds of the domain.
        maxs : np.ndarray
            Upper bounds of the domain.

        Returns
        -------
        list of tuples
            Subdivided strata as a list of (low, high) tuples.
        """
        strata = []
        for indices in np.ndindex(*(self.strata_per_dim,) * len(mins)):
            sub_low = mins + (maxs - mins) * np.array(indices) / self.strata_per_dim
            sub_high = mins + (maxs - mins) * (np.array(indices) + 1) / self.strata_per_dim
            strata.append((sub_low, sub_high))
        return strata



class LHSImportanceSampler(Sampler):
    def __init__(self, oversample_factor: int = 10, epsilon: float = 1e-10):
        """
        Initialize the LHSImportanceSampler.

        Parameters
        ----------
        oversample_factor : int, optional
            Factor by which to oversample initially to ensure good coverage of high-value regions.
        epsilon : float, optional
            Small value to ensure probabilities are not exactly zero.
        """
        self.oversample_factor = oversample_factor
        self.epsilon = epsilon

    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray, 
            f: callable, max_samples_per_region: int = 50, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Latin Hypercube Sampling (LHS) combined with importance sampling.

        Parameters
        ----------
        n : int
            Number of samples.
        mins, maxs : np.ndarray
            1-dimensional arrays of the lower bounds and upper bounds.
        f : callable
            The integrand function to be sampled.
        max_samples_per_region : int, optional
            Maximum number of samples allowed in any given region to prevent over-sampling.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            xs : np.ndarray of shape (n, D)
                The sampled points.
            ys : np.ndarray of shape (n, )
                The integrand values at the sampled points.
        """
        mins, maxs, D = Sampler.handle_mins_maxs(mins, maxs)

        # Stage 1: Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=D)
        lhs_samples = sampler.random(n * self.oversample_factor)
        lhs_samples = qmc.scale(lhs_samples, mins, maxs)

        # Evaluate the function at LHS samples
        lhs_values = np.abs(f(lhs_samples))

        # Clip probabilities to avoid zeros
        probabilities = lhs_values / np.sum(lhs_values)
        probabilities = np.clip(probabilities, self.epsilon, None)

        # Normalize probabilities to sum to 1
        probabilities /= np.sum(probabilities)

        # Ensure that there are enough non-zero probability samples
        non_zero_indices = np.nonzero(probabilities)[0]
        if len(non_zero_indices) < n:
            # If not enough, fallback to uniform sampling for remaining samples
            uniform_samples = np.random.uniform(mins, maxs, (n - len(non_zero_indices), D))
            xs = np.vstack([lhs_samples[non_zero_indices], uniform_samples])
            ys = np.concatenate([lhs_values[non_zero_indices], f(uniform_samples)])
        else:
            # Perform importance sampling
            indices = np.random.choice(non_zero_indices, size=n, replace=False, p=probabilities.flatten())
            xs = lhs_samples[indices]
            ys = lhs_values[indices]

        # Enforce the max_samples_per_region constraint
        final_xs = []
        final_ys = []
        seen_samples = {}

        for x, y in zip(xs, ys):
            rounded_x = tuple(np.round(x, decimals=6))
            if rounded_x not in seen_samples:
                seen_samples[rounded_x] = 0

            if seen_samples[rounded_x] < max_samples_per_region:
                final_xs.append(x)
                final_ys.append(y)
                seen_samples[rounded_x] += 1

            if len(final_xs) >= n:
                break

        final_xs = np.array(final_xs)[:n]
        final_ys = np.array(final_ys)[:n]

        return final_xs, final_ys