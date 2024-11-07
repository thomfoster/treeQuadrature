from typing import Tuple, List, Optional
import numpy as np
from .base_class import Sampler


class MixedSampler(Sampler):
    """
    A sampler that combines multiple samplers, drawing samples
    from each according to specified proportions.
    """

    def __init__(
        self, samplers: List[Sampler],
        proportions: Optional[List[float]] = None
    ):
        """
        Arguments
        ---------
        samplers : List[Sampler]
            A list of sampler instances to be combined.
        proportions : List[float], Optional
            Proportions of samples to draw from each sampler.
            If None, samples will be drawn equally from each sampler.
        """
        self.samplers = samplers

        if proportions is None:
            self.proportions = [1.0 / len(samplers)] * len(samplers)
        else:
            if len(proportions) != len(samplers):
                raise ValueError(
                    "Length of proportions must "
                    "match the number of samplers."
                )
            self.proportions = proportions

        # Normalize proportions to sum to 1
        self.proportions = np.array(
            self.proportions) / np.sum(self.proportions)

    def rvs(
        self, n: int, mins: np.ndarray, maxs: np.ndarray, f: callable, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples by combining samples from multiple samplers
        according to the specified proportions.

        Parameters
        ----------
        n : int
            Total number of samples to generate.
        mins, maxs : np.ndarray
            1-dimensional arrays of the lower and upper bounds.
        f : callable
            The integrand function from which to sample.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The generated samples and their corresponding function values.
        """
        mins, maxs, _ = Sampler.handle_mins_maxs(mins, maxs)

        # Determine the number of samples to draw from each sampler
        samples_distribution = np.floor(n * self.proportions).astype(int)
        remaining_samples = n - np.sum(samples_distribution)

        # Distribute the remaining samples to the samplers
        for i in range(remaining_samples):
            samples_distribution[i % len(samples_distribution)] += 1

        all_samples = []
        all_values = []

        # Draw samples from each sampler
        for sampler, num_samples in zip(
            self.samplers, samples_distribution
        ):
            if num_samples > 0:
                samples, values = sampler.rvs(
                    num_samples, mins, maxs, f, **kwargs)
                all_samples.append(samples)
                all_values.append(values)

        # Concatenate all the samples and values
        final_samples = np.vstack(all_samples)
        final_values = np.vstack(all_values)

        return final_samples, final_values
