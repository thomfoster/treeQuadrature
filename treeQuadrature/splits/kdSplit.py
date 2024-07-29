import numpy as np
import warnings

from .split import Split


class KdSplit(Split):
    """
    Find the axis with the greatest variance and split perpendicular to it.
    """
    def split(self, container):
        samples = container.X

        # Calculate variances along each axis
        variances = samples.var(axis=0)
        split_dimension = np.argmax(variances)

        # Calculate the median value for splitting
        unique_values = np.unique(samples[:, split_dimension])
        if len(unique_values) < 2:
            warnings.warn('no enough unique values to split', 
                          RuntimeWarning)
            return [container]
        median = np.median(unique_values)

        lcont, rcont = container.split(split_dimension, median)

        return [lcont, rcont]
