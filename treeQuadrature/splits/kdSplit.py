import numpy as np


def kdSplit(container) -> list:
    """
    Find the axis with the greatest variance and split perpendicular to it.

    Parameters
    ----------
    container : Container
        The container object with attribute X representing the samples.

    Returns
    -------
    list
        A list containing two sub-containers resulting from the split.
    """

    samples = container.X

    # Calculate variances along each axis
    variances = samples.var(axis=0)
    split_dimension = np.argmax(variances)

    # Calculate the median value for splitting
    unique_values = np.unique(samples[:, split_dimension])
    if len(unique_values) < 2:
        raise ValueError("Not enough unique values to perform a split.")
    median = np.median(unique_values)

    lcont, rcont = container.split(split_dimension, median)

    return [lcont, rcont]
