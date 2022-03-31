import numpy as np


def kdSplit(container):
    '''Finds axis with greatest variance and splits perpendicular to it.'''

    samples = container.X
    variances = samples.var(axis=0)
    split_dimension = np.argmax(variances)
    median = np.median(np.unique(samples[:, split_dimension]))
    lcont, rcont = container.split(split_dimension, median)

    return [lcont, rcont]
