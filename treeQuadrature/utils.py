import numpy as np

def scale(ys):
    '''
    Scale a numpy array so that all values are between 0 and 1

    Parameter
    ----------
    ys : 1-d array
        the array to be scaled

    Return
    ------
    scaled array of the same shape as ys
    '''
    ys = np.array(ys)
    high = np.max(ys)
    low = np.min(ys)

    if high == low:
        raise Exception('cannot scale an array with all entries the same')

    return (ys - low) / (high - low)

