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

def handle_bound(value, D, default_value) -> np.ndarray:
    if value is None:
        return np.array([default_value] * D)
    elif isinstance(value, (int, float)):
        return np.array([value] * D)
    elif isinstance(value, (list, np.ndarray)) and len(value) == D:
        return np.array(value)
    else:
        raise ValueError(
            "value must be a float, list, or numpy.ndarray"
            f"with length {D} when given as a list or numpy.ndarray"
        )