import numpy as np


def scale(ys):
    '''Scale a numpy array between 0 and 1'''
    ys = np.array(ys)
    high = np.max(ys)
    low = np.min(ys)
    return (ys - low) / (high - low)