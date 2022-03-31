import numpy as np
import warnings


def midpointIntegral(container, f):
    '''Boxlike integral. Take height as value at midpoint.'''
    mid_x = container.midpoint
    mid_y = f(mid_x)[0, 0]
    return mid_y * container.volume


def medianIntegral(container, f):
    '''Boxlike integral. Take height as median sample value.'''
    if container.N == 0:
        warnings.warn(
            'Attempted to use medianIntegral on Container object with 0' +
            'samples.')
        return 0

    fs = np.array([f(x) for x in container.X])
    median = np.median(fs)

    return median * container.volume


def randomIntegral(container, f, n=10):
    '''Boxlike integral. Take height as mean value over n uniform samples.'''
    samples = container.rvs(n)
    ys = f(samples)
    container.add(samples, ys)  # for tracking num function evaluations
    # I deliberately ignore previous samples which give skewed estimates
    y = np.median(ys)
    return y * container.volume


def smcIntegral(container, f, n=10):
    samples = container.rvs(n)
    ys = f(samples)
    container.add(samples, ys)  # for tracking num function evaluations
    v = container.volume
    return v * np.mean(ys)
