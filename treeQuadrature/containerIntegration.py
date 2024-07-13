import numpy as np
import warnings

## packages required for rbfIntegral
from sklearn.gaussian_process.kernels import RBF
from gaussianProcess import fit_GP, GP_diagnosis, rbf_Integration


def midpointIntegral(container, f):
    '''
    estimate integral by the 
    function value at mid-point of container
    
    Parameters
    ----------
    container : Container 
    f : function 
        represents the integrand function
        takes a numpy array and return a float
    
    Return
    ------
    float
        the estimated value of integral on the container
    '''
    mid_x = container.midpoint
    mid_y = f(mid_x)[0, 0]
    return mid_y * container.volume


def medianIntegral(container, f):
    '''
    estimate integral by the median of samples in the container. 
    
    Parameters
    ----------
    container : Container 
    f : function 
        represents the integrand function
        takes a numpy array and return a float
    
    Return
    ------
    float
        the estimated value of integral on the container
    '''
    if container.N == 0:
        warnings.warn(
            'Attempted to use medianIntegral on Container object with 0' +
            'samples.')
        return 0

    fs = np.array([f(x) for x in container.X])
    median = np.median(fs)

    return median * container.volume


def randomIntegral(container, f, n=10):
    '''
    Monte Carlo integrator:
    redraw samples, and estimate integral by the median

    Parameters
    ----------
    container : Container 
    f : function 
        represents the integrand function
        takes a numpy array and return a float
    n : int
        number of samples to be redrawn for evaluating the integral
        default : 10
    
    Return
    ------
    float
        the estimated value of integral on the container
    '''
    samples = container.rvs(n)
    ys = f(samples)
    container.add(samples, ys)  # for tracking num function evaluations
    # I deliberately ignore previous samples which give skewed estimates
    y = np.median(ys)
    return y * container.volume


def smcIntegral(container, f, n=10):
    """
    Monte Carlo integrator:
    redraw samples, and estimate integral by the mean

    Parameters
    ----------
    container : Container 
    f : function 
        represents the integrand function
        takes a numpy array and return a float
    n : int
        number of samples to be redrawn for evaluating the integral
        default : 10
    
    Return
    ------
    float
        the estimated value of integral on the container
    """
    samples = container.rvs(n)
    ys = f(samples)
    container.add(samples, ys)  # for tracking num function evaluations
    v = container.volume
    return v * np.mean(ys)

def rbfIntegral(container, f, length=1.0, n_samples=40,
                n_tuning=10, factr=1e7, max_iter=1.5e4, check_GP=False, 
                return_std=False):
    """
    use Gaussian process with RBF kernel 
    to estimate the integral value on a container
    assumes uniform prior used

    Parameters
    ----------
    container : Container 
    f : function
        the integrand
    length : float
        the initial value of the length scale of RBF kernel
        default 1.0
    const : float
        the initial value of the constant kerenl
        default 1.0
    n_samples : int
        number of random samples
        uniformly redrawn from the container
        to fit Gaussian Process.
        Defaults to 40
    n_tuning : int
        number of different initialisations used 
        for tuning length scale of RBF
        default : 10
    factr : float
        convergence criteria for fmin_l_bfgs_b optimiser
        used to fit Gaussian Process
        default 1e7
    max_iter : int
        maximum number of iterations for fmin_l_bfgs_b optimiser
    check_GP : bool
        if true, print diagnostics of GP
        prediction variance, mean squared error and r^2 score
    return_std : bool
        if True, returns the 
        Defaults to False

    Return 
    -------
    integral : float
        the estimated value of integral on the container
    """

    # redraw uniform samples from the container
    xs = container.rvs(n_samples)
    ys = f(xs)
    container.add(xs, ys)  # for tracking num function evaluations

    # fit GP using RBF kernel
    kernel = RBF(length, (length*1e-2, length*1e2))
    gp = fit_GP(xs, ys, kernel, n_tuning, max_iter, factr)

    ### GP diagnosis
    if check_GP:
        # TODO - decide where to plot
        GP_diagnosis(gp, xs, ys, container, 
                 criterion = lambda container : container.volume > 0.1)
    
    return rbf_Integration(gp, container, xs, return_std)