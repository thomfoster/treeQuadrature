import numpy as np
import warnings

# abstract managing
from abc import ABC, abstractmethod
from .container import Container
from typing import Any, Dict, Callable

## packages required for rbfIntegral
from sklearn.gaussian_process.kernels import RBF
from treeQuadrature.gaussianProcess import fit_GP, GP_diagnosis, rbf_Integration


# abstract class for any integrator
class ContainerIntegral(ABC):
    @abstractmethod
    def containerIntegral(self, container: Container, f: Callable[..., np.ndarray], 
                          **kwargs: Any) -> float:
        """
        Arguments
        ---------
        container: Container
            the container on which the integral of f should be evaluated
        f : function
            takes X : np.ndarray and return np.ndarray, 
            see pdf method of Distribution class in exampleDistributions.py
        kwargs : Any
            other arguments allowed, must set default values
        
        Return
        ------
        float
            value of the integral of f on the container
        """
        pass

class MidpointIntegral(ContainerIntegral):
    '''estimate integral by the function value at mid-point of container'''
    def containerIntegral(self, container, f):
        mid_x = container.midpoint
        mid_y = f(mid_x)[0, 0]
        return mid_y * container.volume


class MedianIntegral(ContainerIntegral):
    '''estimate integral by the median of samples in the container. '''
    def containerIntegral(self, container, f):
        
        if container.N == 0:
            warnings.warn(
                'Attempted to use medianIntegral on Container object with 0' +
                'samples.')
            return 0

        fs = np.array([f(x) for x in container.X])
        median = np.median(fs)

        return median * container.volume


class RandomIntegral(ContainerIntegral):
    '''
    Monte Carlo integrator:
    redraw samples, and estimate integral by the median

    Parameters
    ----------
    n : int
        number of samples to be redrawn for evaluating the integral
        default : 10
    '''
    def __init__(self, n: int = 10) -> None:
        self.n = n

    def containerIntegral(self, container, f, **kwargs):
        n = kwargs.get('n', self.n)

        samples = container.rvs(n)
        ys = f(samples)
        container.add(samples, ys)  # for tracking num function evaluations
        # I deliberately ignore previous samples which give skewed estimates
        y = np.median(ys)

        return y * container.volume


class SmcIntegral(ContainerIntegral):
    """
    Monte Carlo integrator:
    redraw samples, and estimate integral by the mean

    Attributes
    ----------
    n : int, optional
        number of samples to be redrawn for evaluating the integral
        default : 10
    """
    def __init__(self, n: int = 10) -> None:
        self.n = n

    def containerIntegral(self, container, f, **kwargs):
        n = kwargs.get('n', self.n)

        samples = container.rvs(n)
        ys = f(samples)
        container.add(samples, ys)  # for tracking num function evaluations
        v = container.volume
        return v * np.mean(ys)

class RbfIntegral(ContainerIntegral):
    """
    use Gaussian process with RBF kernel 
    to estimate the integral value on a container
    assumes uniform prior used

    Attributes
    ----------
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
        Defaults to 15
    n_tuning : int
        number of different initialisations used 
        for tuning length scale of RBF
        default : 10
    factr : int or float
        convergence criteria for fmin_l_bfgs_b optimiser
        used to fit Gaussian Process
        default : 1e7
    max_iter : int
        maximum number of iterations for fmin_l_bfgs_b optimiser
        default : 1e4
    check_GP : bool
        if true, print diagnostics of GP
        prediction variance, mean squared error and r^2 score
    return_std : bool
        if True, returns the 
        Defaults to False
    """
    def __init__(self, **kwargs) -> None:
        
        self.options : Dict[str, Any] = {
            'length': 10,
            'n_samples': 15,
            'n_tuning': 10,
            'factr': 1e7,
            'max_iter': 1e4,
            'check_GP': False,
            'return_std': False
        }
        self.options.update(kwargs)

        RbfIntegral._validate_options(self.options)

        # set options correspondingly
        for key, value in self.options.items():
            setattr(self, key, value)

    @staticmethod
    def _validate_options(options):
        if not isinstance(options['length'], int):
            raise TypeError('length must be an int')
        if not isinstance(options['n_samples'], int):
            raise TypeError('n_samples must be an int')
        if not isinstance(options['n_tuning'], int):
            raise TypeError('n_tuning must be an int')
        if not isinstance(options['factr'], (int, float)):
            raise TypeError('factr must be an int or float')
        if not isinstance(options['max_iter'], (int, float)):
            raise TypeError('max_iter must be an int or float')
        if not isinstance(options['check_GP'], bool):
            raise TypeError('check_GP must be a bool')
        if not isinstance(options['return_std'], bool):
            raise TypeError('return_std must be a bool')

    def containerIntegral(self, container, f, **kwargs: Any):
        options = self.options.copy()
        options.update(kwargs)

        RbfIntegral._validate_options(options)

        # Set instance variables based on self.options
        for key, value in self.options.items():
            setattr(self, key, value)

        # redraw uniform samples from the container
        xs = container.rvs(self.n_samples)
        ys = f(xs)
        container.add(xs, ys)  # for tracking num function evaluations

        # fit GP using RBF kernel
        kernel = RBF(self.length, (self.length*1e-2, self.length*1e2))
        gp = fit_GP(xs, ys, kernel, self.n_tuning, self.max_iter, self.factr)

        ### GP diagnosis
        if self.check_GP:
            # TODO - decide where to plot
            GP_diagnosis(gp, xs, ys, container, 
                    criterion = lambda container : container.volume > 0.1)
        
        return rbf_Integration(gp, container, xs, self.return_std)