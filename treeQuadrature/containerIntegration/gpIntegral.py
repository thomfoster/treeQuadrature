from typing import Dict, Any, Callable, Optional
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import pairwise_distances
import numpy as np
import warnings

from ..gaussianProcess import IterativeGPFitting, GP_diagnosis, kernel_integration, GPFit
from .containerIntegral import ContainerIntegral
from ..container import Container

class RbfIntegral(ContainerIntegral):
    """
    use Gaussian process with RBF kernel 
    to estimate the integral value on a container
    assumes uniform prior used

    Attributes
    ----------
    length : int or float
        the initial value of the length scale of RBF kernel
        default 1.0
    range : int or float
        GPRegressor will search hyper-parameter
        among (length * 1/range, length * range)
        Default 1e2
    thershold : float
        minimum score that must be achieved by 
        Gaussian Process. 
    n_samples : int
        number of samples to be drawn in each round of IterativeGPFitting
    n_splits : int
        number of K-fold cross-validation splits
        if n_splits = 0, K-Fold CV will not be performed. 
    threshold_direction : str
        one of 'up' and 'down'. 
        if 'up', accept the model if score >= performance_threshold; 
        if 'down', accept the model if score >= performance_threshold
    max_redraw : int
        maximum number of times to increase the 
        number of samples in GP fitting. 
        Should NOT be too large. 
        Default = 5
    check_GP : bool
        if true, print diagnostics of GP
        prediction variance, mean squared error and r^2 score
    return_std : bool
        if True, returns the 
        Defaults to False
    fit_residuals : bool
        if True, GP is fitted to residuals
        instead of samples
    gp : GPFit
        default is SklearnGPFit
    iGP : IterativeGPFit
        the iterative fitter used by this instance
    """
    def __init__(self, gp: Optional[GPFit]=None, **kwargs) -> None:
        """
        Arguments
        ---------
        gp : GPFit, optional
            the gaussian process model fitter
            default using sklearn
        **kwargs : Any
            length, range, n_samples, 
            n_splits, max_redraw, threshold,
            threshold_direction, check_GP, 
            return_std, and fit_residuals
            can all be set here. See details in 
            class description 
        """

        self.gp = gp
        
        self.options : Dict[str, Any] = {
            'length': 10,
            'range': 1e2,
            'n_samples': 15,
            'n_splits' : 5,
            'max_redraw': 5,
            'threshold' : 0.8,
            'threshold_direction' : 'up',
            'check_GP': False,
            'return_std': False, 
            'fit_residuals' : True
        }
        self.options.update(kwargs)

        RbfIntegral._validate_options(self.options)

        # set options correspondingly
        for key, value in self.options.items():
            setattr(self, key, value)

        self.kernel = RBF(self.length, (self.length*(1/self.range), 
                                        self.length*self.range))
        self.iGP = IterativeGPFitting(n_samples=self.n_samples, n_splits=self.n_splits, 
                                 max_redraw=self.max_redraw, 
                                 performance_threshold=self.threshold, 
                                 threshold_direction=self.threshold_direction,
                                 gp=self.gp)

    def __str__(self):
        return 'RbfIntegral'

    @staticmethod
    def _validate_options(options):
        length = options['length']
        if not isinstance(length, (int, float)):
            raise TypeError(
                f'length must be an int or float, got {length}'
                )
        range = options['range']
        if not isinstance(range, (int, float)):
            raise TypeError(
                f'range must be an int or float, got {length}'
                )
        n_samples = options['n_samples']
        if not isinstance(n_samples, int):
            raise TypeError(
                f'n_samples must be an int, got {n_samples}'
                )
        max_redraw = options['max_redraw']
        if not isinstance(max_redraw, int):
            raise TypeError(
                f'max_redraw must be an int, got {max_redraw}'
                )
        threshold = options['threshold']
        if not isinstance(threshold, float):
            raise TypeError(
                f'threshold must be a float, got {threshold}'
                )    
        if not isinstance(options['check_GP'], bool):
            raise TypeError('check_GP must be a bool')
        if not isinstance(options['return_std'], bool):
            raise TypeError('return_std must be a bool')
        if not isinstance(options['fit_residuals'], bool):
            raise TypeError('fit_residuals must be a bool')

    def containerIntegral(self, container: Container, f: Callable,
                          **kwargs: Any):
        """
        Gaussian Process is fitted iteratively 

        Arguments
        ---------
        container: Container
            the container on which the integral of f should be evaluated
        f : function
            takes X : np.ndarray and return np.ndarray, 
            see pdf method of Distribution class in exampleDistributions.py
        kwargs : Any
            other arguments allowed (see RbfIntegral attributes)
        
        Return
        ------
        dict
            - integral (float) : the integral estimate
            - std (float) : standard deviation of integral, 
              if self.return_std = True
            - hyper_params (dict): hyper-parameters of the fitted kernel
            - performance (float): GP goodness of fit score
        """

        ### reset options
        options = self.options.copy()
        options.update(kwargs)

        if self.max_redraw * self.n_samples > 400:
            warnings.warn(
                'the computational cost could be extremely high'
                'due to high values of max_iter and n_samples'
                )

        RbfIntegral._validate_options(options)

        for key, value in options.items():
            setattr(self, key, value)

        ### fit GP using RBF kernel
        self.kernel = RBF(self.length, (self.length*(1/self.range), 
                                   self.length*self.range))
        # set up iterative fitting scheme
        self.iGP = IterativeGPFitting(n_samples=self.n_samples, n_splits=self.n_splits, 
                                 max_redraw=self.max_redraw, 
                                 performance_threshold=self.threshold, 
                                 threshold_direction=self.threshold_direction,
                                 gp=self.gp, fit_residuals=self.fit_residuals)
        gp_results = self.iGP.fit(f, container, self.kernel)
        gp = self.iGP.gp

        ### GP diagnosis
        if self.check_GP:
            # TODO - decide where to plot
            GP_diagnosis(self.iGP, container)
        
        ret = kernel_integration(self.iGP, container, gp_results, 
                                             self.return_std)
        
        ret['hyper_params'] = {'length' : gp.hyper_params['length_scale']}
        ret['performance'] = gp_results['performance']
        
        return ret
    

class AdaptiveRbfIntegral(ContainerIntegral):
    """
    Use Gaussian Process with RBF kernel, 
    with length scale and search range determined
    adaptively to the container

    Attributes
    -----------
    max_n_samples : int
        number of samples to draw in the largest container
        for smaller containers, it decreases linearly 
    gp : GPFit
        default is SklearnGPFit
    iGP : IterativeGPFit
        the iterative fitter used by this instance
        though max_redraw = 0 in this case
        as number of samples scale with volume of container instead
    fit_residuals : bool
        if True, GP is fitted to residuals
        instead of samples
    """
    def __init__(self, max_n_samples: int=200,
                 fit_residuals: bool=True,
                 return_std: bool=False, 
                 gp: Optional[GPFit]=None) -> None:
        self.max_n_samples = max_n_samples
        self.gp = gp
        self.fit_residuals = fit_residuals
        self.return_std = return_std

    def containerIntegral(self, container: Container, 
                          f: Callable[..., np.ndarray], 
                          max_cont_size: float) -> Dict:
        """
        Arguments
        ---------
        container: Container
            the container on which the integral of f should be evaluated
        f : function
            takes X : np.ndarray and return np.ndarray, 
            see pdf method of Distribution class in exampleDistributions.py
        max_cont_size : float
            size of the largest container
        
        Return
        ------
        dict
            - integral (float) value of the integral of f on the container
            - std (float) : standard deviation of integral, 
              if self.return_std = True
            - hyper_params (dict): hyper-parameters of the fitted kernel
            - performance (float): GP goodness of fit score
        """
        n_samples = self.max_n_samples * (container.volume / max_cont_size)

        xs = container.rvs(n_samples)
        ys = f(xs)

        pairwise_dists = pairwise_distances(xs)
        mean_dist = np.mean(pairwise_dists)

        D = xs.shape[1]
        initial_length = mean_dist / np.sqrt(D)
        smallest_dist = np.min(pairwise_dists + 
                               np.eye(len(xs)) * np.inf)  # avoid zero diagonal
        largest_dist = np.max(pairwise_dists)

        bounds = (smallest_dist / 10, 10 * largest_dist)

        self.kernel = RBF(initial_length, bounds)

        self.iGP = IterativeGPFitting(n_samples=n_samples, n_splits=5, 
                                 max_redraw=0, 
                                 performance_threshold=0, 
                                 gp=self.gp, fit_residuals=self.fit_residuals)
        # only fit using the samples drawn here
        gp_results = self.iGP.fit(f, container, self.kernel, 
                                  initial_samples=(xs, ys))
        gp = self.iGP.gp

        ret = kernel_integration(self.iGP, container, gp_results, 
                                             self.return_std)
        
        ret['hyper_params'] = {'length' : gp.hyper_params['length_scale']}
        ret['performance'] = gp_results['performance']
        
        return ret