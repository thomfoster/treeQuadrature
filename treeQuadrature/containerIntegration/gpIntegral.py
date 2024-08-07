from typing import Dict, Any, Callable, Optional
from sklearn.gaussian_process.kernels import RBF
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
    """
    def __init__(self, gp: Optional[GPFit]=None, **kwargs) -> None:
        """
        Arguments
        ---------
        gp : GPFit, optional
            the gaussian process model fitter
            default using sklearn
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