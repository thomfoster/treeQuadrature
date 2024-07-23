from typing import Dict, Any, Callable
from sklearn.gaussian_process.kernels import RBF
import warnings

from ..gaussianProcess import IterativeGPFitting, GP_diagnosis, rbf_Integration
from .containerIntegral import ContainerIntegral
from ..container import Container

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
    range : float
        GPRegressor will search hyper-parameter
        among (length * 1/range, length * range)
        Default 1e2
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
    thershold : float
        minimum R2 score that must be achieved by 
        Gaussian Process. 
        Default = 0.8
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
    """
    def __init__(self, **kwargs) -> None:
        
        self.options : Dict[str, Any] = {
            'length': 10,
            'range': 1e2,
            'n_samples': 15,
            'n_tuning': 10,
            'factr': 1e7,
            'max_iter': 1e4,
            'max_redraw': 5,
            'threshold' : 0.8,
            'check_GP': False,
            'return_std': False
        }
        self.options.update(kwargs)

        RbfIntegral._validate_options(self.options)

        # set options correspondingly
        for key, value in self.options.items():
            setattr(self, key, value)
        
        self.name = 'RbfIntegral'

    @staticmethod
    def _validate_options(options):
        if not isinstance(options['length'], (int, float)):
            raise TypeError(
                'length must be an int or float'
                f', got {options['length']}'
                )
        if not isinstance(options['n_samples'], int):
            raise TypeError(
                'n_samples must be an int,'
                f' got {options['n_samples']}'
                )
        if not isinstance(options['n_tuning'], int):
            raise TypeError(
                'n_tuning must be an int,'
                f' got {options['n_tuning']}'
                )
        if not isinstance(options['factr'], (int, float)):
            raise TypeError(
                'factr must be an int or float'
                f', got {options['factr']}'
                )
        if not isinstance(options['max_iter'], (int, float)):
            raise TypeError(
                'max_iter must be an int or float'
                f', got {options['max_iter']}'
                )
        if not isinstance(options['max_redraw'], int):
            raise TypeError(
                'max_redraw must be an int'
                f', got {options['max_redraw']}'
                )
        threshold = options['threshold']
        if not isinstance(threshold, float):
            raise TypeError(
                'threshold must be a float'
                f', got {threshold}'
                )
        if threshold > 1 or threshold < 0:
            raise ValueError(
                'threshold must be in [0, 1]'
                f', got {threshold}'
                )
        if not isinstance(options['check_GP'], bool):
            raise TypeError('check_GP must be a bool')
        if not isinstance(options['return_std'], bool):
            raise TypeError('return_std must be a bool')

    def containerIntegral(self, container: Container, f: Callable, 
                          return_hyper_params: bool = False, 
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
        return_hyper_params : bool
            if True, 
        kwargs : Any
            other arguments allowed (see RbfIntegral attributes)
        
        Return
        ------
        float or tuple
            value of the integral of f on the container, 
            and std if return_std = True, 
            and hyper_parameters of GP fitting if 
        """
        options = self.options.copy()
        options.update(kwargs)

        if self.max_redraw * self.n_samples > 500:
            warnings.warn(
                'the computational cost could be extremely high'
                'due to high values of max_iter and n_samples'
                )

        RbfIntegral._validate_options(options)

        # Set instance variables based on self.options
        for key, value in options.items():
            setattr(self, key, value)

        # fit GP using RBF kernel
        kernel = RBF(self.length, (self.length*(1/self.range), 
                                   self.length*self.range))
        gp = IterativeGPFitting(f, container, kernel,
                 self.n_samples, self.length, self.range, 
                 self.n_tuning, self.max_iter, self.factr, 
                 self.threshold, self.max_redraw).fit()
        
        # Track number of function evaluations
        container.add(gp.X_train_, gp.y_train_)  

        ### GP diagnosis
        if self.check_GP:
            # TODO - decide where to plot
            GP_diagnosis(gp, container)
        
        integral_result = rbf_Integration(gp, container, self.return_std)
        
        if return_hyper_params:
            hyper_params = {'length' : gp.kernel_.length_scale}
            return integral_result, hyper_params
        else:
            return integral_result