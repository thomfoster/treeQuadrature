from typing import Dict, Any, Callable, Optional, Union, List
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import pairwise_distances
import numpy as np
import warnings
from traceback import print_exc

from ..gaussianProcess import IterativeGPFitting, GP_diagnosis, kernel_integration, GPFit, SklearnGPFit
from ..gaussianProcess.kernelIntegration import poly_post
from .containerIntegral import ContainerIntegral
from ..container import Container
from ..gaussianProcess.kernels import Polynomial

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
        Default 1e3
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
    check_GP : bool
        if true, print diagnostics of GP
        prediction variance, mean squared error and r^2 score
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
            'range': 1e3,
            'n_samples': 15,
            'n_splits' : 5,
            'max_redraw': 5,
            'threshold' : 0.8,
            'threshold_direction' : 'up',
            'check_GP': False,
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
        if not isinstance(options['fit_residuals'], bool):
            raise TypeError('fit_residuals must be a bool')

    def containerIntegral(self, container: Container, f: Callable,
                          return_std: bool=False,
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
        return_std : bool
            if True, returns the posterior std of integral estimate
        kwargs : Any
            other arguments allowed (see RbfIntegral attributes)
        
        Return
        ------
        dict
            - integral (float) the integral estimate
            - std (float) standard deviation of integral, if return_std = True
            - hyper_params (dict) hyper-parameters of the fitted kernel
            - performance (float) GP goodness of fit score
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
        self.gp = self.iGP.gp

        ### GP diagnosis
        if self.check_GP:
            # TODO - decide where to plot
            GP_diagnosis(self.iGP, container)
        
        ret = kernel_integration(self.iGP, container, gp_results, 
                                            return_std)
        
        ret['hyper_params'] = {'length' : self.gp.hyper_params['length_scale']}
        ret['performance'] = gp_results['performance']
        
        return ret
    

class AdaptiveRbfIntegral(ContainerIntegral):
    """
    Use Gaussian Process with RBF kernel, 
    with length scale and search range determined
    adaptively to the container

    Attributes
    -----------
    min_n_samples : int
        number of samples to draw in the smallest container
        for larger containers, sample size increases
    max_n_samples : int
        upper cap for samples size to control the fitting time
    n_splits : int
        number of K-fold cross-validation splits
        if n_splits = 0, K-Fold CV will not be performed. 
    max_redraw : int
        maximum number of times to increase the 
        number of samples in GP fitting. 
        Should NOT be too large. 
    thershold : float
        minimum score that must be achieved by 
        Gaussian Process. 
    gp : GPFit
        default is SklearnGPFit
    iGP : IterativeGPFit
        the iterative fitter used by this instance
        though max_redraw = 0 in this case
        as number of samples scale with volume of container instead
    fit_residuals : bool
        if True, GP is fitted to residuals
        instead of samples
    return_std : bool
        if True, returns the posterior std of integral estimate
    scaling method : str or Callable,
        The way sample size increase with volume. (ignored if volume_scaling = False)
        should be one of 'linear', 'sqrt', or 'exponential'; 
        If callable, should take a float (ratio of current volume to smallest volume)
        and return a float (scaled ratio to be multiplied to min_n_samples)
    volume_scaling : bool
        whether to choose sample size based on volume 
        or not 
    alpha : float
        Controls the aggressiveness of exponential sample scaling
    """
    def __init__(self, min_n_samples: int=15, max_n_samples: int=200,
                 n_splits: int=4, max_redraw: int=4, threshold: float=0.7,
                 fit_residuals: bool=True,
                 gp: Optional[GPFit]=None, 
                 volume_scaling: bool=False,
                 scaling_method: Union[str, Callable]='linear', 
                 alpha: float=0.2) -> None:
        if min_n_samples < 1:
            raise ValueError('min_n_samples must be at least 1')
        if min_n_samples > max_n_samples:
            raise ValueError(
                'max_n_samples must be greater than or equal to min_n_samples')
        self.min_n_samples = min_n_samples
        self.max_n_samples = max_n_samples
        self.n_splits = n_splits
        self.max_redraw = max_redraw
        self.threshold = threshold
        self.gp = gp
        self.fit_residuals = fit_residuals
        self.volume_scaling = volume_scaling
        self.scaling_method = scaling_method
        self.alpha = alpha

    def sample_size_scale(self, container: Container, min_cont_size: float):
        ratio = (container.volume / min_cont_size) ** (1 / container.D)
        if self.scaling_method == 'linear':
            n = int(self.min_n_samples * ratio)
        elif self.scaling_method == 'sqrt':
            n = int(self.min_n_samples * np.sqrt(ratio))
        elif self.scaling_method == 'exponential':
            n = int(self.min_n_samples * ratio ** 
                    (self.alpha * container.D))
        elif isinstance(self.scaling_method, Callable):
            try:
                n = int(self.min_n_samples * self.scaling_method(ratio))
            except Exception:
                print_exc()
                raise Exception(
                    'Exception ocured when scaling volume ratio, please check sample_scaling function. '
                    'It should take a float and return a float'
                    )
            
            if n < 1:
                raise ValueError('n should be at least 1, please check sample_scaling function')
        else:
            raise ValueError("sample_scaling must be one of ['linear', 'sqrt', or 'exponential'] "
                             "or a callable function")
        
        if n > self.max_n_samples:
            print('maximum sample size reached')
        return min(self.max_n_samples, n)

    def containerIntegral(self, container: Container, 
                          f: Callable[..., np.ndarray], 
                          min_cont_size: int, return_std: bool=False) -> Dict:
        """
        Arguments
        ---------
        container: Container
            the container on which the integral of f should be evaluated
        f : function
            takes X : np.ndarray and return np.ndarray, 
            see pdf method of Distribution class in exampleDistributions.py
        min_cont_size : float
            volume of the smalelst container
        return_std : bool
            if True, returns the posterior std of integral estimate
        
        Return
        ------
        dict
            - integral (float) value of the integral of f on the container
            - std (float) standard deviation of integral, if .return_std = True
            - hyper_params (dict) hyper-parameters of the fitted kernel
            - performance (float) GP goodness of fit score
        """
        
        if self.volume_scaling:
            n_samples = self.sample_size_scale(container, min_cont_size)
        else: 
            n_samples = self.min_n_samples

        xs = container.rvs(n_samples)
        ys = f(xs)

        pairwise_dists = pairwise_distances(xs)
        # Mask the diagonal
        mask = np.eye(pairwise_dists.shape[0], dtype=bool)
        pairwise_dists = np.ma.masked_array(pairwise_dists, mask)

        mean_dist = np.mean(pairwise_dists)
        D = xs.shape[1]
        initial_length = mean_dist / np.sqrt(D)
        smallest_dist = np.min(pairwise_dists)
        largest_dist = np.max(pairwise_dists)

        # avoid setting 0 bound
        min_bound = 1e-5
        lower_bound = max(smallest_dist / 10, min_bound)
        upper_bound = max(largest_dist * 10, min_bound)

        bounds = (lower_bound, upper_bound)

        self.kernel = RBF(initial_length, bounds)

        self.iGP = IterativeGPFitting(n_samples=n_samples, n_splits=self.n_splits, 
                                 max_redraw=self.max_redraw, 
                                 performance_threshold=self.threshold, 
                                 gp=self.gp, fit_residuals=self.fit_residuals)
        # only fit using the samples drawn here
        gp_results = self.iGP.fit(f, container, self.kernel, 
                                  initial_samples=(xs, ys))
        self.gp = self.iGP.gp

        ret = kernel_integration(self.iGP, container, gp_results, 
                                             return_std)
        
        ret['hyper_params'] = {'length' : self.gp.hyper_params['length_scale']}
        ret['performance'] = gp_results['performance']
        
        return ret
    

class PolyIntegral(ContainerIntegral):
    """
    Use Gaussian Process with a Polynomial kernel, 
    with degree and coefficient determined by  
    grid search. 

    Attributes
    -----------
    degrees : List[int]
        degrees of polynomials to search through
    coeffs : ArrayLike
        range of coefficients to search from 
        for polynomial kernel
    n_samples : int
        number of samples drawn in each iteration
    n_splits : int
        Number of K-fold cross-validation splits.
        If n_splits = 0, K-Fold CV will not be performed.
    max_redraw : int
        Maximum number of times to increase the 
        number of samples in GP fitting. 
        Should NOT be too large.
    threshold : float
        Minimum score that must be achieved by 
        Gaussian Process. 
    gp : GPFit
        Default is SklearnGPFit.
    iGP : IterativeGPFit
        The iterative fitter used by this instance.
        Though max_redraw = 0 in this case,
        as number of samples scale with volume of container instead.
    fit_residuals : bool
        If True, GP is fitted to residuals
        instead of samples.
    return_std : bool
        If True, returns the posterior std of integral estimate.
    """
    
    def __init__(self, degrees: List[int], coeffs=None, n_samples: int=20, 
                 n_splits: int = 4, max_redraw: int = 4, 
                 threshold: float = 0.7, fit_residuals: bool = True, 
                 gp: Optional[GPFit] = None) -> None:
        self.n_samples = n_samples
        self.n_splits = n_splits
        self.max_redraw = max_redraw
        self.threshold = threshold
        self.gp = gp
        self.fit_residuals = fit_residuals
        self.degrees = degrees
        if coeffs:
            self.coeffs = coeffs
        else:
            self.coeffs = np.logspace(-2, 1, 5)  # default coeffs between 0.01 and 10

    def containerIntegral(self, container: Container, 
                          f: Callable[..., np.ndarray], 
                          return_std: bool = False) -> Dict:
        """
        Arguments
        ---------
        container: Container
            The container on which the integral of f should be evaluated.
        f : function
            Takes X : np.ndarray and returns np.ndarray.
        return_std : bool
            If True, returns the posterior std of integral estimate.
        
        Return
        ------
        dict
            - integral (float) value of the integral of f on the container.
            - std (float) standard deviation of integral, if return_std = True.
            - hyper_params (dict) hyper-parameters of the fitted kernel.
            - performance (float) GP goodness of fit score.
        """
        xs = container.rvs(self.n_samples)
        ys = f(xs)

        # Define a function to optimize the hyper-parameters (degree and coefficient)
        def optimize_kernel_hyperparams(d, c):
            # Here, we return a "negative" score because we will minimize this function
            kernel = Polynomial(degree=d, coef0=c)
            gp = SklearnGPFit()
            iGP = IterativeGPFitting(n_samples=self.n_samples, n_splits=self.n_splits, 
                                     max_redraw=self.max_redraw, 
                                     performance_threshold=self.threshold, 
                                     gp=gp, fit_residuals=self.fit_residuals)
            gp_results = iGP.fit(f, container, kernel, initial_samples=(xs, ys))
            return -gp_results['performance']

        # Perform grid search over the hyper-parameters
        best_score = float('inf')
        best_d, best_c = None, None

        for d in self.degrees:
            for c in self.coeffs:
                score = optimize_kernel_hyperparams(d, c)
                if score < best_score:
                    best_score = score
                    best_d, best_c = d, c

        # Fit GP with the best-found hyperparameters
        self.kernel = Polynomial(degree=best_d, coef0=best_c)
        self.iGP = IterativeGPFitting(n_samples=self.n_samples, n_splits=self.n_splits, 
                                      max_redraw=self.max_redraw, 
                                      performance_threshold=self.threshold, 
                                      gp=self.gp, fit_residuals=self.fit_residuals)
        gp_results = self.iGP.fit(f, container, self.kernel, initial_samples=(xs, ys))
        self.gp = self.iGP.gp

        # Perform kernel integration with polynomial kernel
        ret = kernel_integration(self.iGP, container, gp_results, 
                                 return_std, kernel_post=poly_post, d=best_d, c=best_c)
        
        ret['hyper_params'] = {'degree': best_d, 'coef0': best_c}
        ret['performance'] = gp_results['performance']
        
        return ret