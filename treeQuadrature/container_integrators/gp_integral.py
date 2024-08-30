from typing import Dict, Any, Callable, Optional, List, Type, Tuple
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import pairwise_distances
import numpy as np
import warnings
from abc import abstractmethod

from ..gaussian_process import IterativeGPFitting, gp_diagnosis, kernel_integration, GPFit, SklearnGPFit
from ..gaussian_process.kernel_integration import poly_post
from .container_integral import ContainerIntegral
from ..container import Container
from ..gaussian_process.kernels import Polynomial
from ..gaussian_process.scorings import r2
from ..samplers import Sampler, UniformSampler

class KernelIntegral(ContainerIntegral):
    """
    use Gaussian process with RBF kernel by default
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
    def __init__(self, GPFit: Type[GPFit]=SklearnGPFit, 
                 gp_params: dict = {}, kernel: Any = None,
                 **kwargs) -> None:
        """
        Arguments
        ---------
        gp : GPFit, optional
            the gaussian process model fitter
            default using SklearnGPFit
        gp_params : dict
            parameters required to initiliase GPFit
        kernel: Any, optional
            the kernel required
            By default, use the default kernel of
            GPFit.create_kernel. 
            For SklearnGPFit, the default is RBF kernel
        **kwargs : Any
            length, range, n_samples, 
            n_splits, max_redraw, threshold,
            threshold_direction, check_GP, 
            return_std, and fit_residuals
            can all be set here. See details in 
            class description 
        """

        self.GPFit = GPFit
        self.gp_params = gp_params
        self.kernel = kernel
        self.gp = None
        
        self.options : Dict[str, Any] = {
            'n_samples': 15,
            'n_splits' : 5,
            'max_redraw': 5,
            'threshold' : 0.7,
            'threshold_direction' : 'up',
            'check_GP': False,
            'fit_residuals' : True
        }
        self.options.update(kwargs)

        KernelIntegral._validate_options(self.options)

        # set options correspondingly
        for key, value in self.options.items():
            setattr(self, key, value)

    # Lazy initilliasation for parallel processing
    def _initialize_gp(self):
        if self.gp is None:
            self.gp = self.GPFit(**self.gp_params)
            if self.kernel is None:
                self.kernel = self.gp.create_kernel(self.gp_params)

    def __str__(self):
        return 'KernelIntegral'

    @staticmethod
    def _validate_options(options):
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
        if not isinstance(threshold, (int, float)):
            raise TypeError(
                f'threshold must be an integer or float, got {threshold}'
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
            see pdf method of Distribution class in distributions.py
        return_std : bool
            if True, returns the posterior std of integral estimate
        kwargs : Any
            other arguments allowed (see class attributes)
        
        Return
        ------
        dict
            - integral (float) the integral estimate
            - std (float) standard deviation of integral, if return_std = True
            - hyper_params (dict) hyper-parameters of the fitted kernel
            - performance (float) GP goodness of fit score
        """
        self._initialize_gp()

        if self.n_samples < 2:
            raise RuntimeError("Cannot perform GP Integral with less than 2 samples"
                               "please increase 'n_samples'")

        ### reset options
        options = self.options.copy()
        options.update(kwargs)

        if self.max_redraw * self.n_samples > 400:
            warnings.warn(
                'the computational cost could be extremely high'
                'due to high values of max_iter and n_samples'
                )

        KernelIntegral._validate_options(options)

        for key, value in options.items():
            setattr(self, key, value)
        
        gp = self.GPFit(**self.gp_params)
        
        # set up iterative fitting scheme
        iGP = IterativeGPFitting(n_samples=self.n_samples, n_splits=self.n_splits, 
                                 max_redraw=self.max_redraw, 
                                 performance_threshold=self.threshold, 
                                 threshold_direction=self.threshold_direction,
                                 gp=gp, fit_residuals=self.fit_residuals)
        gp_results = iGP.fit(f, container, self.kernel)

        ### GP diagnosis
        if self.check_GP:
            gp_diagnosis(iGP, container)
        
        ret = kernel_integration(iGP, container, gp_results, 
                                            return_std)
        
        ret['hyper_params'] = iGP.gp.hyper_params
        
        ret['performance'] = gp_results['performance']
        
        return ret
    

class AdaptiveRbfIntegral(ContainerIntegral):
    """
    Use Gaussian Process with RBF kernel, 
    with length scale and search range determined
    adaptively to the container

    Attributes
    -----------
    n_samples : int
        number of samples to draw
    n_splits : int
        number of K-fold cross-validation splits
        if n_splits = 0, K-Fold CV will not be performed. 
    max_redraw : int
        maximum number of times to increase the 
        number of samples in GP fitting. 
        Should NOT be too large. 
    scoring : Callable
        A scoring function to evaluate the GP predictions. 
        It must accept three arguments: 
        the true values, the predicted values, posterior std. 
        If not provided, default is predictive log likelihood
    thershold : float
        minimum score that must be achieved by 
        Gaussian Process. 
    threshold_direction : str
        one of 'up' and 'down'. for iterative GP fitting
        if 'up', accept the model if score >= performance_threshold; 
        if 'down', accept the model if score >= performance_threshold
    GPFit : Type[GPFit]
        any subclass of GPFit
        for fitting gaussian process
    gp_params : dict
        The parameters for initialising
        GPFit object
    fit_residuals : bool
        if True, GP is fitted to residuals
        instead of samples
    sampler : Sampler
        used for drawing samples in the container
    keep_samples : bool
        whether keep the original samples used 
        to build tree or not when fitting GP
    check_GP: bool
        if True, generate the GP diagnostic plots
    """
    def __init__(self, n_samples: int=15, 
                 n_splits: int=4, max_redraw: int=4, threshold: float=0.7, 
                 threshold_direction: str='up',
                 fit_residuals: bool=True, scoring: Optional[Callable]=None,
                 GPFit: Type[GPFit]=SklearnGPFit, gp_params: dict={},
                 sampler: Sampler=UniformSampler(), 
                 keep_samples: bool=False, 
                 check_GP: bool=False) -> None:
        if n_samples < 1:
            raise ValueError('min_n_samples must be at least 1')
        self.n_samples = n_samples
        self.n_splits = n_splits
        self.max_redraw = max_redraw
        self.scoring = scoring
        self.threshold = threshold
        self.threshold_direction = threshold_direction
        self.GPFit = GPFit
        self.gp_params = gp_params
        self.fit_residuals = fit_residuals
        self.sampler = sampler
        self.keep_samples = keep_samples
        self.check_GP = check_GP
        self.gp=None

    def _initialize_gp(self, xs: np.ndarray):
        """Lazy initialisation"""
        if self.gp is None:
            self.gp = self.GPFit(**self.gp_params)
            
        # construct kernel
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

        self.kernel = self.gp.create_kernel({'kernel_type': 'RBF', 
                                        'length': initial_length, 'bounds': bounds})

    def containerIntegral(self, container: Container, 
                          f: Callable[..., np.ndarray], 
                          return_std: bool=False) -> Dict:
        """
        Arguments
        ---------
        container: Container
            the container on which the integral of f should be evaluated
        f : function
            takes X : np.ndarray and return np.ndarray
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
        if self.n_samples < 2:
            raise RuntimeError("Cannot perform GP Integral with less than 2 samples"
                               "please increase 'n_samples'")

        # generate samples
        xs, ys = self.sampler.rvs(self.n_samples, container.mins, 
                              container.maxs, f)
        
        # check samples are correct
        if xs.shape[0] != ys.shape[0]:
            raise ValueError("the shape of xs and ys generated by sampler does not match")
        if xs.shape[0] != self.n_samples:
            raise RuntimeError("Too many samples! "
                               f"sampler.rvs generated {xs.shape[0]} samples"
                               f"while expecting {self.n_samples}")

        # initialize GP model and kernel
        self._initialize_gp(xs)
        
        iGP = IterativeGPFitting(n_samples=self.n_samples, n_splits=self.n_splits, 
                                 max_redraw=self.max_redraw, scoring=self.scoring,
                                 performance_threshold=self.threshold, 
                                 threshold_direction=self.threshold_direction,
                                 gp=self.gp, fit_residuals=self.fit_residuals)
        
        # only fit using the samples drawn here
        if self.keep_samples:
            container.add(xs, ys)
            gp_results = iGP.fit(f, container, self.kernel, 
                                  initial_samples=(np.vstack([xs, container.X]), 
                                                   np.vstack([ys, container.y])), 
                                                   add_samples=False)
            container.add(gp_results['new_samples'][0], gp_results['new_samples'][1])
        else:
            container.add(xs, ys)
            gp_results = iGP.fit(f, container, self.kernel, 
                                    initial_samples=(xs, ys))
            
        ### GP diagnosis
        if self.check_GP:
            gp_diagnosis(iGP, container, plot=True)

        ret = kernel_integration(iGP, container, gp_results, 
                                             return_std)
        
        ret['hyper_params'] = {'length' : iGP.gp.hyper_params['length_scale']}
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
    GPFit : Type[GPFit]
        any subclass of GPFit
        for fitting gaussian process
    gp_params : dict
        The parameters for initialising
        GPFit object
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
                 threshold: float = 10, fit_residuals: bool = True, 
                 GPFit: Type[GPFit]=SklearnGPFit,
                 gp_params: dict = {}, 
                 sampler: Sampler=UniformSampler()) -> None:
        self.n_samples = n_samples
        self.n_splits = n_splits
        self.max_redraw = max_redraw
        self.threshold = threshold
        self.GPFit = GPFit
        self.gp_params = gp_params
        self.fit_residuals = fit_residuals
        self.degrees = degrees
        if coeffs:
            self.coeffs = coeffs
        else:
            self.coeffs = np.logspace(-2, 1, 5)  # default coeffs between 0.01 and 10
        self.sampler = sampler
        self.gp = None
    
    def _initialize_gp(self):
        """Lazy initialisation"""
        if self.gp is None:
            self.gp = self.GPFit(**self.gp_params)

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
        self._initialize_gp()

        if self.n_samples < 2:
            raise RuntimeError("Cannot perform GP Integral with less than 2 samples"
                               "please increase 'n_samples'")
        
        xs, ys = self.sampler.rvs(self.n_samples, container.mins,  
                                  container.maxs, f)
        
        # check samples are correct
        if xs.shape[0] != ys.shape[0]:
            raise ValueError("the shape of xs and ys generated by sampler does not match")
        if xs.shape[0] != self.n_samples:
            raise RuntimeError("Too many samples! "
                               f"sampler.rvs generated {xs.shape[0]} samples"
                               f"while expecting {self.n_samples}")

        # Define a function to optimize the hyper-parameters (degree and coefficient)
        def optimize_kernel_hyperparams(d, c):
            # Here, we return a "negative" score because we will minimize this function
            kernel = Polynomial(degree=d, coef0=c)
            iGP = IterativeGPFitting(n_samples=self.n_samples, n_splits=self.n_splits, 
                                     max_redraw=self.max_redraw, 
                                     performance_threshold=self.threshold, 
                                     gp=self.gp, fit_residuals=self.fit_residuals)
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
        iGP = IterativeGPFitting(n_samples=self.n_samples, n_splits=self.n_splits, 
                                      max_redraw=self.max_redraw, 
                                      performance_threshold=self.threshold, 
                                      gp=self.gp, fit_residuals=self.fit_residuals)

        container.add(xs, ys)
        gp_results = iGP.fit(f, container, self.kernel, initial_samples=(xs, ys))

        # Perform kernel integration with polynomial kernel
        ret = kernel_integration(iGP, container, gp_results, 
                                 return_std, kernel_post=poly_post, d=best_d, c=best_c)
        
        ret['hyper_params'] = {'degree': best_d, 'coef0': best_c}
        ret['performance'] = gp_results['performance']
        
        return ret
    

class IterativeGpIntegral(ContainerIntegral):
    def __init__(self, n_samples: int,
                 n_splits: int, fit_residuals: bool, 
                 scoring: Callable, score_direction: str,
                 GPFit: Type[GPFit], gp_params: dict, 
                 sampler: Sampler) -> None:
        if n_samples < 1:
            raise ValueError('n_samples must be at least 1')
        self.n_samples = n_samples
        self.n_splits = n_splits
        self.scoring = scoring
        self.score_direction = score_direction
        self.GPFit = GPFit
        self.gp_params = gp_params
        self.fit_residuals = fit_residuals
        self.sampler = sampler

    @abstractmethod
    def containerIntegral(self, container: Container, f: Callable, return_std: bool,
                          previous_samples: Tuple[np.ndarray, 
                                                  np.ndarray]):
        """
        Takes previous samples and draw new samples to perform fitting again.

        Arguments 
        --------
        container: Container
            the container on which the integral of f should be evaluated
        f : function
            takes X : np.ndarray and return np.ndarray, 
        return_std : bool
            if True, returns the posterior std of integral estimate
        previous_samples : tuple, optional
            if provided, it contains (xs, ys) from the previous iteration

        Return
        ------
        dict
            - integral (float) value of the integral of f on the container
            - std (float) standard deviation of integral, if .return_std = True
            - hyper_params (dict) hyper-parameters of the fitted kernel
            - performance (float) GP goodness of fit score
        tuple(np.ndarray, np.ndarray)
            samples xs, ys used by this iteration
        """
        pass

class IterativeRbfIntegral(IterativeGpIntegral):
    """
    A modified version of AdaptiveRbfIntegral that allows iterative
    sample drawing and fitting of the Gaussian Process model.
    This is specifically used for integrators where 
    containers should be coordinating together.
    e.g. DistributedGpTreeIntegrator

    Attributes
    -----------
    n_samples : int
        number of samples to draw in each container
    n_splits : int
        number of K-fold cross-validation splits
        if n_splits = 0, K-Fold CV will not be performed. 
    scoring : Callable
        A scoring function to evaluate the GP predictions. 
        It must accept three arguments: 
        the true values, the predicted values, posterior std. 
        If not provided, default is predictive log likelihood
    score_direction : str
        one of 'up' and 'down'
        if up, higher score is better
        if down, lower score is better
    GPFit : Type[GPFit]
        any subclass of GPFit
        for fitting gaussian process
    gp_params : dict
        The parameters for initialising
        GPFit object
    fit_residuals : bool
        if True, GP is fitted to residuals
        instead of samples
    sampler : Sampler
        used for drawing samples in the container
    """
    def __init__(self, n_samples: int=20,
                 n_splits: int=4, fit_residuals: bool=True, 
                 scoring: Optional[Callable]=r2, score_direction='up',
                 GPFit: Type[GPFit]=SklearnGPFit, gp_params: dict={}, 
                 sampler: Sampler=UniformSampler()) -> None:
        super().__init__(n_samples, n_splits, fit_residuals, scoring, score_direction,
                         GPFit, gp_params, sampler)

    def containerIntegral(self, container: Container, 
                          f: Callable[..., np.ndarray], return_std: bool=False,
                          previous_samples: Optional[Tuple[np.ndarray, 
                                                           np.ndarray]] = None):
        if self.n_samples < 2:
            raise RuntimeError("Cannot perform GP Integral with less than 2 samples"
                               "please increase 'n_samples'")
        
        begin_n = container.N

        # Draw new samples
        new_xs, new_ys = self.sampler.rvs(self.n_samples, container.mins,  
                                  container.maxs, f)
        
        # check samples are correct
        if new_xs.shape[0] != new_ys.shape[0]:
            raise ValueError("the shape of xs and ys generated by sampler does not match")
        if new_xs.shape[0] != self.n_samples:
            raise RuntimeError("Number of samples do not match self.n_samples! "
                               f"sampler.rvs generated {new_xs.shape[0]} samples"
                               f"while expecting {self.n_samples}")

        if previous_samples:
            # Combine previous samples with the new ones
            xs = np.vstack([previous_samples[0], new_xs])
            ys = np.vstack([previous_samples[1], new_ys])
            n_prev = previous_samples[0].shape[0]
        else:
            xs = new_xs
            ys = new_ys
            n_prev = 0

        if xs.shape[0] == 0: 
            raise ValueError(f'xs has no samples, got {n_prev} previous samples'
                             f'and {self.n_samples} new samples')

        # Compute pairwise distances and determine bounds
        pairwise_dists = pairwise_distances(xs)
        mask = np.eye(pairwise_dists.shape[0], dtype=bool)
        pairwise_dists = np.ma.masked_array(pairwise_dists, mask)

        mean_dist = np.mean(pairwise_dists)
        D = xs.shape[1]
        initial_length = mean_dist / np.sqrt(D)
        smallest_dist = np.min(pairwise_dists)
        largest_dist = np.max(pairwise_dists)

        min_bound = 1e-5
        lower_bound = max(smallest_dist / 10, min_bound)
        upper_bound = max(largest_dist * 10, min_bound)
        bounds = (lower_bound, upper_bound)

        self.kernel = RBF(initial_length, bounds)

        gp = self.GPFit(**self.gp_params)
        iGP = IterativeGPFitting(n_samples=self.n_samples, n_splits=self.n_splits, 
                                 threshold_direction='up', performance_threshold=0.0,
                                 max_redraw=0, scoring=self.scoring,
                                 gp=gp, fit_residuals=self.fit_residuals)

        # Fit the GP with all accumulated samples
        gp_results = iGP.fit(f, container, self.kernel, 
                             initial_samples=(xs, ys), add_samples=False)
        
        container.add(new_xs, new_ys)

        # Perform integration using the fitted GP model
        ret = kernel_integration(iGP, container, gp_results, 
                                 return_std)
        gp = iGP.gp

        ret['hyper_params'] = {'length': gp.hyper_params['length_scale']}
        ret['performance'] = gp_results['performance']

        if container.N - begin_n != self.n_samples:
            raise RuntimeError(f"added {container.N - begin_n}, "
                               f"but expecting {self.n_samples}")

        return ret, (xs, ys)