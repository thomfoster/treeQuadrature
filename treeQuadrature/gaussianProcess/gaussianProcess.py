import numpy as np
import warnings
from numpy.typing import ArrayLike
from typing import Callable, Union, Optional, List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import Kernel
from scipy.stats import norm

from scipy.optimize import fmin_l_bfgs_b

from abc import ABC, abstractmethod

from ..container import Container
from .scorings import r2


class GPFit(ABC):
    """
    Abstract base class for fitting Gaussian Process models.

    Subclasses must implement the `fit_GP` method.
    """

    def __init__(self):
        self.gp = None

    @abstractmethod
    def fit(self, xs, ys, kernel):
        """
        Each run of fit should re-define self.gp

        Arguments
        ---------
        xs, ys : array like
            the training dataset 
        kerenl : Any
            the covariance (similarity measure)  

        Return
        ------
        the fitted GP model
        """
        pass

    @abstractmethod
    def predict(self, xs, return_std: bool):
        """
        Arguments
        ---------
        xs : array like
            the input
        return_std : bool
            whether return the std of the predictions or not

        Return
        ------
        the predicted y (output) values and 
        standard deviation if return_std = True
        """
        pass

    @property
    @abstractmethod
    def X_train_(self) -> ArrayLike:
        """
        Returns the training set used to fit the model
        """
        pass

    @property
    @abstractmethod
    def y_train_(self) -> ArrayLike:
        """
        Returns the training labels (outputs) used to fit the model
        """
        pass

    @property
    @abstractmethod
    def hyper_params(self) -> dict:
        """
        Returns the hyper-parameters of the fitted gp 
        as an dictionary
        """
        pass

    @property
    @abstractmethod
    def kernel_(self):
        """
        Returns the kernel used in gp 
        """
        pass


class SklearnGPFit(GPFit):
    """
    Default implementation of GP fitting using 
    sklearn.gaussian_process.GaussianProcessRegressor

    Attributes
    ----------
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        the fitted Gaussian model. 
        must run fit_gp before accessing gp
    n_tuning : int
        Number of restarts of the optimizer for finding the 
        kernel's parameters.
    ignore_warning : bool, optional (default=True)
        If True, Convergence Warning of GP Regressor will be ignored.
    optimizer : function
        the optimizer used to tune hyper-parameters of gp
        Default is fmin_l_bfgs_b

    Method
    -------
    fit_gp(xs, ys, kernel)
        fit gp and return the model. gp will also be stored in attribute
    predict(xs)

    Example
    -------
    >>> gp = DefaultGPFit()
    >>> kernel = sklearn.gaussian_process.kernels.RBF(1.0, (1e-2, 1e2))
    >>> gp.fit_gp(xs_train, ys_train, kernel)
    >>> ys_pred = gp_fitter.predict(xs_train)
    """

    def __init__(self, n_tuning: int=10, max_iter: float=1e4, factr: float=1e7, 
                 alpha: float=1e-10,
                 ignore_warning: bool=True) -> None:
        """
        Arguments
        ---------
        n_tuning : int
            Number of restarts of the optimizer for finding the 
            kernel's parameters.
            Default : 10
        factr : int or float
            convergence criteria for fmin_l_bfgs_b optimiser
            used to fit Gaussian Process. 
            Default : 1e7
        max_iter : int
            maximum number of iterations for fmin_l_bfgs_b optimiser
            Default : 1e4
        alpha : float, optional (default=1e-10)
            small jitter added to the diagonal of kernel matrix
            in GaussianProcessRegressor for numerical
            stability
        ignore_warning : bool, optional (default=True)
            If True, Convergence Warning of GP Regressor will be ignored.
        """
        super().__init__()

        self.ignore_warning = ignore_warning
        if not isinstance(n_tuning, int):
            raise ValueError("n_tuning must be an integer")
        self.n_tuning = n_tuning
        if not isinstance(max_iter, (float, int)):
            raise ValueError("max_iter must be an float or integer")
        self.max_iter = max_iter
        if not isinstance(alpha, float):
            raise ValueError("alpha must be an float")
        self.alpha = alpha
        if not isinstance(factr, (float, int)):
            raise ValueError("factr must be an float or integer")
        self.factr = factr
    
    def fit(self, xs, ys, kernel: Kernel) -> GaussianProcessRegressor:
        # Define a constant mean function with the mean of ys
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=self.alpha,
            n_restarts_optimizer=self.n_tuning, 
            optimizer=self._optimizer
        )

        # Fit the GP model without convergence warnings
        if self.ignore_warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning) 
                gp.fit(xs, ys)
                warnings.simplefilter("default", ConvergenceWarning)
                warnings.simplefilter("default", UserWarning) 
        else:
            gp.fit(xs, ys)

        self.gp = gp

        return gp
    
    def _optimizer(self, obj_func, initial_theta, bounds):
        return fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, 
                             maxiter=self.max_iter, factr=self.factr)[:2]
    
    def predict(self, xs, return_std: bool=False):
        if self.gp is None:
            raise RuntimeError(
                "The Gaussian Process model is not trained. Please call 'fit_gp' before 'predict'."
                )
        else: 
            if self.ignore_warning:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning) 
                    predictions = self.gp.predict(xs, return_std)
                    warnings.simplefilter("default", UserWarning) 
            else:
                predictions = self.gp.predict(xs, return_std)
                
            return predictions
        
    @property
    def X_train_(self) -> ArrayLike:
        if self.gp is None:
            raise RuntimeError(
                "The Gaussian Process model is not trained. Please call 'fit_gp' before 'predict'."
                )
        else: 
            return self.gp.X_train_
    @property
    def y_train_(self) -> ArrayLike:
        if self.gp is None:
            raise RuntimeError(
                "The Gaussian Process model is not trained. Please call 'fit_gp' before 'predict'."
                )
        else: 
            return self.gp.y_train_
    @property
    def hyper_params(self) -> dict:
        if self.gp is None:
            raise RuntimeError(
                "The Gaussian Process model is not trained. Please call 'fit_gp' before 'predict'."
                )
        else: 
            return self.gp.kernel_.get_params()
    @property
    def alpha_(self) -> ArrayLike:
        if self.gp is None:
            raise RuntimeError(
                "The Gaussian Process model is not trained. Please call 'fit_gp' before 'predict'."
                )
        else: 
            return self.gp.alpha_
    @property
    def kernel_(self) -> Kernel:
        if self.gp is None:
            raise RuntimeError(
                "The Gaussian Process model is not trained. Please call 'fit_gp' before 'predict'."
                )
        else: 
            return self.gp.kernel_



def gp_kfoldCV(xs, ys, kernel, gp: GPFit, 
               n_splits: int=5, scoring: Callable = r2):
    """
    Perform k-fold Cross-Validation (CV) to evaluate the 
    performance of a Gaussian Process model.

    Parameters
    ----------
    xs : array-like, shape (n_samples, n_features)
        Training data.
    ys : array-like, shape (n_samples,)
        Target values.
    kernel : Any
        The kernel specifying the covariance function of the GP.
    gp_fitter : GPFitBase
        An instance of a GPFitBase subclass for fitting the GP. 
    n_splits : int, optional (default=5)
        Number of folds for cross-validation.
    scoring : Callable, optional (default=predictive_ll)
        A scoring function to evaluate the predictions. It must accept three 
        arguments: the true values, the predicted values and predicted variance

    Returns
    -------
    performance : float
        Performance measure using k-fold CV based on the provided scoring function.
    """

    assert len(xs) == len(ys), (
        "The number of samples in xs and ys must match. "
        f"shape of xs : {xs.shape}; "
        f"shape of ys : {ys.shape}"
    )

    ys = np.ravel(ys)
    
    kf = KFold(n_splits=n_splits)
    scores = []

    for train_index, test_index in kf.split(xs):
        xs_train, xs_test = xs[train_index], xs[test_index]
        ys_train, ys_test = ys[train_index], ys[test_index]

        gp.fit(xs_train, ys_train, kernel)

        ys_pred, sigma = gp.predict(xs_test, return_std=True)
        score = scoring(ys_test, ys_pred, sigma)
        scores.append(score)

    return np.mean(scores)



def is_poor_fit(performance: float, threshold: float, 
                threshold_direction: str) -> bool:
    if threshold_direction == 'up':
        return performance <= threshold
    elif threshold_direction == 'down':
        return performance >= threshold
    else:
        raise ValueError(
                    "threshold_direction should be one of 'up' and 'down'")

class IterativeGPFitting:
    """
    Class to perform iterative Gaussian Process (GP) fitting, 
    incresaing the number of samples by `n_samples` each time
    until a performance criterion on 
    Cross Validation r2 score is met.

    Parameters
    ----------
    gp : GPFit
        the gp fitter storing the gp model
        default is SklearnGPFit
    n_samples : int
        increment of number of samples each time
    n_splits : int, optional
        number of K-fold cross-validation splits
        if n_splits = 0, K-Fold CV will not be performed. 
    scoring : Callable, optional (default=predictive_ll)
        A scoring function to evaluate the predictions. It must accept two 
        arguments: the true values and the predicted values.
    max_redraw : int
        Maximum number of iterations for the iterative process.
        To control number of samples, 
        recommended not exceed 10
    performance_threshold : float
        Performance threshold for scoring to 
        stop the iterative process.
    threshold_direction : str
        one of 'up' and 'down'. 
        if 'up', accept the model if score >= performance_threshold; 
        if 'down', accept the model if score >= performance_threshold
    fit_residuals : bool
        if true, fit GP to residuals
    """
    def __init__(self, n_samples: int, max_redraw: int, n_splits: int,
                 performance_threshold: float, threshold_direction: str='up', 
                 gp: Optional[GPFit]=None, scoring: Optional[Callable]=None, 
                 fit_residuals: bool=True) -> None:
        self.n_samples = n_samples

        if max_redraw < 0:
            raise ValueError('max_redraw must be a positive integer')
        self.max_redraw = max_redraw

        self.performance_threshold = performance_threshold
        if threshold_direction not in ['up', 'down']:
            raise ValueError(
                "thershold_direction should be one of 'up' and 'down'"
                )
        self.threshold_direction = threshold_direction
        if gp is None:
            gp = SklearnGPFit()
        self.gp = gp
        self.n_splits = n_splits
        if scoring:
            self.scoring = scoring
        else:
            self.scoring = r2
        self.fit_residuals = fit_residuals

    def fit(self, f: Callable, container: Union[Container, List[Container]], 
                 kernel, add_samples: bool=True,
                 initial_samples: Optional[tuple]=None) -> dict:
        """
        fit GP on the container,
        the results can be accessed in self.gp

        Arguments
        ---------
        f : callable
            Function to evaluate the samples.
        container : Container or List[Container]
            Container object(s) to draw samples from 
            and track evaluations.
        kernel : Any
            the kernel used to fit GP
        add_sample : bool, optional (default=True)
            if true, add samples to the container(s)
            iniital_samples will NOT be added
        initial_samples: tuple, optional
            if given, start the fit using initial_samples
            (xs, ys)

        Return
        ------
        dict
            - performance (float) the performance of the best GP model under KFold CV
            - y_mean (float) mean of the final training samples
            - new_samples (list) if add_samples is False, a list of the new samples drawn
        """
        iteration = 0
        all_xs, all_ys = None, None
        new_xs, new_ys = [], []

        while iteration <= self.max_redraw:
            # Draw samples
            if initial_samples and iteration == 0:
                if isinstance(initial_samples, tuple) and len(initial_samples) == 2:
                    xs = initial_samples[0]
                    ys = initial_samples[1]
                    samples = [(xs, ys, container)]
                else:
                    raise ValueError("initial_samples should be a tuple of length 2")
            else:
                if isinstance(container, list):
                    samples = self.draw_samples_from_containers(container, self.n_samples, f)
                else:
                    xs = container.rvs(self.n_samples)
                    ys = f(xs)
                    samples = [(xs, ys, container)]

            # Extract samples for fitting
            xs = np.vstack([s[0] for s in samples])
            ys = np.vstack([s[1] for s in samples])

            # Collect samples
            if all_xs is None:
                all_xs = xs
                all_ys = ys
            else:
                all_xs = np.vstack([all_xs, xs])
                all_ys = np.vstack([all_ys, ys])

            if self.fit_residuals:
                mean_y = np.mean(all_ys)
                residuals = all_ys - mean_y
            else:
                mean_y = 0
                residuals = all_ys

            # fit GP 
            if self.n_splits == 0: # no cross validation
                self.gp.fit(all_xs, residuals, kernel)
                warnings.filterwarnings("ignore", category=UserWarning)
                ys_pred, sigma = self.gp.predict(all_xs, return_std=True)
                warnings.filterwarnings("default", category=UserWarning)
                performance = self.scoring(residuals, ys_pred, sigma)
            elif self.n_splits > 0:
                performance = gp_kfoldCV(xs=all_xs, ys=residuals, kernel=kernel, gp=self.gp, 
                                         scoring=self.scoring, n_splits=self.n_splits)
            else:
                raise ValueError('n_splits cannot be negative')

            # Add samples to respective containers
            if not (initial_samples and iteration == 0):
                if add_samples:
                    for (xs, ys, c) in samples:
                        c.add(xs, ys)
                else:
                    new_xs.append(xs)
                    new_ys.append(ys)

            if not is_poor_fit(performance, self.performance_threshold, 
                               self.threshold_direction):
                break

            iteration += 1

        # Final fit with the full set of samples
        if self.n_splits > 0:
            self.gp.fit(all_xs, residuals, kernel)

        result = {
            'performance': performance,
            'y_mean': mean_y
        }

        if not add_samples:
            if new_xs and new_ys:
                result['new_samples'] = (np.vstack(new_xs), np.vstack(new_ys))
            else:
                D = container.D if isinstance(container, Container) else container[0].D
                result['new_samples'] = (np.empty((0, D)), np.empty((0, 1)))

        return result
    
    def draw_samples_from_containers(self, containers: List[Container], n: int, 
                                 f: Callable) -> List[Tuple[np.ndarray, 
                                                            np.ndarray, 
                                                            Container]]:
        """
        Draw samples randomly from multiple containers.

        Parameters
        ----------
        containers : List[Container]
            The list of containers to draw samples from.
        n : int
            The total number of samples to draw.
        f : callable
            The function to evaluate the samples.

        Returns
        -------
        samples : List[Tuple[np.ndarray, np.ndarray, Container]]
            A list of tuples, each containing samples, their evaluations, 
            and the container they were drawn from.
        """
        # Calculate the total volume
        total_volume = sum(c.volume for c in containers)

        # Calculate the weights based on the volume of each container
        weights = [c.volume / total_volume for c in containers]

        samples = []
        for c, w in zip(containers, weights):
            n_samples = max(1, int(n * w))
            x = c.rvs(n_samples)
            y = f(x)
            samples.append((x, y, c))

        return samples