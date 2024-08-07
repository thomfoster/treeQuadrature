import numpy as np
import warnings
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from typing import Callable, Union, Optional, List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import Kernel, RBF, Sum, Product
from sklearn.metrics import r2_score, mean_squared_error
from scipy.special import erf
from scipy.optimize import fmin_l_bfgs_b
from scipy.integrate import quad

from abc import ABC, abstractmethod

from .container import Container


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

    def __init__(self, n_tuning: int=10, max_iter: int=1e4, factr: float=1e7, 
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
        ignore_warning : bool, optional (default=True)
            If True, Convergence Warning of GP Regressor will be ignored.
        """
        super().__init__()

        self.ignore_warning = ignore_warning
        self.n_tuning = n_tuning
        self.max_iter = max_iter
        self.factr = factr
    
    def fit(self, xs, ys, kernel: Kernel) -> GaussianProcessRegressor:
        # Define a constant mean function with the mean of ys
        gp = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=self.n_tuning, 
            optimizer=self._optimizer
        )

        # Fit the GP model without convergence warnings
        if self.ignore_warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(xs, ys)
                warnings.simplefilter("default", ConvergenceWarning)
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
            return self.gp.predict(xs, return_std)
        
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
               n_splits: int=5, scoring: Callable = r2_score):
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
    scoring : Callable, optional (default=r2_score)
        A scoring function to evaluate the predictions. It must accept two 
        arguments: the true values and the predicted values.

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
    y_true = []
    y_pred = []

    for train_index, test_index in kf.split(xs):
        xs_train, xs_test = xs[train_index], xs[test_index]
        ys_train, ys_test = ys[train_index], ys[test_index]

        gp.fit(xs_train, ys_train, kernel)

        y_pred.extend(gp.predict(xs_test))
        y_true.extend(ys_test)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    performance = scoring(y_true, y_pred)

    return performance



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
    scoring : Callable, optional (default=r2_score)
        A scoring function to evaluate the predictions. It must accept two 
        arguments: the true values and the predicted values.
    max_redraw : int
        Maximum number of iterations for the iterative process.
        To control number of samples, 
        recommended not exceed 10
    performance_threshold : float
        Performance threshold for the r2 score to 
        stop the iterative process.
    threshold_direction : str
        one of 'up' and 'down'. 
        if 'up', accept the model if score >= performance_threshold; 
        if 'down', accept the model if score >= performance_threshold
    fit_residuals : bool
        if true, fit GP to residuals
    """
    def __init__(self, n_samples: int, max_redraw: int, n_splits: int,
                 performance_threshold: float, threshold_direction: str, 
                 gp: Optional[GPFit]=None, scoring: Callable = r2_score, 
                 fit_residuals: bool=True) -> None:
        self.n_samples = n_samples
        self.max_redraw = max_redraw
        self.performance_threshold = performance_threshold
        if threshold_direction not in ['up', 'down']:
            raise ValueError("thershold_direction should be one of 'up' and 'down'")
        self.threshold_direction = threshold_direction
        if gp is None:
            gp = SklearnGPFit()
        self.gp = gp
        self.n_splits = n_splits
        self.scoring = scoring
        self.fit_residuals = fit_residuals

    def fit(self, f: Callable, container: Union[Container, List[Container]], 
                 kernel, add_samples: bool=True) -> dict:
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

        Return
        ------
        dict
            - performance (float) the performance of best GP model under KFold CV
            - y_mean (float) mean of the final training samples
        """
        iteration = 0

        all_xs, all_ys = None, None

        while iteration <= self.max_redraw:
            # Draw samples
            if isinstance(container, list):
                samples = self.draw_samples_from_containers(container, 
                                                                self.n_samples, f)
            else:
                xs = container.rvs(self.n_samples)
                ys = f(xs)
                samples = [(xs, ys, container)]

            # Extract samples for fitting
            xs = np.vstack([s[0] for s in samples])
            ys = np.vstack([s[1] for s in samples])

            # Fit the GP model
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

            if self.n_splits == 0: # no cross validation
                self.gp.fit(all_xs, residuals, kernel)
                ys_pred = self.gp.predict(all_xs)
                performance = self.scoring(residuals, ys_pred)
            elif self.n_splits > 0:
                performance = gp_kfoldCV(all_xs, residuals, kernel, self.gp, 
                                         scoring=self.scoring, n_splits=self.n_splits)
            else:
                raise ValueError('n_splits cannot be negative')

            if not is_poor_fit(performance, self.performance_threshold, 
                               self.threshold_direction):
                break

            if add_samples:
                # Add samples to respective containers
                for (xs, ys, c) in samples:
                    c.add(xs, ys)

            iteration += 1

        # Final fit with the full set of samples
        if self.n_splits > 0:
            self.gp.fit(all_xs, residuals, kernel)

        return {
            'performance': performance,
            'y_mean': mean_y
        }
    
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
        ## TODO - allow user defined weights
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

def default_criterion(container: Container) -> bool:
    return True

def GP_diagnosis(igp: IterativeGPFitting, container: Container, 
                 criterion: Callable[[Container], 
                                     bool]=default_criterion, 
                                     plot: bool=False) -> None:
    """
    Check the performance of a Gaussian Process (GP) model.
    
    Parameters
    ----------
    igp : IterativeGPFitting
        The fitted Gaussian Process model.
    container : Container
        Container object that holds the samples and boundaries.
    criterion : function, Optional
        A function that takes a container and returns a boolean indicating 
        whether to plot the posterior mean.
        Default criterion: always True
    plot : bool, optional
        if true, 1D problems will be plotted
        Default: False
    
    Returns
    -------
    None
    """
    xs = igp.gp.X_train_
    ys = igp.gp.y_train_
    n = xs.shape[0]

    # Make predictions
    y_pred = igp.gp.predict(xs)

    # Check R-squared and MSE
    score = igp.scoring(ys, y_pred)
    mse = mean_squared_error(ys, y_pred)

    if is_poor_fit(score, igp.performance_threshold, 
                   igp.threshold_direction):
        print(f'number of training samples : {n}')
        print(f'volume of container : {container.volume}')
        print(f"GP Score: {score:.3f}")
        print(f"Mean Squared Error: {mse:.3f}") 

    # posterior mean plot
    if xs.shape[1] == 1 and criterion(container) and plot:
        plotGP(igp.gp, xs, ys, 
               mins=container.mins, maxs=container.maxs)

    
def get_length_scale(kernel):
    """Recursively find the length scale in a composite kernel."""
    if isinstance(kernel, RBF):
        return kernel.length_scale
    if hasattr(kernel, 'k1'):
        return get_length_scale(kernel.k1)
    if hasattr(kernel, 'k2'):
        return get_length_scale(kernel.k2)
    raise ValueError("No RBF kernel found in the composite kernel.")

def rbf_mean_post(gp: GPFit, container: Container, gp_results: dict):
    """
    calculate the posterior mean of integral using RBF kernel

    Returns
    -------
    float, numpy.ndarray
        the posterior mean fo integral and partial mean of kernel
    """

    ### Extract necessary information
    l = get_length_scale(gp.kernel_)
    
    xs = gp.X_train_

    # Container boundaries
    b = container.maxs   # right boarder
    a = container.mins   # left boarder

    K_inv_y = gp.alpha_.reshape(-1)

    ### apply the formulae using scipy.erf
    erf_b = erf((b - xs) / (np.sqrt(2) * l))
    erf_a = erf((a - xs) / (np.sqrt(2) * l))
    k_tilde = (erf_b - erf_a).prod(axis=1) * (
        (l * np.sqrt(np.pi / 2)) ** container.D)

    try:
        y_mean = gp_results['y_mean']
    except KeyError:
        raise KeyError('cannot find y_mean in gp_results')
    
    return np.dot(K_inv_y, k_tilde) + y_mean * container.volume, k_tilde


def rbf_var_post(container: Container, gp: GPFit, k_tilde: np.ndarray, 
                 jitter: float = 1e-8, threshold: float = 1e10):
    """calculate the posterior variance of integral estimate obtained using RBF kernel"""
    b = container.maxs   # right boarder
    a = container.mins   # left boarder

    xs = gp.X_train_

    l = get_length_scale(gp.kernel_)

    def integrand(x_j_prime, a_j, b_j):
                return erf((b_j - x_j_prime) / (np.sqrt(2) * l)
                        ) - erf(
                            (a_j - x_j_prime) / (np.sqrt(2) * l)
                            )
    result = 1
    for j in range(container.D):
        integ_j, _ = quad(integrand, a[j], b[j], args=(a[j], b[j]))
        result *= integ_j
    k_mean = l * np.sqrt(np.pi / 2) * result

    K = gp.kernel_(xs)
    if np.linalg.cond(K) > threshold:
        K += np.eye(K.shape[0]) * jitter
    K_inv = np.linalg.inv(K)
    # posterior variance
    var_post = k_mean - np.dot(k_tilde.T, np.dot(K_inv, k_tilde))

    return var_post


def contains_rbf(kernel: Kernel) -> bool:
    """
    Check if the given kernel or any of its components is an RBF kernel.

    Parameters
    ----------
    kernel : Kernel
        The kernel to check.

    Returns
    -------
    bool
        True if the kernel or any of its components is an RBF kernel, False otherwise.
    """
    if isinstance(kernel, RBF):
        return True
    elif isinstance(kernel, (Sum, Product)):
        return contains_rbf(kernel.k1) or contains_rbf(kernel.k2)
    return False

def kernel_integration(igp: IterativeGPFitting, container: Container, 
                       gp_results: dict, return_std: bool, 
                    kernel_mean_post: Optional[Callable]=None,
                    kernel_var_post: Optional[Callable]=None,
                    kernel_post: Optional[Callable]=None) -> dict:
    """
    Estimate the integral of the RBF kernel over 
    a given container and set of points.

    Parameters
    ----------
    igp : IterativeGPFitting
        The fitted Gaussian Process model.
    container : Container
        The container object that holds the boundaries.
    gp_results : dict
        results from gp.fit() necessary for performing the 
        kernel integral
    return_std : bool
        When True, return the standard deviation of GP estimation
    kernel_mean_post, kernel_var_post : functions, optional
        must be provided if not using RBF kernel
        kernel_mean_post : takes a GPFit, Container, 
          and dictionary (gp fitting results)
          and returns a float (integral estimate)
        kernel_var_post : takes a GPFit, Container, 
          and returns a float (posterior variance)
    kernel_post : function, optional
        alternative to kernel_mean_post, kernel_var_post
        must take GPFit, Container, dict (gp fitting results) 
          and return_std : bool
          and return estiamte and var_post simulaneously 

    Returns
    -------
    dict
        - integral (float) the integral estimate
        - std (float) standard deviation of integral
    """
    gp = igp.gp

    if contains_rbf(gp.kernel_):   # RBF kernel        
        integral, k_tilde = rbf_mean_post(gp, container, gp_results)

        if return_std:
            try:
                performance = gp_results['performance']
            except KeyError:
                raise KeyError('cannot find performance in gp_results')
            
            # filter out GP with poor fits
            if igp.performance_threshold is not None and (
                performance is not None) and (
                igp.threshold_direction is not None):
                if is_poor_fit(performance, igp.performance_threshold, 
                               igp.threshold_direction):
                    warnings.warn(
                        'Warning: GP fitness is poor'
                        f' score = {performance}'
                        '. std will be set to zero.', 
                        UserWarning)
                    var_post = 0
                else: 
                    var_post = rbf_var_post(container, gp, k_tilde)
            else:
                # mean of kernel on the container
                var_post = rbf_var_post(container, gp, k_tilde)
    elif kernel_mean_post is not None and (
        kernel_var_post is not None):  
        integral = kernel_mean_post(gp, container, gp_results)
        if return_std:
            var_post = kernel_var_post(gp, container)
    elif kernel_post is not None:
        integral, var_post = kernel_post(gp, container, gp_results, return_std)
    else:
        raise Exception(
            'kernel not RBF, '
            'either kernel_mean_post, kernel_var_post '
            'or kernel_post must be provided'
            )
    
    ret = {'integral' : integral}
            
    # value check
    if return_std and var_post < 0:
        if var_post < -1e-2:
            warnings.warn(
                'Warning: variance estimate in a container is negative'
                f' with value : {var_post}'
                '. Will be set to zero.', 
                UserWarning)
            print('------ GP diagnosis --------')
            GP_diagnosis(igp, container)
            print('----------------------------')
        var_post = 0

    if return_std:
        ret['std'] = np.sqrt(var_post)

    return ret


def plotGP(gp: GPFit, xs: np.ndarray, ys: np.ndarray, 
           mins: np.ndarray, maxs: np.ndarray, plot_ci: Optional[bool]=True):
    """
    Plot the Gaussian Process posterior mean and the data points.

    Parameters
    ----------
    gp : GPFit
        The trained GP model.
    xs, ys : numpy.ndarray
        The data points.
    mins : np.ndarray
        The lower bounds for plotting.
    maxs : np.ndarray
        The upper bounds for plotting.
    plot_ci : bool, optional
        If True, the confidence interval will be plotted. Default is True.
    """
    if xs.shape[1] == 1:
        _plotGP1D(gp, xs, ys, mins[0], maxs[0], plot_ci)
    elif xs.shape[1] == 2:
        assert len(mins) == 2 and len(maxs) == 2, (
            'mins and maxs must have two elements for 2-dimensional problems'
            )
        _plotGP2D(gp, xs, ys, mins[0], maxs[0], mins[1], maxs[1], plot_ci)
    else:
        raise ValueError(
            'This function only supports 1-dimensional and 2-dimensional problems'
            )

def _plotGP1D(gp: GPFit, xs: np.ndarray, ys: np.ndarray, 
           x_min: float, x_max: float, plot_ci: Optional[bool]=True):
    """
    Plot the Gaussian Process posterior mean
    and the data points

    Parameters
    ----------
    gp : GPFit
        the trained GP model
    xs, ys : numpy.ndarray
        the data points
    x_min, x_max : float
        the lower and upper bounds for plotting
    plot_ci : bool
        if True, the confidence interval will be plotted.
        Default True
    """
    assert xs.shape[1] == 1, 'only supports 1-dimensional problems'

    x_plot = np.linspace(x_min, x_max, 1000).reshape(-1, 1)

    # Predict the mean and standard deviation of the GP model
    y_mean, y_std = gp.predict(x_plot, return_std=True)

    # Plot the original points
    plt.scatter(xs, ys, c='r', marker='x', label='Data points')

    # Plot the GP mean function
    plt.plot(x_plot, y_mean, 'b-', label='GP mean')

    # Plot the confidence interval
    if plot_ci:
        plt.fill_between(x_plot.ravel(),
                        y_mean - 1.96 * y_std,
                        y_mean + 1.96 * y_std,
                        alpha=0.2,
                        color='b',
                        label='95% confidence interval')

    # Add labels and legend
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.show()


def _plotGP2D(gp: GPFit, xs: np.ndarray, ys: np.ndarray, 
           x_min: float, x_max: float, y_min: float, y_max: float, 
           plot_ci: Optional[bool]=True):
    """
    Plot the Gaussian Process posterior mean and the data points for a 2D problem.

    Parameters
    ----------
    gp : GPFit
        The trained GP model.
    xs, ys : numpy.ndarray
        The data points.
    x_min, x_max : float
        The lower and upper bounds for the first axis (x-axis) for plotting.
    y_min, y_max : float
        The lower and upper bounds for the second axis (y-axis) for plotting.
    plot_ci : bool, optional
        If True, the confidence interval will be plotted. Default is True.
    """
    assert xs.shape[1] == 2, 'This function only supports 2-dimensional problems'

    # Create a grid over the input space
    x1_plot = np.linspace(x_min, x_max, 100)
    x2_plot = np.linspace(y_min, y_max, 100)
    x1_grid, x2_grid = np.meshgrid(x1_plot, x2_plot)
    x_plot = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T

    # Predict the mean and standard deviation of the GP model
    y_mean, y_std = gp.predict(x_plot, return_std=True)
    y_mean = y_mean.reshape(x1_grid.shape)
    y_std = y_std.reshape(x1_grid.shape)

    # Plot the GP mean function
    plt.figure(figsize=(12, 6))
    plt.contourf(x1_grid, x2_grid, y_mean, cmap='viridis', alpha=0.8)
    plt.colorbar(label='GP mean')

    # Plot the original points
    scatter = plt.scatter(xs[:, 0], xs[:, 1], c=ys, cmap='viridis', edgecolors='k', 
                          marker='o', label='Data points')

    # Plot the confidence interval
    if plot_ci:
        ci_lower = y_mean - 1.96 * y_std
        ci_upper = y_mean + 1.96 * y_std
        plt.contour(x1_grid, x2_grid, ci_lower, levels=1, colors='blue', linestyles='dashed', alpha=0.5)
        plt.contour(x1_grid, x2_grid, ci_upper, levels=1, colors='red', linestyles='dashed', alpha=0.5)

        # Add dummy plots for the legend
        lower_dummy_line = plt.Line2D([0], [0], linestyle='dashed', color='blue', alpha=0.5)
        upper_dummy_line = plt.Line2D([0], [0], linestyle='dashed', color='red', alpha=0.5)
        plt.legend([scatter, lower_dummy_line, upper_dummy_line], ['Data points', 
                                                                   '95% CI Lower Bound', 
                                                                   '95% CI Upper Bound'])

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Gaussian Process Regression for 2D Problems')
    plt.show()