import numpy as np
import warnings
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from typing import Callable, Union, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import Kernel, RBF
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

        def custom_optimizer(obj_func, initial_theta, bounds):
            result = fmin_l_bfgs_b(obj_func, initial_theta, 
                                bounds=bounds, 
                                maxiter=max_iter, 
                                factr=factr)
            return result[0], result[1]
    
        self.optimizer = custom_optimizer
        
    
    def fit(self, xs, ys, kernel: Kernel) -> GaussianProcessRegressor:
        gp = GaussianProcessRegressor(kernel=kernel, 
                                      n_restarts_optimizer=self.n_tuning, 
                                      optimizer=self.optimizer)

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
    """
    def __init__(self, n_samples: int, max_redraw: int, n_splits: int,
                 performance_threshold: float, threshold_direction: str, 
                 gp: Optional[GPFit]=None):
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

    def fit(self, f: Callable, container: Container, 
                 kernel, scoring: Callable = r2_score) -> float:
        """
        fit GP on the container,
        the results can be accessed in self.gp

        Arguments
        ---------
        f : callable
            Function to evaluate the samples.
        container : Container
            Container object to draw samples from 
            and track evaluations.
        kernel : Any
            the kernel used to fit GP
        scoring : Callable, optional (default=r2_score)
            A scoring function to evaluate the predictions. It must accept two 
            arguments: the true values and the predicted values.

        Return
        ------
        performance : float
            the performance of best GP model under KFold CV
        """
        n = self.n_samples
        iteration = 0
        while iteration < self.max_redraw:
            # Redraw uniform samples from the container
            xs = container.rvs(n)
            ys = f(xs)

            if self.n_splits == 0:
                self.gp.fit(xs, ys, kernel)
                ys_pred = self.gp.predict(xs)
                performance = scoring(ys, ys_pred)
            elif self.n_splits > 0:
                performance = gp_kfoldCV(xs, ys, kernel, self.gp, 
                                        scoring=scoring, n_splits=self.n_splits)
            else:
                raise ValueError('n_splits cannot be negative')

            if self.threshold_direction == 'up':
                if performance >= self.performance_threshold:
                    break
            elif self.threshold_direction == 'down':
                if performance <= self.performance_threshold:
                    break
            else:
                raise ValueError(
                    "thershold_direction should be one of 'up' and 'down'")

            # increase the number of samples for the next iteration
            n += self.n_samples  
            iteration += 1

        # replace the GP model fitted in K-Fold CVv
        if self.n_splits > 0:
            self.gp.fit(xs, ys, kernel)

        return performance

def default_criterion(container: Container) -> bool:
    return True

def GP_diagnosis(gp: GPFit, container: Container, 
                 criterion: Callable[[Container], 
                                     bool]=default_criterion) -> None:
    """
    Check the performance of a Gaussian Process (GP) model.
    
    Parameters
    ----------
    gp : GPFit
        The fitted Gaussian Process model.
    container : Container
        Container object that holds the samples and boundaries.
    criterion : function, Optional
        A function that takes a container and returns a boolean indicating 
        whether to plot the posterior mean.
        Default criterion: always True
    
    Returns
    -------
    None
    """
    xs = gp.X_train_
    ys = gp.y_train_

    # Make predictions
    y_pred, sigma = gp.predict(xs, return_std=True)
    print(f'average predictive variance {np.mean(sigma)}')

    # Check R-squared and MSE
    r2 = r2_score(ys, y_pred)
    mse = mean_squared_error(ys, y_pred)
    print(f"R-squared: {r2:.3f}")
    print(f"Mean Squared Error: {mse:.3f}") 

    # posterior mean plot
    if (xs.shape[1] == 1 or xs.shape[1] == 2
        ) and criterion(container):
        plotGP(gp, xs, ys, 
               mins=container.mins, maxs=container.maxs)

    
def rbf_mean_post(gp: GPFit, container: Container):
    """
    calculate the posterior mean of integral using RBF kernel

    Returns
    -------
    float, numpy.ndarray
        the posterior mean fo integral and partial mean of kernel
    """

    ### Extract necessary information
    try:
        l = gp.hyper_params['length_scale']
    except KeyError:
        raise KeyError(
            "The hyperparameter 'length_scale' is missing in the GP model. "
            f"hyper_params = {gp.hyper_params}"
            )
    
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

    return np.dot(K_inv_y, k_tilde), k_tilde


def rbf_var_post(container: Container, gp: GPFit, k_tilde: np.ndarray):
    """calculate the posterior variance of integral estimate obtained using RBF kernel"""
    b = container.maxs   # right boarder
    a = container.mins   # left boarder

    xs = gp.X_train_

    try:
        l = gp.hyper_params['length_scale']
    except KeyError:
        raise KeyError(
            "The hyperparameter 'length_scale' is missing in the GP model. "
            f"hyper_params = {gp.hyper_params}"
            )

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
    K_inv = np.linalg.inv(K)
    # posterior variance
    var_post = k_mean - np.dot(k_tilde.T, np.dot(K_inv, k_tilde))

    return var_post


def kernel_integration(gp: GPFit, container: Container, 
                    return_std: bool, performance: Optional[float]=None, 
                    threshold: Optional[float]=None, 
                    threshold_direction: Optional[str]=None, 
                    kernel_mean_post: Optional[Callable]=rbf_mean_post,
                    kernel_var_post: Optional[Callable]=rbf_var_post,
                    kernel_post: Optional[Callable]=None) -> Union[float, tuple]:
    """
    Estimate the integral of the RBF kernel over 
    a given container and set of points.

    Parameters
    ----------
    gp : GPFit
        The fitted Gaussian Process model.
    container : Container
        The container object that holds the boundaries.
    return_std : bool
        When True, return the standard deviation of GP estimation
    performance : float
        the performance of the GP model under KFold CV
    threshold : float
        Performance threshold for the r2 score to 
        stop the iterative process.
    threshold_direction : str
        one of 'up' and 'down'. 
        if 'up', accept the model if score >= performance_threshold; 
        if 'down', accept the model if score >= performance_threshold
    kernel_mean_post, kernel_var_post : functions, optional
        must be provided if not using RBF kernel
        both take GPFit and Container and returns a float
    kernel_post : function, optional
        alternative to above, 
        must take GPFit and Container and return_std : bool
        and return estiamte and var_post simulaneously 

    Returns
    -------
    float or tuple
        float is the estimated integral value, 
        tuple has length 2, the second value is 
          integral evaluation std.
    """
    if isinstance(gp.kernel_, RBF):   #RBF kernel
        integral, k_tilde = kernel_mean_post(gp, container)

        if return_std:
            def is_poor_fit(performance, threshold, direction):
                if direction == 'up':
                    return performance <= threshold
                elif direction == 'down':
                    return performance >= threshold
                return False
            
            # filter out GP with poor fits
            if threshold is not None and performance is not None and (
                threshold_direction is not None):
                if is_poor_fit(performance, threshold, threshold_direction):
                    warnings.warn(
                        'Warning: GP fitness is poor'
                        f' score = {performance}'
                        '. std will be set to zero.', 
                        UserWarning)
                    var_post = 0
                else: 
                    var_post = kernel_var_post(container, gp, k_tilde)
            else:
                # mean of kernel on the container
                var_post = rbf_var_post(container, gp, k_tilde)
    elif kernel_mean_post is not None and kernel_var_post is not None:  
        integral = kernel_mean_post(gp, container)
        if return_std:
            var_post = kernel_var_post(gp, container)
    elif kernel_post is not None:
        integral, var_post = kernel_post(gp, container, return_std)
    else:
        raise Exception(
            'kernel not RBF, '
            'either kernel_mean_post, kernel_var_post or kernel_post must be provided'
            )
            
    # value check
    if return_std and var_post < 0:
        if var_post < -1e-2:
            warnings.warn(
                'Warning: variance estimate in a container is negative'
                f' with value : {var_post}'
                '. Will be set to zero.', 
                UserWarning)
            print('------ GP diagnosis --------')
            GP_diagnosis(gp, container)
            print('----------------------------')
        var_post = 0

    if return_std:
        return (integral, np.sqrt(var_post))
    else:
        return integral


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
        raise ValueError('This function only supports 1-dimensional and 2-dimensional problems')

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
    scatter = plt.scatter(xs[:, 0], xs[:, 1], c=ys, cmap='viridis', edgecolors='k', marker='o', label='Data points')

    # Plot the confidence interval
    if plot_ci:
        ci_lower = y_mean - 1.96 * y_std
        ci_upper = y_mean + 1.96 * y_std
        plt.contour(x1_grid, x2_grid, ci_lower, levels=1, colors='blue', linestyles='dashed', alpha=0.5)
        plt.contour(x1_grid, x2_grid, ci_upper, levels=1, colors='red', linestyles='dashed', alpha=0.5)

        # Add dummy plots for the legend
        lower_dummy_line = plt.Line2D([0], [0], linestyle='dashed', color='blue', alpha=0.5)
        upper_dummy_line = plt.Line2D([0], [0], linestyle='dashed', color='red', alpha=0.5)
        plt.legend([scatter, lower_dummy_line, upper_dummy_line], ['Data points', '95% CI Lower Bound', '95% CI Upper Bound'])

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Gaussian Process Regression for 2D Problems')
    plt.show()