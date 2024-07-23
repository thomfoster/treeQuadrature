import numpy as np
import warnings
from typing import Callable, Union
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import Kernel, RBF
from sklearn.metrics import r2_score, mean_squared_error
from scipy.special import erf
from scipy.optimize import fmin_l_bfgs_b
from scipy.integrate import quad

from .visualisation import plotGP
from .container import Container


def fit_GP(xs, ys, kernel: Kernel, n_tuning: int, optimizer: Callable, 
           ignore_warning=True) -> GaussianProcessRegressor:
    """
    Fit a Gaussian Process (GP) model with custom optimization parameters.

    Parameters
    ----------
    xs : array-like, shape (n_samples, n_features)
        Training data.
    ys : array-like, shape (n_samples,)
        Target values.
    kernel : sklearn.gaussian_process.kernels.Kernel
        The kernel specifying the covariance function of the GP.
    n_tuning : int
        Number of restarts of the optimizer for finding the kernel's parameters 
        which maximize the log-marginal likelihood.
    optimizer : function
        to be used in fitting GP
    ignore_warning : bool, optional (default=True)
        If True, Convergence Warning of GP Regressor will be ignored.

    Returns
    -------
    gp : GaussianProcessRegressor
        Fitted Gaussian Process model.
    """
    
    gp = GaussianProcessRegressor(kernel=kernel, 
                                  n_restarts_optimizer=n_tuning, 
                                  optimizer=optimizer)

    ### Fit the GP model without convergence warnings
    if ignore_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(xs, ys)
            warnings.simplefilter("default", ConvergenceWarning)
    else:
        gp.fit(xs, ys)

    return gp

def gp_kfoldCV(xs, ys, kernel: Kernel, n_tuning: int, optimizer: Callable, 
               n_splits=5, ignore_warning=True):
    """
    Perform k-fold Cross-Validation (CV) to evaluate the 
    performance of a Gaussian Process model.

    Parameters
    ----------
    xs : array-like, shape (n_samples, n_features)
        Training data.
    ys : array-like, shape (n_samples,)
        Target values.
    kernel : sklearn.gaussian_process.kernels.Kernel
        The kernel specifying the covariance function of the GP.
    n_tuning : int
        Number of restarts of the optimizer for 
        finding the kernel's hyper-parameters 
    optimizer : Callable
        Optimizer to be used in fitting GP.
    n_splits : int, optional (default=5)
        Number of folds for cross-validation.
    ignore_warning : bool, optional (default=True)
        If True, Convergence Warning of GP Regressor will be ignored.

    Returns
    -------
    performance : float
        Performance measure (Mean Squared Error) using k-fold CV.
    """

    kf = KFold(n_splits=n_splits)
    y_true = []
    y_pred = []

    for train_index, test_index in kf.split(xs):
        xs_train, xs_test = xs[train_index], xs[test_index]
        ys_train, ys_test = ys[train_index], ys[test_index]

        gp_temp = GaussianProcessRegressor(kernel=kernel, 
                                           n_restarts_optimizer=n_tuning, 
                                           optimizer=optimizer)

        if ignore_warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp_temp.fit(xs_train, ys_train)
                warnings.simplefilter("default", ConvergenceWarning)
        else:
            gp_temp.fit(xs_train, ys_train)

        y_pred.extend(gp_temp.predict(xs_test))
        y_true.extend(ys_test)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    performance = r2_score(y_true, y_pred)

    return performance

class IterativeGPFitting:
    """
    Class to perform iterative Gaussian Process (GP) fitting, 
    incresaing the number of samples by `n_samples` each time
    until a performance criterion on 
    Cross Validation r2 score is met.

    Parameters
    ----------
    f : callable
        Function to evaluate the samples.
    container : Container
        Container object to draw samples from 
        and track evaluations.
    kernel : sklearn.gaussian_process.kernels.Kernel
        the kernel used to fit GP
    n_samples : int
        increment of number of samples each time
    max_iter : int, optional (default=5)
        Maximum number of iterations for the iterative process.
        To control number of samples, 
        recommended not exceed 10
    performance_threshold : float, optional (default=0.8)
        Performance threshold for the r2 score to 
        stop the iterative process.
    n_tuning : int
        Number of restarts for the GP optimizer.
    max_iter_optimizer : int
        Maximum number of iterations for the optimizer.
    factr : float
        Tolerance for the optimizer.
    """
    def __init__(self, f: Callable, container: Container, 
                 kernel: Kernel,
                 n_samples: int, n_tuning: int, 
                 max_iter_optimizer: int, factr: float, 
                 performance_threshold: float=0.8, 
                 max_iter: int=5):
        self.f = f
        self.container = container
        self.n_samples = n_samples
        self.kernel = kernel
        self.max_iter = max_iter
        self.n_tuning = n_tuning
        self.factr = factr
        self.performance_threshold = performance_threshold

        def custom_optimizer(obj_func, initial_theta, bounds):
            result = fmin_l_bfgs_b(obj_func, initial_theta, 
                                bounds=bounds, 
                                maxiter=max_iter_optimizer, 
                                factr=factr)
            return result[0], result[1]
    
        self.optimizer = custom_optimizer

    def fit(self) -> GaussianProcessRegressor:
        """
        Returns
        ------
        gp : sklearn.gaussian_process.GaussianProcessRegressor
            the ultimate fitted GPRegressor
        """
        n = self.n_samples
        iteration = 0
        while iteration < self.max_iter:
            # Redraw uniform samples from the container
            xs = self.container.rvs(n)
            ys = self.f(xs)

            performance = gp_kfoldCV(xs, ys, self.kernel, 
                                     self.n_tuning, self.optimizer)

            if performance >= self.performance_threshold:
                break

            # increase the number of samples for the next iteration
            n += self.n_samples  
            iteration += 1

        gp = fit_GP(xs, ys, self.kernel, self.n_tuning, self.optimizer)

        return gp



def default_criterion(container: Container) -> bool:
    return True

def GP_diagnosis(gp: GaussianProcessRegressor, container: Container, 
                 criterion: Callable[[Container], 
                                     bool]=default_criterion) -> None:
    """
    Check the performance of a Gaussian Process (GP) model.
    
    Parameters
    ----------
    gp : GaussianProcessRegressor
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

def rbf_Integration(gp: GaussianProcessRegressor, container: Container, 
                    return_std: bool) -> Union[float, tuple]:
    """
    Estimate the integral of the RBF kernel over 
    a given container and set of points.

    Parameters
    ----------
    gp : GaussianProcessRegressor
        The fitted Gaussian Process model.
    container : Container
        The container object that holds the boundaries.
    return_std : bool
        When True, return the standard deviation of GP estimation

    Returns
    -------
    float or tuple
        float is the estimated integral value, 
        tuple has length 2, the second value is 
          integral evaluation std.
    """
    if not isinstance(gp.kernel_, RBF):
        raise TypeError('this method only works for RBF kernels')

    ### Extract necessary information
    l = gp.kernel_.length_scale
    xs = gp.X_train_
    ys = gp.y_train_

    # Container boundaries
    b = container.maxs   # right boarder
    a = container.mins   # left boarder

    # Estimate the integral
    K_inv_y = gp.alpha_.reshape(-1)

    erf_b = erf((b - xs) / (np.sqrt(2) * l))
    erf_a = erf((a - xs) / (np.sqrt(2) * l))
    k_tilde = (erf_b - erf_a).prod(axis=1) * (
        (l * np.sqrt(np.pi / 2)) ** container.D)

    integral = np.dot(K_inv_y, k_tilde)

    if return_std:
        # mean of kernel on the container
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

        if var_post < 0:
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