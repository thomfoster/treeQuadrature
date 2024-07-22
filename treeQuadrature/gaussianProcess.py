import numpy as np
import warnings
from typing import Callable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import Kernel, RBF
from sklearn.metrics import r2_score, mean_squared_error
from scipy.special import erf
from scipy.optimize import fmin_l_bfgs_b
from scipy.integrate import quad

from .visualisation import plotGP
from .container import Container


def fit_GP(xs, ys, kernel: Kernel, n_tuning: int, max_iter: int, factr: float, 
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
    max_iter : int
        Maximum number of iterations for the optimizer.
    factr : float
        The tolerance of the optimizer.
    ignore_warning : bool
        if True, Convergence Warning of GP Regressor will be ignored
    
    Returns
    -------
    gp : GaussianProcessRegressor
        Fitted Gaussian Process model.
    """
    # allow control of tolerance and maximum iterations
    def custom_optimizer(obj_func, initial_theta, bounds):
        result = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=max_iter, factr=factr)
        return result[0], result[1]
    
    gp = GaussianProcessRegressor(kernel=kernel, 
                                  n_restarts_optimizer=n_tuning, optimizer=custom_optimizer)

    ### Fit the GP model without convergence warnings
    if ignore_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(xs, ys)
            warnings.simplefilter("default", ConvergenceWarning)
    else:
        gp.fit(xs, ys)

    return gp


def default_criterion(container: Container) -> bool:
    return True

def GP_diagnosis(gp: GaussianProcessRegressor, xs, ys, container: Container, 
                 criterion: Callable[[Container], bool]=default_criterion) -> None:
    """
    Check the performance of a Gaussian Process (GP) model.
    
    Parameters
    ----------
    gp : GaussianProcessRegressor
        The fitted Gaussian Process model.
    xs : array-like, shape (n_samples, n_features)
        Test data.
    ys : array-like, shape (n_samples,)
        True values for the test data.
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

def rbf_Integration(gp: GaussianProcessRegressor, container: Container, return_std: bool):
    """
    Estimate the integral of the RBF kernel over a given container and set of points.

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
                GP_diagnosis(gp, xs, ys, container)
                print('----------------------------')
            var_post = 0

    if return_std:
        return (integral, np.sqrt(var_post))
    else:
        return integral