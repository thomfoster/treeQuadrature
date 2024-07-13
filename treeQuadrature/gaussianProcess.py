import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score, mean_squared_error
from scipy.special import erf
from scipy.optimize import fmin_l_bfgs_b
from scipy.integrate import quad

from treeQuadrature.visualisation import plotGP

def fit_GP(xs, ys, kernel, n_tuning, max_iter, factr, ignore_warning=True):
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

def GP_diagnosis(gp, xs, ys, container, criterion):
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
    criterion : function
        A function that takes a container and returns a boolean indicating 
        whether to plot the posterior mean.
    
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
    if xs.shape[1] == 1 and criterion(container):
        plotGP(gp, xs, ys, container.maxs[0], container.mins[0])

def rbf_Integration(gp, container, xs, return_std):
    """
    Estimate the integral of the RBF kernel over a given container and set of points.

    Parameters
    ----------
    gp : GaussianProcessRegressor
        The fitted Gaussian Process model.
    container : Container
        The container object that holds the boundaries.
    xs : array-like, shape (n_samples, n_features)
        The points at which to evaluate the integral.
    return_std : bool
        When True, return the standard deviation of GP estimation

    Returns
    -------
    integral : float
        The estimated integral value.
    """
    # Extract length scaled
    l = gp.kernel_.length_scale

    # Container boundaries
    b = container.maxs   # right boarder
    a = container.mins   # left boarder

    # Estimate the integral
    K_inv_y = gp.alpha_.reshape(-1)

    erf_b = erf((b - xs) / (np.sqrt(2) * l))
    erf_a = erf((a - xs) / (np.sqrt(2) * l))
    k_tilde = (erf_b - erf_a).prod(axis=1) * ((l * np.sqrt(np.pi / 2)) ** container.D)

    integral = np.dot(K_inv_y, k_tilde)

    if return_std:
        # mean of kernel on the container
        def integrand(x_j_prime, a_j, b_j):
            return erf((b_j - x_j_prime) / (np.sqrt(2) * l)) - erf((a_j - x_j_prime) / (np.sqrt(2) * l))
        result = 1
        for j in range(container.D):
            integ_j, _ = quad(integrand, a[j], b[j], args=(a[j], b[j]))
            result *= integ_j
        k_mean = l * np.sqrt(np.pi / 2) * result

        K = gp.kernel_(xs)
        # posterior variance
        var_post = k_mean - np.dot(k_tilde.T, np.dot(K, k_tilde))

        if var_post < 0:
            warnings.warn(f'Warning: variance estimate is negative with value : {var_post}', UserWarning)
            var_post = 0

    if return_std:
        return (integral, np.sqrt(var_post))
    else:
        return integral