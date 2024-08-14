import warnings
from scipy.special import erf, binom
from scipy.integrate import quad
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from typing import Callable, Optional, Tuple

from .gaussianProcess import GPFit, IterativeGPFitting
from ..container import Container
from .kernels import Polynomial



def rbf_mean_post(gp: GPFit, container: Container, gp_results: dict):
    """
    calculate the posterior mean of integral using RBF kernel

    Returns
    -------
    float, numpy.ndarray
        the posterior mean fo integral and partial mean of kernel
    """

    ### Extract necessary information
    l = gp.kernel_.length_scale
    
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

    l = gp.kernel_.length_scale

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


def poly_post(gp: GPFit, container: Container, gp_results: dict, 
              return_std: bool,
              d: int, c: float, 
              jitter: float = 1e-8, 
              threshold: float = 1e10) -> Tuple[float, float]:
    """
    Calculate the posterior mean and variance of the integral estimate using a polynomial kernel.

    Parameters
    ----------
    gp : GaussianProcessRegressor
        The fitted Gaussian Process model.
    container : Container
        The container object that holds the boundaries.
    gp_results : dict
        Results from gp.fit() necessary for performing the kernel integral.
    d : int
        The degree of the polynomial kernel.
    c : float
        The coefficient in the polynomial kernel.
    jitter : float, optional
        Small value added to the diagonal of the kernel matrix to ensure stability.
        Defaults to 1e-8.
    threshold : float, optional
        Condition number threshold for the kernel matrix.
        Defaults to 1e10.

    Returns
    -------
    Tuple[float, float]
        The posterior mean and variance of the integral estimate.
    """

    xs = gp.X_train_

    # Container boundaries
    b = container.maxs   # right boundary
    a = container.mins   # left boundary

    # Precompute K_inv_y for posterior mean
    K_inv_y = gp.alpha_.reshape(-1)

    k_tilde = np.zeros(xs.shape[0])

    # Compute the kernel partial mean (k_tilde) and mean (y_mean) integral
    for j in range(1, d+1):
        coeff = binom(d, j) * c**(d-j)
        integral = np.prod([
            (b[k]**(j+1) - a[k]**(j+1)) / (j+1)
            for k in range(container.D)
        ])
        k_tilde += coeff * np.prod(xs**j, axis=1) * integral

    try:
        y_mean = gp_results['y_mean']
    except KeyError:
        raise KeyError('Cannot find y_mean in gp_results')
    
    posterior_mean = np.dot(K_inv_y, k_tilde) + y_mean * container.volume

    if return_std:
        # Compute the kernel mean (k_mean) integral for posterior variance
        k_mean = 0
        for j in range(1, d+1):
            coeff = binom(d, j) * c**(d-j)
            integral = np.prod([
                (b[k]**(j+1) - a[k]**(j+1)) / (j+1)
                for k in range(container.D)
            ])
            k_mean += coeff * integral**2

        # Compute K_inv for posterior variance
        K = gp.kernel_(xs)
        if np.linalg.cond(K) > threshold:
            K += np.eye(K.shape[0]) * jitter
        K_inv = np.linalg.inv(K)
        
        posterior_variance = k_mean - np.dot(k_tilde.T, np.dot(K_inv, k_tilde))
    else:
        posterior_variance = None

    return posterior_mean, posterior_variance

def kernel_integration(igp: IterativeGPFitting, container: Container, 
                       gp_results: dict, return_std: bool, 
                    kernel_mean_post: Optional[Callable]=None,
                    kernel_var_post: Optional[Callable]=None,
                    kernel_post: Optional[Callable]=None, 
                    **kernel_params) -> dict:
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
    kernel_params : dict
        Additional parameters specific to the kernel functions.

    Returns
    -------
    dict
        - integral (float) the integral estimate
        - std (float) standard deviation of integral
    """
    gp = igp.gp

    if isinstance(gp.kernel_, RBF):   # RBF kernel        
        integral, k_tilde = rbf_mean_post(gp, container, gp_results)

        if return_std:
            var_post = rbf_var_post(container, gp, k_tilde)
    elif kernel_mean_post is not None and (
        kernel_var_post is not None):  
        integral = kernel_mean_post(gp, container, gp_results, **kernel_params)
        if return_std:
            var_post = kernel_var_post(gp, container, **kernel_params)
    elif kernel_post is not None:
        integral, var_post = kernel_post(gp, container, gp_results, return_std, **kernel_params)
    else:
        raise Exception(
            'kernel not RBF, '
            'either kernel_mean_post, kernel_var_post '
            'or kernel_post must be provided'
            )
    
    if not isinstance(integral, float):
        raise ValueError('result of kernel_mean_post must be a float! ')
    if return_std and not isinstance(var_post, float):
        raise ValueError('result of kernel_var_post must be a float! ')
    
    ret = {'integral' : integral}
            
    # value check
    if return_std and var_post < 0:
        if var_post < -1e-2:
            warnings.warn(
                'Warning: variance estimate in a container is negative'
                f' with value : {var_post}'
                '. Will be set to zero.', 
                UserWarning)
        var_post = 0

    if return_std:
        ret['std'] = np.sqrt(var_post)

    return ret