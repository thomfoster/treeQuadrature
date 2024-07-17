from typing import Dict, Any
from sklearn.gaussian_process.kernels import RBF

from ..gaussianProcess import fit_GP, GP_diagnosis, rbf_Integration
from .containerIntegral import ContainerIntegral

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
            'n_samples': 15,
            'n_tuning': 10,
            'factr': 1e7,
            'max_iter': 1e4,
            'check_GP': False,
            'return_std': False
        }
        self.options.update(kwargs)

        RbfIntegral._validate_options(self.options)

        # set options correspondingly
        for key, value in self.options.items():
            setattr(self, key, value)

    @staticmethod
    def _validate_options(options):
        if not isinstance(options['length'], int):
            raise TypeError('length must be an int')
        if not isinstance(options['n_samples'], int):
            raise TypeError('n_samples must be an int')
        if not isinstance(options['n_tuning'], int):
            raise TypeError('n_tuning must be an int')
        if not isinstance(options['factr'], (int, float)):
            raise TypeError('factr must be an int or float')
        if not isinstance(options['max_iter'], (int, float)):
            raise TypeError('max_iter must be an int or float')
        if not isinstance(options['check_GP'], bool):
            raise TypeError('check_GP must be a bool')
        if not isinstance(options['return_std'], bool):
            raise TypeError('return_std must be a bool')

    def containerIntegral(self, container, f, **kwargs: Any):
        options = self.options.copy()
        options.update(kwargs)

        RbfIntegral._validate_options(options)

        # Set instance variables based on self.options
        for key, value in self.options.items():
            setattr(self, key, value)

        # redraw uniform samples from the container
        # TODO -keep adjusting the number of samples
        # if container.volume > 0.05:
        #     xs = container.rvs(self.n_samples * 8)
        # elif container.volume > 0.01:
        #     xs = container.rvs(self.n_samples * 2)
        # elif container.volume > 0.025:
        #     xs = container.rvs(self.n_samples * 4)
        # else:
        #     xs = container.rvs(self.n_samples)
        xs = container.rvs(self.n_samples)
        ys = f(xs)
        container.add(xs, ys)  # for tracking num function evaluations

        # fit GP using RBF kernel
        kernel = RBF(self.length, (self.length*1e-2, self.length*1e2))
        gp = fit_GP(xs, ys, kernel, self.n_tuning, self.max_iter, self.factr)

        ### GP diagnosis
        if self.check_GP:
            # TODO - decide where to plot
            GP_diagnosis(gp, xs, ys, container, 
                    criterion = lambda container : container.volume > 0.05)
        
        return rbf_Integration(gp, container, xs, self.return_std)