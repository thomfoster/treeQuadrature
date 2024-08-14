import numpy as np
from .containerIntegral import ContainerIntegral
from ..container import Container
from scipy.optimize import curve_fit
from scipy.stats import norm


def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


class GaussianCurveIntegral(ContainerIntegral):
    """
    Gaussian Curve Integral:
    use all samples, and estimate integral by a fitted Gaussian Curve

    Attributes
    ----------
    """

    def __init__(self) -> None:
        self.name = 'GaussianCurveIntegral'

    def containerIntegral(self, container: Container, f, **kwargs):
        lb,ub = container.mins, container.maxs
        X,y = container.X, np.array(container.y).T[0]
        total_area = 1.
        for dim in range(container.D):
            mini,maxi = lb[dim], ub[dim]
            x = np.array(X[:, dim])
            wmean = np.dot(x,y)/sum(y)
            sigma = np.sqrt(np.dot(y,(x-wmean)**2)/sum(y))
            try:
                popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), wmean, sigma])
                area = popt[0]*(norm.cdf((maxi-popt[1])/popt[2]) - norm.cdf((mini-popt[1])/popt[2]))*np.sqrt(2*np.pi)*popt[2]
            except:
                print('error')
            total_area *= area

        return total_area