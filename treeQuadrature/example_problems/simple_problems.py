from typing import Optional
import numpy as np

from .base_class import Problem
from ..utils import handle_bound


class ProductPeak(Problem):
    def __init__(
        self, D: int, u: Optional[np.ndarray] = None, a: Optional[np.ndarray] = None
    ):
        """
        Initialize a product peak problem with the given parameters.

        Parameters
        ----------
        D : int
            The dimension of the problem.
        u : np.ndarray or None, optional
            The location of the peaks in each dimension. If None, it is set
            to be evenly spaced between 0.2 and 0.8 across the dimensions.
        a : np.ndarray or None, optional
            The sharpness of the peaks in each dimension. If None, it is set
            to an array of ones, meaning equal sharpness in all dimensions.

        Notes
        -----
        `ProductPeak` is a multidimensional integration problem where the
        integrand has peaks at specific locations in each dimension, controlled by
        the vector `u`. The sharpness of these peaks is determined by the vector `a`,
        where larger values of `a` result in sharper peaks, meaning the function values
        drop off more quickly as you move away from the peak location.

        The integration domain is always [0, 1]^D
        """
        super().__init__(D, lows=0.0, highs=1.0)

        self.a = handle_bound(a, D, 1.0)

        if u is None:
            self.u = np.linspace(0.2, 0.8, D)
        else:
            self.u = u

        self.answer = np.prod(
            self.a * (np.arctan(self.a * (1 - self.u)) + np.arctan(self.a * self.u))
        )

    def integrand(self, X) -> np.ndarray:
        """
        Product Peak integrand function.

        Mathematically, the integrand function is defined as:

        .. math::
            f(x) = \frac{1}{\prod_{i=1}^D \left(a_i^{-2} + (x_i - u_i)^2 \right)}

        where:
        - \( x = (x_1, x_2, \dots, x_D) \) is the input vector.
        - \( u = (u_1, u_2, \dots, u_D) \) is a vector representing the peak location.
        - \( a = (a_1, a_2, \dots, a_D) \) is a vector controlling the
            sharpness of the peak along each dimension.
        - \( D \) is the dimension of the input space.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        X = self.handle_input(X)
        f = [1 / np.prod(self.a ** (-2.0) + (x - self.u) ** 2) for x in X]
        return np.array(f).reshape(-1, 1)

    def __str__(self) -> str:
        return f"ProductPeak(D={self.D})"


class CornerPeak(Problem):
    def __init__(self, D: int, a: Optional[np.ndarray] = None):
        """
        Initialize a corner peak problem with the given parameters.

        Parameters
        ----------
        D : int
            The dimension of the problem.
        a : np.ndarray or None, optional
            A vector controlling the "steepness" of the corner peak in each dimension.
            If None, it defaults to an array of ones,
            meaning equal steepness in all dimensions.

        Notes
        -----
        `CornerPeak` is a multidimensional integration problem where the
        integrand has a peak located near the corner of the domain [0, 1]^D.
        The steepness of the peak is controlled by the vector `a`.
        Larger values of `a` result in a steeper
        peak, meaning the function value drops off more quickly as you move away from
        the corner where the peak is located.
        """
        super().__init__(D, lows=0.0, highs=1.0)
        self.a = handle_bound(a, D, 1.0)

        self.answer = self.compute_answer(a0=1, a=self.a)

    def compute_answer(self, a0, a):
        if len(a) == 1:
            return 1 / (a0 * (a0 + a[0]))
        else:
            term1 = self.compute_answer(a0=a0, a=a[:-1])
            term2 = self.compute_answer(a0=a0 + a[-1], a=a[:-1])
            return (term1 - term2) / (a[-1] * len(a))

    def integrand(self, X) -> np.ndarray:
        """
        Corner Peak integrand function.

        .. math::
        f(x) = \left(1 + \sum_{i=1}^{D} a_i \cdot x_i\right)^{-(D+1)}

        where:
        - :math:`x` is a vector of input values,
        - :math:`a` is a coefficient vector,
        - :math:`D` is the dimensionality of the input space.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector of shape (N, D),
            where N is the number of samples, and D is the dimensionality.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        X = self.handle_input(X)
        dotprods = np.array([np.dot(x, self.a) for x in X])
        # prevent raising 0 to a power
        epsilon = 1e-13
        dotprods = np.clip(dotprods, -1 + epsilon, None)
        f = (1 + dotprods) ** (-self.D - 1)
        return np.array(f).reshape(-1, 1)

    def __str__(self) -> str:
        return f"CornerPeak(D={self.D})"


class C0(Problem):
    def __init__(
        self, D: int, u: Optional[np.ndarray] = None, a: Optional[np.ndarray] = None
    ):
        """
        Initialize the C0 problem with the given parameters.

        Parameters
        ----------
        D : int
            The dimensionality of the problem.
        u : np.ndarray or None, optional
            The location parameter that shifts the peak of the function.
            If None, defaults to a linearly spaced array between 0.2 and 0.8.
        a : np.ndarray or None, optional
            A vector controlling the rate of decay of the function.
            If None, defaults to an array of ones, meaning equal decay in all dimensions.

        Notes
        -----
        `C0` is a multidimensional integration problem where the
        integrand is a smooth, exponentially decaying function. The function
        is centered around the vector `u`, and its rate of decay is controlled
        by the vector `a`.
        """
        super().__init__(D, lows=0.0, highs=1.0)

        self.a = handle_bound(a, D, 1.0)

        if u is None:
            self.u = np.linspace(0.2, 0.8, D)
        else:
            self.u = u

        self.answer = np.prod(
            (2 - np.exp(-self.a * self.u) - np.exp(-self.a * (1 - self.u))) / self.a
        )

    def integrand(self, X) -> np.ndarray:
        """
        C0 integrand function.

        .. math::
        f(X) = \exp\left(-\sum_{i=1}^{D} a_i \cdot |x_i - u_i|\right)

        where:

        - \(X = (x_1, x_2, \dots, x_D)\) is the input vector,
        - \(a = (a_1, a_2, \dots, a_D)\) is a vector of coefficients,
        - \(u = (u_1, u_2, \dots, u_D)\) is the vector of center points.

        The function represents an exponential decay based on the
        sum of weighted absolute differences between each component
        of the input vector and the corresponding component of the center vector.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        X = self.handle_input(X)
        f = [np.exp(-np.sum(self.a * np.abs(x - self.u))) for x in X]
        return np.array(f).reshape(-1, 1)

    def __str__(self) -> str:
        return f"C0function(D={self.D})"


class Discontinuous(Problem):
    """
    A problem with a discontinuous integrand function.

    The integrand is designed to be discontinuous at specified points `u1` and `u2`.
    The function value drops to zero beyond these thresholds, creating a discontinuity
    in the domain.

    Attributes
    ----------
    D : int
        Dimensionality of the problem.
    a : np.ndarray
        one dimensional array,
        Coefficient vector for the exponential function.
    u1 : float
        Threshold for discontinuity in the first dimension.
    u2 : float
        Threshold for discontinuity in the second dimension (if D > 1).
    answer : float
        The analytical solution of the integral for comparison.
    """

    def __init__(self, D, a=None):
        super().__init__(D, lows=0.0, highs=1.0)

        self.a = handle_bound(a, D, 1.0)

        self.u1 = 0.3
        self.u2 = 0.5

        if D == 1:
            self.answer = (np.exp(self.a[0] * self.u1) - 1) / self.a[0]
        else:
            term1 = (np.exp(self.a[0] * self.u1) - 1) / self.a[0]
            term2 = (np.exp(self.a[0] * self.u2) - 1) / self.a[1]
            term3 = np.prod((np.exp(self.a[2:]) - 1) / self.a[2:])
            self.answer = term1 * term2 * term3

    def integrand(self, X) -> np.ndarray:
        """
        Discontinuous integrand function.

        The Discontinuous integrand function is defined as:
    
        .. math::
            f(X) = 
            \begin{cases} 
            \exp\left(\sum_{i=1}^{D} a_i \cdot x_i\right) & \text{if } X \text{ lies in the specified region}, \\
            0 & \text{otherwise}
            \end{cases}
        
        where:
        
        - \(X = (x_1, x_2, \dots, x_D)\) is the input vector,
        - \(a = (a_1, a_2, \dots, a_D)\) is a vector of coefficients,
        - \(u_1, u_2\) are predefined thresholds.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        X = self.handle_input(X)
        dotprods = np.array([np.dot(x, self.a) for x in X])
        if self.D == 1:
            f = np.array([np.where(x[0] > self.u1, 0, 1) for x in X]) * np.exp(dotprods)
        else:
            f = np.array(
                [np.where((x[0] > self.u1 or x[1] > self.u2), 0, 1) for x in X]
            ) * np.exp(dotprods)

        return f.reshape(-1, 1)

    def __str__(self) -> str:
        return f"Discontinuous(D={self.D})"

    def handle_input(self, xs) -> np.ndarray:
        """
        Check the shape of xs and
        change xs to the correct shape (N, D)

        Parameter
        --------
        xs : numpy.ndarray
            the array to be handled

        Return
        numpy.ndarray
            the handled array
        """
        if isinstance(xs, list):
            xs = np.array(xs)
        elif not isinstance(xs, np.ndarray):
            raise TypeError("xs must be either a list or numpy.ndarray")

        if xs.ndim == 2 and xs.shape[1] == self.D:
            return xs
        elif xs.ndim == 1 and xs.shape[0] == self.D:  # array with one sample
            return xs.reshape(1, -1)
        else:
            raise ValueError(
                "xs must be either two dimensional array of shape (N, D)"
                "or one dimensional array of shape (D,)"
            )


class Pyramid(Problem):
    def __init__(self, D):
        super().__init__(D, lows=-1.0, highs=1.0)
        self.answer = (2**self.D) / (self.D + 1)

    def integrand(self, X) -> np.ndarray:
        """
        Pyramid integrand function.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray
            1-dimensional array of the same length as X.
        """

        ys = 1 - np.max(np.abs(X), axis=1)
        return ys.reshape(-1, 1)

    def __str__(self) -> str:
        return f"Pyramid(D={self.D})"


class Quadratic(Problem):
    def __init__(self, D):
        """
        Quadratic function sum x_i^2
        on [-1.0, 1.0]^D

        Parameters
        ----------
        D : int
            dimension
        """
        super().__init__(D, lows=-1.0, highs=1.0)
        self.answer = self.exact_integral(self.lows, self.highs)

    def integrand(self, X) -> np.ndarray:
        """
        Quadratic integrand function for the sum of squares.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        xs = self.handle_input(X)
        ys = np.sum(xs**2, axis=1)
        return ys.reshape(-1, 1)

    def exact_integral(self, mins, maxs):
        """
        Calculate the exact integral from mins to maxs for the sum of squares polynomial.

        Parameters
        ----------
        mins : numpy.ndarray
            Lower bounds of the integration.

        maxs : numpy.ndarray
            Upper bounds of the integration.

        Returns
        -------
        float
            The value of the integral.
        """
        D = len(mins)
        integral_sum = 0

        for i in range(D):
            term = (maxs[i] ** 3 - mins[i] ** 3) / 3
            product = np.prod([maxs[j] - mins[j] for j in range(D) if j != i])
            integral_sum += term * product

        return integral_sum

    def __str__(self) -> str:
        return f"Quadratic(D={self.D})"


class ExponentialProduct(Problem):
    def __init__(self, D):
        """
        Exponential product
        on [-1.0, 1.0]^D

        Parameters
        ----------
        D : int
            dimension
        """
        super().__init__(D, lows=-1.0, highs=1.0)
        self.answer = self.exact_integral(self.lows, self.highs)

    def integrand(self, X) -> np.ndarray:
        """
        Exponential product integrand function.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray
            1-dimensional array of the same length as X.
        """
        xs = self.handle_input(X)
        ys = np.prod(np.exp(xs), axis=1)
        return ys.reshape(-1, 1)

    def exact_integral(self, mins, maxs):
        """
        Calculate the exact integral from mins to maxs for the product of exponentials.

        Parameters
        ----------
        mins : numpy.ndarray
            Lower bounds of the integration.

        maxs : numpy.ndarray
            Upper bounds of the integration.

        Returns
        -------
        float
            The value of the integral.
        """
        return np.prod([np.exp(maxs[i]) - np.exp(mins[i]) for i in range(len(mins))])

    def __str__(self) -> str:
        return f"ExponentialProduct(D={self.D})"
