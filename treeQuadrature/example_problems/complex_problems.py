from typing import Optional
import numpy as np

from .base_class import Problem


class Ripple(Problem):
    """
    A problem class representing a ripple-like integrand function
    for numerical integration or other mathematical evaluations.

    The ripple function has an oscillatory behavior controlled by the
    parameter `a` and is defined over a D-dimensional space with bounds
    from -10 to 10 in each dimension.

    Attributes
    ----------
    D : int
        The dimensionality of the problem.
    a : float, optional
        A parameter controlling the frequency of the ripple in the
        integrand function. Default is 3.
    answer : float
        The analytic solution to the integral of the function over the
        entire domain.
    """

    def __init__(self, D, a=3):
        super().__init__(D, lows=-10.0, highs=10.0)
        self.a = a

        # Compute the answer
        temp = (1 / (1 + 16 * self.a**2)) ** \
            (self.D / 4) * np.cos(
            self.D * np.arccos(
                np.sqrt((1 / (1 + 16 * self.a**2)))
            ) / 2
        )
        self.answer = np.sqrt(2 * np.pi) ** \
            self.D * (1 + temp) / 2

    def integrand(self, X) -> np.ndarray:
        """
        Ripple integrand function.

        Mathematically, the integrand function is defined as:

        .. math::
            f(x) = \\exp\\left(-\\frac{\\|x\\|^2}{2}\\right)
                \\cdot \\cos^2\\left(a \\|x\\|^2\\right)

        where:
        - \\( x = (x_1, x_2, \\dots, x_D) \\) is the input vector.
        - \\( \\|x\\| \\) is the Euclidean norm of the vector \\( x \\),
           given by \\( \\|x\\| = \\sqrt{\\sum_{i=1}^D x_i^2} \\).
        - \\( a \\) is a scalar that controls the frequency of the ripples.

        The function exhibits a combination of exponential decay and
        oscillatory ripple patterns as \\( \\|x\\| \\) varies.

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
        norms = np.linalg.norm(X, axis=1)
        f = np.exp(-(norms**2) / 2) * \
            np.cos(self.a * norms**2) ** 2
        return np.array(f).reshape(-1, 1)

    def __str__(self) -> str:
        return f"Ripple(D={self.D})"


class Oscillatory(Problem):
    """
    A problem class representing an oscillatory integrand function
    for numerical integration or other mathematical evaluations.

    The oscillatory function depends on a
    frequency vector `a` and a phase shift parameter `u`.
    The integration bounds are [0, 1] in each dimension.

    Attributes
    ----------
    D : int
        The dimensionality of the problem.
    a : numpy.ndarray
        The frequency vector of the oscillatory function.
    u : float
        The phase shift parameter of the oscillatory function.
    answer : float
        The analytic solution to the integral of the function over the
        entire domain.
    """

    def __init__(self, D: int, u: int = 0,
                 a: Optional[np.ndarray] = None):
        """
        Initialise the Oscillatory Problem with the given dimensionality,
        frequency vector, and phase shift parameter.

        Parameters
        ----------
        D : int
            The dimensionality of the problem.
        u : float, optional
            The phase shift parameter of the oscillatory function.
            Default is 0.
        a : np.ndarray or None, optional
            The frequency vector that controls the
            frequency of oscillation in each dimension. \n
            If None, it is set to 5 / np.linspace(1, D, D),
            which generates a decreasing frequency across dimensions.

        Notes
        -----
        The vector `a` affects the oscillatory behavior of the integrand.
        Specifically, each element of `a` controls
        the frequency of oscillation
        along the corresponding dimension. \n
        Larger values in `a` result in higher
        frequencies, causing more rapid oscillations
        along that dimension.

        The integration domain is always [0, 1]^D
        """
        super().__init__(D, lows=0.0, highs=1.0)
        if a is None:
            self.a = 5 / np.linspace(1, D, D)
        else:
            self.a = a
        self.u = u
        self.answer = self.compute_answer(a=self.a,
                                          u=self.u)

    def compute_answer(self, a, u):
        if len(a) > 1:
            term1 = self.compute_answer(a[:-1],
                                        u - 1 / 4 + a[-1] / (2 * np.pi))
            term2 = self.compute_answer(a[:-1],
                                        u - 1 / 4)
            return (term1 - term2) / a[-1]
        else:
            return (np.sin(2 * np.pi * u + a[0]) -
                    np.sin(2 * np.pi * u)) / a[0]

    def integrand(self, X) -> np.ndarray:
        """
        Oscillatory integrand function.

        Mathematically, the integrand function is defined as:

        .. math::
            f(x) = \\cos\\left(2\\pi u + \\sum_{i=1}^D a_i x_i\\right)

        where:
        - \\( x = (x_1, x_2, \\dots, x_D) \\) is the input vector.
        - \\( u \\) is a scalar that shifts the phase of the cosine function.
        - \\( a = (a_1, a_2, \\dots, a_D) \\) is a vector
            that controls the frequency of
            oscillation along each dimension.
        - \\( D \\) is the dimension of the input space.

        The function exhibits oscillatory behavior
        as \\( x \\) varies, with the
        frequency of oscillation determined by the
        values in the vector \\( a \\).

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
        f = np.cos(2 * np.pi * self.u + dotprods)
        return np.array(f).reshape(-1, 1)

    def __str__(self) -> str:
        return f"Oscillatory(D={self.D})"
