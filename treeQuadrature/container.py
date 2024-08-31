import numpy as np
import warnings
from typing import Optional, List


class Container:
    """
    Represents a hyper-rectangle in n-dim space
    with finite volume
    and the samples it holds.

    Attributes
    ----------
    mins, maxs : numpy.ndarray of shape (D,)
        the low and high boundaries of the
        hyper-rectangle containers
    volume : float
        volume of the container
    is_finite : bool
        indicator of whether the container has finite volume
    midpoint : numpy.ndarray of shape (D,)
        the midpoint of container

    Properties
    ----------
    N : int
        number of samples (also number of evaluations)
    X, y : numpy.ndarray
        samples and evaluations
        X is numpy.ndarray of shape (N, D)
        y is numpy.ndarray of shape (N, 1)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 mins=None, maxs=None):
        """
        Parameters
        ----------
        X : numpy.ndarray of shape (N, D)
            each row is a sample
        y : numpy.ndarray of shape (N, 1) or (N,)
            the function value at each sample
        mins, maxs : int or float or list or numpy.ndarray of
        shape (D,), optional
            the low and high boundaries of the hyper-rectangle
            could be +- np.inf
        """

        if np.any(mins == np.inf):
            raise ValueError(f"mins cannot have np.inf, got {mins}")
        if np.any(maxs == -np.inf):
            raise ValueError(f"maxs cannot have -np.inf, got {maxs}")

        X, y = self._handle_X_y(X, y)

        # create empty lists to store samples and evaluations
        self._X = []
        self._y = []

        self.D = X.shape[1]

        # if mins (maxs) are None, create unbounded container
        self.mins = self._handle_min_max_bounds(mins, -np.inf)
        self.maxs = self._handle_min_max_bounds(maxs, np.inf)

        self.volume = np.prod(self.maxs - self.mins)
        self.is_finite = not np.isinf(self.volume)
        if self.is_finite:
            self.midpoint = (self.mins + self.maxs) / 2
        else:
            self.midpoint = np.nan

        X_filtered, y_filtered = self.filter_points(X, y)
        self.add(X_filtered, y_filtered)

    def _handle_min_max_bounds(self, bounds,
                               default_value) -> np.ndarray:
        """Handle different types of min/max bounds."""
        if isinstance(bounds, (int, float)):
            return np.array([bounds] * self.D)
        elif isinstance(bounds, list):
            # dimensionality checks
            if len(bounds) != self.D:
                raise ValueError("bound should have length D")
            return np.array(bounds)
        elif isinstance(bounds, np.ndarray):
            if bounds.shape[0] != self.D:
                raise ValueError("bound should have length D")
            return bounds
        else:
            return np.array([default_value] * self.D)

    def _handle_X_y(self, X: np.ndarray, y: np.ndarray):
        """Handle the input X and y arrays"""
        # basic checks
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-dimensional array, got {X.ndim} dimensions"
            )
        if (y.ndim != 2 or y.shape[1] != 1) and y.ndim != 1:
            raise ValueError(
                "y must be a 1-dimensional array, or 2-dimensional array"
                f" with shape (N, 1), got shape {y.shape}"
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The number of samples in X and y must be the same, "
                f"got {X.shape[0]} and {y.shape[0]}"
            )

        if y.ndim == 1:
            ret_y = y.reshape(-1, 1)
        else:
            ret_y = y.copy()

        return X, ret_y

    def filter_points(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        return_bool: bool = False,
        warning: bool = False,
    ):
        """
        Check whether all the points X are in the container
        and return a numpy.ndarray with those in the container. \n
        Throw a warning if any point is not
        in the container.

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            An array of points to check.
        y : np.ndarray of shape (N, ), optional
            corresponding values
        return_bool : bool, optional
            if true, return a bool inside of filtered samples
            Defaults to False
        warning : bool, optional
            if true, throw a warning if some points are outside
            Defaults to False

        Returns
        -------
        np.ndarray, np.ndarray or bool
            Two arrays: one of the points that are within the container,
            and another of the corresponding y values.
            bool: indicates whether all points are inside the container

        """

        in_bounds = np.all((X >= self.mins) &
                           (X <= self.maxs), axis=1)
        if not np.all(in_bounds):
            inside = False
            if warning:
                warnings.warn(
                    "Some points are out of the container bounds: "
                    f"indices {np.where(~in_bounds)[0]}"
                )
        else:
            inside = True

        if return_bool:
            return inside
        else:
            return X[in_bounds] if (
                y is None) else X[in_bounds], y[in_bounds]

    def add(self, new_X: np.ndarray, new_y: np.ndarray):
        """
        Parameters
        ----------
        new_X : numpy.ndarray of shape (N, D)
            each row is a new sample
        new_y : numpy.ndarray of shape (N, 1) or (N,)
            the function value at each sample
        """
        # rearrange the shapes
        new_X, new_y = self._handle_X_y(new_X, new_y)

        new_X, new_y = self.filter_points(new_X, new_y)

        self._X.append(new_X)
        self._y.append(new_y)

    @property
    def N(self) -> int:
        return sum(x.shape[0] for x in self._X)

    @property
    def X(self) -> np.ndarray:
        return np.vstack(self._X)

    @property
    def y(self) -> np.ndarray:
        return np.vstack(self._y)

    def rvs(self, n: int) -> np.ndarray:
        """
        Draw uniformly random samples from the container

        Attribute
        ---------
        n : int
            number of samples

        Return
        ------
        rs : numpy.ndarray of shape (n, D)
            each row is a sample
        """

        rs = np.empty((n, self.D))

        for d in range(self.D):
            if np.isinf(self.mins[d]) and np.isinf(self.maxs[d]):
                # Both bounds are infinite:
                # sample from a standard normal distribution
                rs[:, d] = np.random.normal(size=n)
            elif np.isinf(self.mins[d]):
                # Lower bound is infinite:
                # sample from a exponential distribution
                rs[:, d] = self.maxs[d] - \
                    np.random.exponential(scale=1.0, size=n)
            elif np.isinf(self.maxs[d]):
                # Upper bound is infinite:
                # sample from a exponential distribution
                rs[:, d] = self.mins[d] + \
                    np.random.exponential(scale=1.0, size=n)
            else:
                # Both bounds are finite:
                # sample uniformly between the bounds
                rs[:, d] = np.random.uniform(
                    low=self.mins[d], high=self.maxs[d], size=n
                )

        return rs

    def split(self, split_dimension: int,
              split_value: float) -> List["Container"]:
        """
        Divide perpendicular to an axis

        Parameters
        ----------
        split_dimension : int
            the axis to split along
        split_value : float
            the value to split at

        Return
        ------
        list of two sub-containers
        """

        # Partition samples
        idxs = self.X[:, split_dimension] <= split_value

        lX = self.X[idxs]
        ly = self.y[idxs]

        rX = self.X[np.logical_not(idxs)]
        ry = self.y[np.logical_not(idxs)]

        # Calculate the new space partitions
        left_mins = np.array(self.mins, copy=True)
        left_maxs = np.array(self.maxs, copy=True)
        left_maxs[split_dimension] = split_value

        right_mins = np.array(self.mins, copy=True)
        right_maxs = np.array(self.maxs, copy=True)
        right_mins[split_dimension] = split_value

        # Create new Container instances
        left_container = Container(lX, ly, mins=left_mins, maxs=left_maxs)
        right_container = Container(rX, ry, mins=right_mins, maxs=right_maxs)

        return [left_container, right_container]
