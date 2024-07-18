import numpy as np
import warnings


class ArrayList:
    def __init__(self, D):
        self.D = D
        self.data = np.empty(shape=(100, self.D))
        self.capacity = 100
        self.freeSpace = 100  # should always have N + freeSpace = capacity
        self.N = 0

    def add(self, xs):
        n = xs.shape[0]
        if self.freeSpace < n:
            newCapacity = 4 * (self.N + n)
            newData = np.empty(shape=(newCapacity, self.D))
            newData[:self.N] = self.data[:self.N]
            self.data = newData
            self.capacity = newCapacity
            self.freeSpace = self.capacity - self.N

        self.data[self.N:self.N + n] = xs
        self.N += n
        self.freeSpace = self.capacity - self.N

    def printer(self):
        print('D: ', self.D)
        print('capacity: ', self.capacity)
        print('N: ', self.N)
        print('freeSpace: ', self.freeSpace)
        print('contents: ')
        print(self.contents)

    @property
    def contents(self):
        return self.data[:self.N]


class Container:
    '''
    Represents a hyper-rectangle in n-dim space
    with finite volume
    and the samples it holds.

    Attributes
    ----------
    _X, _y : ArrayList
        stores the samples and evaluations efficiently 
    mins, maxs : float or list or numpy.ndarray of shape (D,)
        the low and high boundaries of the 
        only for hyper-rectangle containers
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

    Methods 
    -------
    add(new_x, new_y)
        add new sample points
    rvs(n)
        uniformly randomly draw n samples in this container
        Return : numpy.ndarray of shape (n, D)
    split(split_dimension, split_value)
        split the container into two along split_dimension at split_value
        Return : list of two sub-containers
    '''

    def __init__(self, X, y, mins=None, maxs=None):
        """
        Attributes
        ----------
        X : numpy.ndarray of shape (N, D)
            each row is a sample
        y : numpy.ndarray of shape (N, 1)
            the function value at each sample
        mins, maxs : numpy.ndarray of shape (D,), optional
            the low and high boundaries of the hyper-rectangle
            could be +- np.inf
        """
        # basic checks
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-dimensional array, got {X.ndim} dimensions")
        if y.ndim != 2:
            raise ValueError(f"y must be a 2-dimensional array, got {y.ndim} dimensions")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"The number of samples in X and y must be the same, got {X.shape[0]} and {y.shape[0]}")
        if y.shape[1] != 1:
            raise ValueError(f"y must have shape (N, 1), got shape {y.shape}")
        if np.any(mins == np.inf):
            raise ValueError(f'mins cannot have np.inf, got {mins}')
        if np.any(maxs == -np.inf):
            raise ValueError(f'maxs cannot have -np.inf, got {maxs}')

        self.D = X.shape[1]

        # if mins (maxs) are None, create unbounded container
        self.mins = self._handle_min_max_bounds(mins, -np.inf) 
        self.maxs = self._handle_min_max_bounds(maxs, np.inf)
        # dimensionality checks
        if self.mins.shape[0] != self.D:
            raise ValueError('mins should have length D')
        if self.maxs.shape[0] != self.D:
            raise ValueError('maxs should have length D')

        self.volume = np.prod(self.maxs - self.mins)
        self.is_finite = not np.isinf(self.volume)
        self.midpoint = (
            self.mins + self.maxs) / 2 if self.is_finite else np.nan

        ### add sample points into the hidden ArrayList
        # create empty ArrayList
        self._X = ArrayList(D=self.D)
        self._y = ArrayList(D=1)

        # filter points
        X_filtered, y_filtered = self.filter_points(X, y)
        self.add(X_filtered, y_filtered)

    def _handle_min_max_bounds(self, bounds, default_value):
        """Handle different types of min/max bounds."""
        if isinstance(bounds, (int, float)):
            return np.array([bounds] * self.D)
        elif isinstance(bounds, (list, np.ndarray)):
            return np.array(bounds)
        else:
            return np.array([default_value] * self.D)
        
    def filter_points(self, X, y=None, return_bool=False, warning=False):
        """
        Check whether all the points X are in the container
        and return a numpy.ndarray with those in the container. Throw a warning if any point is not
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

        in_bounds = np.all((X >= self.mins) & (X <= self.maxs), axis=1)
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
            return X[in_bounds] if y is None else X[in_bounds], y[in_bounds]

    def add(self, new_X, new_y):
        if new_X.ndim != 2:
            raise ValueError(f"new_X must be a 2-dimensional array, got {new_X.ndim} dimensions")
        if new_y.ndim != 2:
            raise ValueError(f"new_y must be a 2-dimensional array, got {new_y.ndim} dimensions")
        if new_X.shape[0] != new_y.shape[0]:
            raise ValueError(f"The number of samples in new_X and new_y must be the same, got {new_X.shape[0]} and {new_y.shape[0]}")
        if new_X.shape[1] != self.D:
            raise ValueError(f"new_X must have {self.D} features, got {new_X.shape[1]}")
        if new_y.shape[1] != 1:
            raise ValueError(f"new_y must have shape (N, 1), got shape {new_y.shape}")
        if not np.all(new_X >= self.mins):
            raise ValueError(f"Some values in new_X are below the minimum bounds: {new_X[new_X < self.mins]}")
        if not np.all(new_X <= self.maxs):
            raise ValueError(f"Some values in new_X are above the maximum bounds: {new_X[new_X > self.maxs]}")

        self._X.add(new_X)
        self._y.add(new_y)

    @property
    def N(self):
        return self._X.N

    @property
    def X(self):
        return self._X.contents

    @property
    def y(self):
        return self._y.contents

    def rvs(self, n):
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
                # Both bounds are infinite: sample from a standard normal distribution
                rs[:, d] = np.random.normal(size=n)
            elif np.isinf(self.mins[d]):
                # Lower bound is infinite: sample from a exponential distribution
                rs[:, d] = self.maxs[d] - np.random.exponential(scale=1.0, size=n)
            elif np.isinf(self.maxs[d]):
                # Upper bound is infinite: sample from a exponential distribution 
                rs[:, d] = self.mins[d] + np.random.exponential(scale=1.0, size=n)
            else:
                # Both bounds are finite: sample uniformly between the bounds
                rs[:, d] = np.random.uniform(low=self.mins[d], high=self.maxs[d], size=n)

        return rs

    def split(self, split_dimension, split_value):
        '''
        Divide perpendicular to an axis (only for hyper-rectangles!)
        '''

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
        left_container = Container(
            lX, ly, mins=left_mins, maxs=left_maxs)
        right_container = Container(
            rX, ry, mins=right_mins, maxs=right_maxs)

        return [left_container, right_container]
