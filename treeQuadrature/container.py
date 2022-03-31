import numpy as np


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
    Represents a finite region of n-dim space
    and the samples it holds.
    '''

    def __init__(self, X, y, mins=None, maxs=None):
        assert X.ndim == 2
        assert y.ndim == 2
        assert X.shape[0] == y.shape[0]
        assert y.shape[1] == 1

        self.D = X.shape[1]

        # Compute container properties
        self.mins = np.array(mins) if mins is not None else np.array(
            [-np.inf] * self.D)
        self.maxs = np.array(maxs) if maxs is not None else np.array(
            [np.inf] * self.D)
        self.volume = np.product(self.maxs - self.mins)
        self.is_finite = not np.isinf(self.volume)
        self.midpoint = (
            self.mins + self.maxs) / 2 if self.is_finite else np.nan

        assert self.mins.shape[0] == self.D
        assert self.maxs.shape[0] == self.D

        self._X = ArrayList(D=self.D)
        self._y = ArrayList(D=1)

        self.add(X, y)

    def add(self, new_X, new_y):
        assert new_X.ndim == 2
        assert new_y.ndim == 2
        assert new_X.shape[0] == new_y.shape[0]
        assert new_X.shape[1] == self.D
        assert new_y.shape[1] == 1
        assert np.all(new_X >= self.mins), new_X[new_X < self.mins]
        assert np.all(new_X <= self.maxs), new_X[new_X > self.maxs]

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
        rs = np.random.uniform(size=(n, self.D))
        ranges = self.maxs - self.mins
        rs = self.mins + ranges * rs
        return rs

    def split(self, split_dimension, split_value):
        '''Divide perpendicular to an axis'''

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
