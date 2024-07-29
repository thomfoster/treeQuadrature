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

    def split_by_point(self, point1, point2):
        """
        Split the container based on whether samples in self.X are in the left or right container.

        Parameters
        ----------
        point1, point2 : list of floats of length 2
            The boundary points to split.

        Returns
        -------
        list of two Container instances
            The sub-containers resulting from the best split.
        """
        assert (point1 in self.boundary_points.tolist()) and \
            (point2 in self.boundary_points.tolist()), (
            'point1 and point2 must be one of the boundary points'
        )

        # Ensure point1 and point2 are not neighboring points
        idx1 = self.boundary_points.tolist().index(point1)
        idx2 = self.boundary_points.tolist().index(point2)
        assert (abs(idx1 - idx2) != 1) and \
            (abs(idx1 - idx2) != len(self.boundary_points) - 1), (
            'point1 and point2 must not be neighboring points'
        )

        # Compute the line equation for the best split
        A = point2[1] - point1[1]
        B = point1[0] - point2[0]
        C = point2[0] * point1[1] - point1[0] * point2[1]

        # Partition samples based on the split line
        left_idxs, right_idxs = [], []
        for k, x in enumerate(self.X):
            value = A * x[0] + B * x[1] + C
            if value < 0:
                left_idxs.append(k)
            else:
                right_idxs.append(k)

        left_X, left_y = self.X[left_idxs], self.y[left_idxs]
        right_X, right_y = self.X[right_idxs], self.y[right_idxs]

        # Partition boundary points
        # splitting points are copied into two sub-containers
        left_boundary_points, right_boundary_points = [point1, point2], []
        for point in self.boundary_points:
            value = A * point[0] + B * point[1] + C
            if value < 0:
                left_boundary_points.append(point)
            else:
                right_boundary_points.append(point)

        left_container = Container(left_X, left_y, boundary_points=np.array(left_boundary_points))
        right_container = Container(right_X, right_y, boundary_points=np.array(right_boundary_points))

        return [left_container, right_container]

    def visualize(self, plot_samples=False, ax=None):
        """
        Plot the container.

        This method handles both 2D and 3D containers, plotting either the 
        hyper-rectangle (if defined by `mins` and `maxs`) or the convex hull 
        (if defined by `boundary_points`).

        Parameters
        ----------
        plot_samples : bool
            If True, plot the sample points stored in self.X.
            defaults to False
        ax : Axes, optional
            a canvas to plot the container on
            when not given, a new plot will be generated
        """
        assert self.is_finite, 'cannot visualize infinite container'

        if self.D == 2:
            self._plot_2d(plot_samples, ax)
        elif self.D == 3:
            self._plot_3d(plot_samples, ax)
        else:
            raise ValueError("Plotting is only supported for 2D and 3D containers.")

    def _plot_2d(self, plot_samples, ax):
        if ax is None:
            _, ax = plt.subplots()
            new_plot = True
        else:
            new_plot = False

        if self.is_rectangle:
            self._plot_2d_hyperrectangle(ax)
        else:
            self._plot_2d_convex_hull(ax)

        if plot_samples and self.X is not None:
            self._plot_2d_samples(ax)

        if new_plot:
            ax.set_xlim([self.mins[0] - 1, self.maxs[0] + 1])
            ax.set_ylim([self.mins[1] - 1, self.maxs[1] + 1])
            ax.set_aspect('equal', adjustable='box')

        if new_plot:
            plt.show()

    def _plot_2d_convex_hull(self, ax):
        for simplex in self.hull.simplices:
            ax.plot(self.boundary_points[simplex, 0], self.boundary_points[simplex, 1], 'k-')
        ax.fill(self.boundary_points[self.hull.vertices, 0], 
                self.boundary_points[self.hull.vertices, 1], 
                'k', alpha=0.2)

    def _plot_2d_hyperrectangle(self, ax):
        rect = plt.Rectangle(self.mins, *(self.maxs - self.mins), fill=None, edgecolor='r')
        ax.add_patch(rect)

    def _plot_2d_samples(self, ax):
        ax.scatter(self.X[:, 0], self.X[:, 1], color='red', marker='x', label='Samples')

    def _plot_3d(self, plot_samples, ax):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            new_plot = True
        else: 
            new_plot = False

        if self.is_rectangle:
            self._plot_3d_hyperrectangle(ax)
        else:
            self._plot_3d_convex_hull(ax)
            
        if plot_samples and self.X is not None:
            self._plot_3d_samples(ax)

        if new_plot:
            ax.set_xlim([self.mins[0] - 1, self.maxs[0] + 1])
            ax.set_ylim([self.mins[1] - 1, self.maxs[1] + 1])
            ax.set_zlim([self.mins[2] - 1, self.maxs[2] + 1])

        if new_plot:
            plt.show()

    def _plot_3d_convex_hull(self, ax):
        for simplex in self.hull.simplices:
            simplex = np.append(simplex, simplex[0])  # Cycle back to the first point
            ax.plot(self.boundary_points[simplex, 0], self.boundary_points[simplex, 1], 
                    self.boundary_points[simplex, 2], 'k-')
        poly3d = [[self.boundary_points[vertice] for vertice in face] for face in self.hull.simplices]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='k', linewidths=1, alpha=0.2))

    def _plot_3d_hyperrectangle(self, ax):
        # Vertices of the 3D hyperrectangle
        vertices = np.array(list(product(*zip(self.mins, self.maxs))))

        # Define the 12 edges of the 3D hyperrectangle
        edges = [
            [vertices[j] for j in [0, 1, 3, 2]],
            [vertices[j] for j in [4, 5, 7, 6]],
            [vertices[j] for j in [0, 1, 5, 4]],
            [vertices[j] for j in [2, 3, 7, 6]],
            [vertices[j] for j in [0, 2, 6, 4]],
            [vertices[j] for j in [1, 3, 7, 5]]
        ]

        # Plot the edges
        for edge in edges:
            ax.add_collection3d(Poly3DCollection([edge], color='r', linewidths=1, edgecolors='r', alpha=.25))

    def _plot_3d_samples(self, ax):
        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], color='red', marker='x', label='Samples')