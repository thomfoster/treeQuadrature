import numpy as np
from scipy.spatial import ConvexHull
import warnings

# for plotting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from itertools import product


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
    Represents a convex hull in n-dim space
    with finite volume
    and the samples it holds.

    Attributes
    ----------
    _X, _y : ArrayList
        stores the samples and evaluations efficiently 
    mins, maxs : float or list or numpy array of shape (D,)
        the low and high boundaries of the 
        only for hyper-rectangle containers
    volume : float
        volume of the container
    is_finite : bool
        indicator of whether the container has finite volume
    midpoint : numpy array of shape (D,)
        the midpoint of container
    boundary_points : numpy array
        each row is a boundary point of the convex hull
        will be automatically arranged in clockwise order 
    hull : ConvexHull
        created based on boundary_points.
        None if container is not finite or the problem is 1D
    is_rectangle : bool
        indicates whether the Container is a hyper-rectangle

    Methods 
    -------
    add(new_x, new_y)
        add new sample points
    rvs(n)
        uniformly randomly draw n samples in this container
        Return : numpy array of shape (n, D)
    split(split_dimension, split_value)
        split the container into two along split_dimension at split_value
        Return : list of two sub-containers
    '''

    def __init__(self, X, y, mins=None, maxs=None, boundary_points=None):
        """
        Attributes
        ----------
        X : numpy array of shape (N, D)
            each row is a sample
        y : numpy array of shape (N, 1)
            the function value at each sample
        mins, maxs : numpy array of shape (D,)
            the low and high boundaries of the hyper-rectangle
            could be +- np.inf
        """
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

        ### compute basic properties
        if self.is_finite and self.D != 1:
            # reorder boundary points clockwise
            self.boundary_points = Container._reorder_clockwise(self.boundary_points)
            self.hull = ConvexHull(self.boundary_points)
            self.volume = self.hull.volume
        else:
            self.hull = None
        self.midpoint = np.mean(self.boundary_points, axis=0) if self.is_finite else np.nan

        ### add sample points into the hidden ArrayList
        # create empty ArrayList
        self._X = ArrayList(D=self.D)
        self._y = ArrayList(D=1)

        # filter points
        X_filtered, y_filtered = self._filter_points(X, y)
        self.add(X_filtered, y_filtered)

    def _handle_min_max_bounds(self, bounds, default_value):
        """Handle different types of min/max bounds."""
        if isinstance(bounds, (int, float)):
            return np.array([bounds] * self.D)
        elif isinstance(bounds, (list, np.ndarray)):
            return np.array(bounds)
        else:
            return np.array([default_value] * self.D)
        
    def _filter_points(self, X, y, return_bool = False):
        """
        Check whether all the points X are in the convex hull defined by self.boundary_points,
        and return a numpy array with those in the container. Throw a warning if any point is not
        in the container.

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            An array of points to check.
        y : np.ndarray of shape (N, )
            corresponding values
        return_bool : bool
            if true, return a bool inside 
        
        Returns
        -------
        np.ndarray, np.ndarray, bool
            Two arrays: one of the points that are within the convex hull or bounds, 
            and another of the corresponding y values.
            bool: indicates whether all points are inside the container
        
        """

        # 1-D case: check whether between bounds
        if self.D == 1:
            in_bounds = (X >= self.mins) & (X <= self.maxs)
            in_bounds = in_bounds.flatten()  # Ensure it's a 1D boolean array for indexing
            if not np.all(in_bounds):
                inside = False
                warnings.warn("Some points are out of the container bounds.")
            else: 
                inside = True

            if return_bool:
                return X[in_bounds], y[in_bounds], inside
            else: 
                return X[in_bounds], y[in_bounds]

        # general case
        in_hull = np.ones(len(X), dtype=bool)
        
        # check whether all ponits are in the container
        for i, point in enumerate(X):
            # allows for small numerical margin 1e-12
            if not all((np.dot(eq[:-1], point) + eq[-1] <= 1e-12) for eq in self.hull.equations):
                in_hull[i] = False
                warnings.warn(f"Point {point} is not in the container")

        inside = np.all(in_hull)
        if not inside:
            print(f'mins: {self.mins}, maxs: {self.maxs}')

        if return_bool:
            return X[in_hull], y[in_hull], inside
        else: 
            return X[in_hull], y[in_hull]

    @staticmethod
    def _is_convex(points):
        """
        Check if the points form a convex polygon.
        
        Parameters
        ----------
        points : np.ndarray
            An array of points representing the boundary of the polygon.
        
        Returns
        -------
        bool
            True if the polygon is convex, False otherwise.
        """
        n = len(points)
        if n < 4:
            return True  # A triangle is always convex

        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        is_positive = None
        for i in range(n):
            o = points[i]
            a = points[(i + 1) % n]
            b = points[(i + 2) % n]
            cross = cross_product(o, a, b)
            if cross != 0:
                if is_positive is None:
                    is_positive = cross > 0
                elif (cross > 0) != is_positive:
                    return False
        return True
    
    @staticmethod
    def _reorder_clockwise(points):
        """
        Reorder the boundary points so that they are in clockwise order.

        Parameters
        ----------
        points : np.ndarray
            An array of points representing the boundary of the polygon.

        Returns
        -------
        np.ndarray
            The reordered array of points in clockwise order.
        """
        # Calculate the centroid of the points
        centroid = np.mean(points, axis=0)

        # Calculate the angles of each point relative to the centroid
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

        # Sort the points based on the angles in clockwise order
        clockwise_order = np.argsort(angles)
        return points[clockwise_order]

    def _construct_hyperrectangle(self):
        """
        Construct boundary points of a hyper-rectangle from mins and maxs.
        
        Returns
        -------
        points : numpy array of shape (2^D, D)
            The vertices of the hyper-rectangle.
        """
        D = self.D
        # Create a meshgrid of indices for all combinations
        grid = np.indices((2,) * D).reshape(D, -1).T
        
        # Use the grid to select elements from mins and maxs
        mins_expanded = np.expand_dims(self.mins, axis=0)
        maxs_expanded = np.expand_dims(self.maxs, axis=0)
        
        points = mins_expanded + grid * (maxs_expanded - mins_expanded)
        return points

    def add(self, new_X, new_y):
        assert new_X.ndim == 2
        assert new_y.ndim == 2
        assert new_X.shape[0] == new_y.shape[0]
        assert new_X.shape[1] == self.D
        assert new_y.shape[1] == 1
        # TODO - figure out why are NaN values generated 
        # assert np.isnan(new_X).any(), 'new_X has NaN!'
        # assert np.isnan(new_y).any(), 'new_y has NaN!'
        # assert np.all(new_X >= self.mins), new_X[new_X < self.mins]
        # assert np.all(new_X <= self.maxs), new_X[new_X > self.maxs]
        _, _, all_inside = self._filter_points(new_X, new_y, return_bool=True)
        assert all_inside, (
            'Some points are not inside the container, '
            'cannot add new_X and new_y to container see warnings above'
        )


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
        rs : numpy array of shape (n, D)
            each row is a sample
        """
        ## TODO - modify this to deal with the case self.mins is -inf or self.maxs is inf

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