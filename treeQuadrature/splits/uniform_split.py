import numpy as np

from ..container import Container
from .base_class import Split


class UniformSplit(Split):
    """
    Split container into 2^n_divisions voxelled containers.

    Attributes
    ----------
    n_divisions : int
        The number of divisions along each axis.
        Default : 2
    """

    def __init__(self, n_divisions: int = 2):
        self.n_divisions = n_divisions

    def split(self, container, **kwargs):
        n_divisions = kwargs.get("n_divisions", self.n_divisions)

        # Ensure n_divisions is positive
        if n_divisions <= 0:
            raise ValueError("n_divisions must be a positive integer")

        # Get the mins and maxs of the container
        mins = container.mins
        maxs = container.maxs

        # Generate the split points along each dimension
        split_points = [
            np.linspace(mins[i], maxs[i], n_divisions + 1)
            for i in range(container.D)
        ]

        subcontainers = []

        # Iterate over all possible combinations of split points
        for indices in np.ndindex(*(n_divisions,) * container.D):
            sub_mins = [split_points[dim][indices[dim]]
                        for dim in range(container.D)]
            sub_maxs = [
                split_points[dim][indices[dim] + 1]
                for dim in range(container.D)
            ]

            # Filter points that lie within the sub-container
            mask = np.all((container.X >= sub_mins) &
                          (container.X < sub_maxs), axis=1)
            sub_X = container.X[mask]
            sub_y = container.y[mask]

            # Create a new sub-container
            subcontainer = Container(sub_X, sub_y,
                                     mins=sub_mins, maxs=sub_maxs)
            subcontainers.append(subcontainer)

        return subcontainers
