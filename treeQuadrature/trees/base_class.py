from abc import ABC, abstractmethod
from typing import List

from ..container import Container


class Tree(ABC):
    @abstractmethod
    def construct_tree(self, root: Container,
                       *args, **kwargs) -> List[Container]:
        """
        Construct a tree from the given data.

        Parameters
        ----------
        root : Container
            The root container with all initial samples.

        Returns
        -------
        List[Container]
            A list of containers that form the tree. (partition of the space)
        """
        pass

    def _check_root(self, root: Container):
        """
        Check if the root is a container
        and has samples

        Parameters
        ----------
        root : Container
            The root container to be checked.
        """
        if not isinstance(root, Container):
            raise Exception("Root must be a container")

        if root.N == 0:
            raise Exception("Root container has no samples")
