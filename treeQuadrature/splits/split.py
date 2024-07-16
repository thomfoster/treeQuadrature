from abc import ABC, abstractmethod
from typing import List
from ..container import Container

class Split(ABC):
    """
    Abstract base class for splitting a container into containers.
    """

    @abstractmethod
    def split(self, container: Container) -> List[Container]:
        """
        Split the given container into two containers.

        Parameters
        ----------
        container : Container
            The container to be split.

        Returns
        -------
        List[Container]
            A list of two Container instances.
        """
        pass