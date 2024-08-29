from typing import Callable, Optional, List
import numpy as np
import time

from .tree import Tree
from ..splits import Split, MinSseSplit
from ..queues import ReservoirQueue
from ..container import Container


def default_stopping_condition(container: Container) -> bool: return False

class WeightedTree(Tree):
    """
    A Tree subclass that constructs the tree using a weighted queue mechanism
    for prioritisation.
    Stopping Criteria: when stopping_condition is met.

    Attributes
    ---------
    max_splits : int or np.inf
        a limit on the number of splits the integrator can perform. 
        Default : np.inf
    split : Split
        the method to split the containers
    weighting_function : Callable[[Container], float]
        maps Container -> float to give priority of container
        in queue
    active_N : int
        the number of active samples to draw before a container is
        split. 
        Default: 0
    n_splits : int
        actual number of splits 
    stopping_condition : Callable[[Container], bool], optional
        maps Container -> Bool to indicate whether
        container should no longer be split
    queue : ReservoirQueue, optional
        where containers are pushed and popped from along with their
        weights
    """

    def __init__(self, weighting_function: Callable[[Container], float], 
                 max_splits: int = float('inf'), active_N: int = 0, 
                 stopping_condition: Callable[[Container], bool] = default_stopping_condition,
                 split: Optional[Split] = None, 
                 queue: Optional[ReservoirQueue] = None,
                 *args, **kwargs):
        """
        Parameters
        --------
        weighting_function : Callable[[Container], float]
            maps Container -> float to give priority of container
            in queue
        max_splits : int or np.inf, optional
            a limit on the number of splits the integrator can perform
            Default: np.inf
        active_N : int, optional
            the number of active samples to draw before a container is
            split in each iteration.
            Warning: active_N > 0 will create more function evaluations. 
            Default: 0
        stopping_condition : Callable[[Container], bool], optional
            maps Container -> Bool to indicate whether
            container should no longer be split
        split : Optional[Split], optional
            the method to split the containers. 
            Default is MinSseSplit
        queue : Optional[ReservoirQueue], optional
            where containers are pushed and popped from along with their
            weights. 
            Default: ReservoirQueue(accentuation_factor=100)
        """
        super().__init__(*args, **kwargs)

        if np.isinf(max_splits) and (
                stopping_condition == default_stopping_condition):
            raise Exception(
                "Integrator with never terminate - either provide a stopping" +
                "condition or a maximum number of splits (max_splits)")
    
        self.max_splits = max_splits
        self.active_N = active_N
        self.split = split if split is not None else MinSseSplit()
        self.weighting_function = weighting_function
        self.stopping_condition = stopping_condition
        self.queue = queue if queue is not None else ReservoirQueue(accentuation_factor=100)

    def construct_tree(self, root: Container, integrand: Callable[[np.ndarray], float], 
                       **kwargs) -> List[Container]:
        """
        Construct the tree using a weighted queue mechanism.

        Parameters
        ----------
        root : Container
            The root container with all initial samples. 
        integrand : callable
            The integrand function to be integrated.
        """
        if self.active_N == 0:
            self._check_root(root)

        verbose = kwargs.get('verbose', False)

        self.n_splits = 0
        finished_containers = []
        q = self.queue
        q.put(weight=1, item=root) 

        # for verbose tracking
        start_time = time.time()
        iteration_count = 0

        while (not q.empty()) and (self.n_splits < self.max_splits):
            iteration_count += 1

            c = q.get()

            if self.stopping_condition(c):
                finished_containers.append(c)

            else:
                children = self.split.split(c)
                for child in children:

                    if self.active_N > 0:
                        X = child.rvs(self.active_N)
                        y = integrand(X)
                        child.add(X, y)

                    weight = self.weighting_function(child)
                    q.put(child, weight)

                self.n_splits += 1

            if iteration_count % 100 == 0 and verbose:  # Log every 100 iterations
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration_count}: Queue size = {q.qsize()}, "
                    f"number of containers = {len(finished_containers)}, "
                    f"Elapsed time = {elapsed_time:.2f}s")

        # Empty queue
        while not q.empty():
            finished_containers.append(q.get())        
        
        total_time = time.time() - start_time
        if verbose:
            print(f"Total finished containers: {len(finished_containers)}")
            print(f"Total iterations: {iteration_count}")
            print(f"Total time taken: {total_time:.2f}s")

        return finished_containers

        