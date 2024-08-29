from queue import SimpleQueue
from typing import List, Callable, Optional
import time, warnings

from .tree import Tree
from ..container import Container
from ..splits import Split, MinSseSplit
from ..queues import ReservoirQueue

class LimitedSampleTree(Tree):
    def __init__(self, weighting_function: Callable[[Container], float],
                    N: int = 1000,
                    active_N: int = 0,
                    split: Split=MinSseSplit(),
                    queue: Optional[ReservoirQueue]=None,
                    *args, **kwargs):
        """
        Parameters
        ----------
        N : int
            Total number of samples to use.
        active_N : int
            Number of active samples per iteration.
        split : function
            Function to split a container into sub-containers.
        weighting_function : function
            Function to compute the weight of a container.
        queue : class
            Queue class to manage the containers, default is PriorityQueue.
        """
        super().__init__(*args, **kwargs)
        self.N = N
        self.weighting_function = weighting_function
        self.active_N = active_N
        self.split = split
        self.queue = queue if queue is not None else ReservoirQueue(accentuation_factor=100)
        

    def construct_tree(self, root: Container, integrand: Callable, 
                       verbose: bool = False, **kwargs) -> List[Container]:
        """
        Actively refine the containers with samples.

        Parameters
        ----------
        root : Container
            The root container with all initial samples. 
        integrand : Callable
            The integrand function.
        max_iter : int, optional
            Maximum number of iterations, 
            by default 1e4
        verbose : bool, optional
            Whether to print verbose output, by default False.

        Returns
        -------
        List[Container]
            A list of finished containers.
        """
        if self.active_N == 0:
            self._check_root(root)

        max_iter = kwargs.get('max_iter', 1e4)

        base_N = root.N

        q = self.queue
        q.put(root, 1)

        finished_containers = []
        num_samples_left = self.N - base_N

        # for verbose tracking
        start_time = time.time()
        iteration_count = 0

        while not q.empty() and iteration_count < max_iter:

            # save_weights_image(q)

            c = q.get()

            if num_samples_left >= self.active_N:
                X = c.rvs(self.active_N)
                y = integrand(X)
                c.add(X, y)
                num_samples_left -= self.active_N

            elif c.N < 2:
                finished_containers.append(c)
                continue

            children = self.split.split(c)
            for child in children:
                weight = self.weighting_function(child)
                if weight == 0:
                    warnings.warn(f"Container has 0 weight, mins: {child.mins}, maxs: {child.maxs}"
                                  f"volume: {child.volume}")
                q.put(child, weight)

            if iteration_count % 100 == 0 and verbose:  # Log every 100 iterations
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration_count}: Queue size = {q.n}, "
                      f"Number of containers = {len(finished_containers)}, "
                      f"Elapsed time = {elapsed_time:.2f}s")
                
            iteration_count += 1
        
        total_time = time.time() - start_time

        if iteration_count == max_iter:
            warnings.warn(
                'Maximum iterations reached. Either increase max_iter or check split and samples.', 
                RuntimeWarning
            )
            # append containers left
            while not q.empty():
                c = q.get()
                finished_containers.append(c)

        if verbose:
            print(f"Total finished containers: {len(finished_containers)}")
            print(f"Total iterations: {iteration_count}")
            print(f"Total time taken: {total_time:.2f}s")

        return finished_containers