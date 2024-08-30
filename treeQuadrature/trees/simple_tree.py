from queue import SimpleQueue
from typing import List
import time, warnings

from .base_class import Tree
from ..container import Container
from ..splits import Split, MinSseSplit



class SimpleTree(Tree):
    """
    Construct the tree using a simple queue (with no priority). 
    Stopping Criteria: when all containers hold less than P samples.
    """
    def __init__(self, split: Split=MinSseSplit(), 
                 P: int = 40, max_iter: int = 2000,
                 *args, **kwargs):
        """
        Initialize the SimpleTree.

        Parameters
        ----------
        split : Split, optional
            The method to split the containers, by default MinSseSplit.
        P : int, optional
            Maximum number of samples in each container, by default 40.
        max_iter : int, optional
            Maximum number of binary splits, by default 2000.
        """
        super().__init__(*args, **kwargs)
        self.split = split
        self.P = P
        self.max_iter = max_iter
        

    def construct_tree(self, root: Container, verbose: bool=False, 
                       max_iter=1e3) -> List[Container]:
        """
        Construct a tree of containers.

        Parameters
        ----------
        root : Container
            The root container with all initial samples. 
        verbose : bool, optional
            Whether to print verbose output, by default False.
        max_iter : int, optional
            Maximum number of binary splits.
            by default 2000.

        Returns
        -------
        List[Container]
            A list of finished containers.
        """
        self._check_root(root)

        # Construct tree
        finished_containers = []
        q = SimpleQueue()
        q.put(root)

        # for verbose tracking
        start_time = time.time()
        iteration_count = 0

        while not q.empty() and iteration_count < max_iter:
            iteration_count += 1
            c = q.get()

            if c.N <= self.P:
                finished_containers.append(c)
            else:
                children = self.split.split(c)
                if len(children) == 1:
                    finished_containers.append(c)
                else:
                    for child in children:
                        q.put(child)
            
            if iteration_count % 100 == 0 and verbose:  # Log every 100 iterations
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration_count}: Queue size = {q.qsize()}, "
                    f"number of containers = {len(finished_containers)}, "
                    f"Elapsed time = {elapsed_time:.2f}s")
                
        total_time = time.time() - start_time
        if verbose:
            print(f"Total finished containers: {len(finished_containers)}")
            print(f"Total iterations: {iteration_count}")
            print(f"Total time taken: {total_time:.2f}s")
        
        if iteration_count == max_iter:
            warnings.warn(
                'maximum iterations reached for constructing the tree, '
                'either incresae max_iter or check split and samples', 
                RuntimeWarning)
            # append containers left
            while not q.empty():
                c = q.get()
                finished_containers.append(c)
                
        return finished_containers