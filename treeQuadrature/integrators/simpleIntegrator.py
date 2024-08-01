from queue import SimpleQueue
from typing import Optional, List
import time, warnings

from .treeIntegrator import TreeIntegrator
from ..containerIntegration import ContainerIntegral
from ..splits import Split
from ..samplers import Sampler
from ..container import Container



class SimpleIntegrator(TreeIntegrator):
    def __init__(self, base_N: int, P: int, split: Split, integral: ContainerIntegral, 
                 sampler: Optional[Sampler]=None):
        '''
        A simple integrator with the following steps:
            - Draw <N> samples
            - Keep performing <split> method on containers...
            - ...until each container has less than <P> samples
            - Then perform  <integral> on each container and sum.

        Attributes
        ----------
        base_N : int
            total number of initial samples
        P : int
            maximum number of samples in each container
        split : Split
            a method to split a container (for tree construction)
        integral : ContainerIntegral 
            a method to evaluate the integral of f on a container
        sampler : Sampler
            a method for generating initial samples
            when problem does not have rvs method. 
            Default: UniformSampler
        
        Methods
        -------
        __call__(problem, return_N, return_all)
            solves the problem given

        Example
        -------
        >>> from treeQuadrature.integrators import SimpleIntegrator
        >>> from treeQuadrature.splits import MinSseSplit
        >>> from treeQuadrature.containerIntegration import RandomIntegral
        >>> from treeQuadrature.exampleProblems import SimpleGaussian
        >>> problem = SimpleGaussian(D=2)
        >>> 
        >>> minSseSplit = MinSseSplit()
        >>> randomIntegral = RandomIntegral()
        >>> integ = SimpleIntegrator(N=2_000, P=40, minSseSplit, randomIntegral)
        >>> estimate = integ(problem)
        >>> print("error of random integral =", 
        >>>      str(100 * np.abs(estimate - problem.answer) / problem.answer), "%")
        '''
        if sampler is None:
            super().__init__(split, integral, base_N)
        else:
            super().__init__(split, integral, base_N, sampler=sampler)
        self.P = P

    def construct_tree(self, root: Container, 
                       verbose: bool=False, max_iter: int=1e4) -> List[Container]:
        """
        Construct a tree of containers.

        Parameters
        ----------
        root : Container
            The root container.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        max_iter : int, optional
            Maximum number of iterations, 
            by default 1e4.

        Returns
        -------
        List[Container]
            A list of finished containers.
        """
        # Construct tree
        finished_containers = []
        q = SimpleQueue()
        q.put(root)

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
                'maximum iterations reached, either '
                'incresae max_iter or check split and samples', 
                RuntimeWarning)
                
        return finished_containers