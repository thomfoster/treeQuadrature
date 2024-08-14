import matplotlib.pyplot as plt
import time, warnings

from .treeIntegrator import TreeIntegrator
from ..queues import ReservoirQueue
from ..container import Container
from ..splits import Split
from ..samplers import Sampler
from ..containerIntegration import ContainerIntegral
from ..exampleProblems import Problem

from typing import Callable, Optional


default_queue = ReservoirQueue(accentuation_factor=100)


# a function for debugging
def save_weights_image(q):
    weights = q.weights
    ps = q.get_probabilities(weights)
    plt.figure()
    plt.yscale("log")
    plt.hist(ps)
    plt.savefig(
        "/home/t/Documents/4yp/evidence-with-kdtrees/" +
        "src/treeQuadrature/results/images/ps_"
        + str(q.n) + ".png")
    plt.close()


class LimitedSampleIntegrator(TreeIntegrator):
    def __init__(
            self,
            N: int,
            base_N: int,
            active_N: int,
            split: Split,
            integral: ContainerIntegral,
            weighting_function: Callable,
            sampler: Optional[Sampler]=None,
            queue: ReservoirQueue=default_queue):
        """
        Integrator that builds on from queueIntegrator with more friendly
        controls - just keeps sampling until all samples used up.
        User does not need to specify the stopping condition

        Parameters
        ----------
        N : int
            Total number of samples to use.
        base_N : int
            Number of initial base samples.
        active_N : int
            Number of active samples per iteration.
        split : function
            Function to split a container into sub-containers.
        integral : function
            Function to compute the integral over a container.
        weighting_function : function
            Function to compute the weight of a container.
        sampler : Sampler
            a method for generating initial samples
            when problem does not have rvs method. 
            Default: UniformSampler
        queue : class
            Queue class to manage the containers, default is PriorityQueue.

        Example
        -------
        >>> from treeQuadrature.integrators import LimitedSampleIntegrator
        >>> from treeQuadrature.splits import MinSseSplit
        >>> from treeQuadrature.containerIntegration import RandomIntegral
        >>> from treeQuadrature.exampleProblems import SimpleGaussian
        >>> problem = SimpleGaussian(D=2)
        >>> 
        >>> minSseSplit = MinSseSplit()
        >>> randomIntegral = RandomIntegral()
        >>> volume_weighting = lambda container: container.volume
        >>> integ_limited = LimitedSampleIntegrator(
        >>>    N=1000, base_N=500, active_N=10, split=minSseSplit, integral=randomIntegral, 
        >>>    weighting_function=volume_weighting
        >>> )
        >>> estimate = integ_limited(problem)
        >>> print("error of random integral =", 
        >>>      str(100 * np.abs(estimate - problem.answer) / problem.answer), "%")
        """
        
        if sampler is None:
            super().__init__(split, integral, base_N)
        else:
            super().__init__(split, integral, base_N, sampler=sampler)
        self.N = N
        self.active_N = active_N
        self.weighting_function = weighting_function
        self.queue = queue

    def __call__(self, problem: Problem, return_N: bool=False, return_containers: bool=False, 
                 return_std: bool=False, verbose: bool=False, **kwargs):
        return super().__call__(problem, return_N=return_N, 
                                return_containers=return_containers, 
                                return_std=return_std, 
                                verbose=verbose,
                                integrand=problem.integrand, **kwargs)

    def construct_tree(self, root: Container, integrand: Callable, 
                       verbose: bool = False, **kwargs):
        """
        Actively refine the containers with samples.

        Parameters
        ----------
        root : Container
            The root container.
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
        max_iter = kwargs.get('max_iter', 1e4)

        q = self.queue
        q.put(root, 1)
        finished_containers = []
        num_samples_left = self.N - self.base_N

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