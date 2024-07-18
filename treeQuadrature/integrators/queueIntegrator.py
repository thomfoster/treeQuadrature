import numpy as np

from .treeIntegrator import TreeIntegrator
from ..queues import ReservoirQueue
from ..container import Container
from ..splits import Split
from ..containerIntegration import ContainerIntegral
from ..exampleProblems import Problem
from typing import Callable


# Default finished condition will never prevent container being split
def default_stopping_condition(container: Container) -> bool: return False


default_queue = ReservoirQueue(accentuation_factor=100)


class QueueIntegrator(TreeIntegrator):
    def __init__(
            self,
            base_N: int,
            split: Split,
            integral: ContainerIntegral,
            weighting_function: Callable,
            active_N: int=0,
            max_splits=np.inf,
            stopping_condition: Callable=default_stopping_condition,
            queue: ReservoirQueue=default_queue):
        """
        Integrator that builds on from SimpleIntegrator with more customised
        queueing.

        Parameters
        --------
        base_N : int
            the base number of samples to draw from the problem
                distribution
        split : Split
            the method to split the containers
        integral : ContainerIntegral
            the method to integrate the containers
        weighting_function : function
            maps Container -> R to give priority of container
            in queue
        active_N : int, optional
            the number of active samples to draw before a container is
                split
            Default: 0
        max_splits : int or np.inf, optional
            a limit on the number of splits the integrator can perform
            Default : np.inf
        n_splits : int
            actual number of splits 
        stopping_condition : function, optional
            maps Container -> Bool to indicate whether
                container should no longer be split
        queue : ReservoirQueue, optional
            where containers are pushed and popped from along with their
                weights

        Example
        -------
        >>> from treeQuadrature.integrators import QueueIntegrator
        >>> from treeQuadrature.splits import MinSseSplit
        >>> from treeQuadrature.containerIntegration import RandomIntegral
        >>> from treeQuadrature.exampleProblems import SimpleGaussian
        >>> problem = SimpleGaussian(D=2)

        >>> minSseSplit = MinSseSplit()
        >>> randomIntegral = RandomIntegral()
        >>> volume_weighting = lambda container: container.volume
        >>> stopping_small_containers = lambda container: container.N < 2
        >>> integ_queue = QueueIntegrator(
        >>>     base_N=1000, split=minSseSplit, integral=randomIntegral, 
        >>>     weighting_function=volume_weighting, max_splits=50, 
        >>>     stopping_condition=stopping_small_containers
        >>> )
        >>> estimate = integ_queue(problem)
        >>> print("error of random integral =", 
        >>>       str(100 * np.abs(estimate - problem.answer) / problem.answer), "%")
        """
        
        if np.isinf(
                max_splits) and (
                stopping_condition == default_stopping_condition):
            raise Exception(
                "Integrator with never terminate - either provide a stopping" +
                "condition or a maximum number of splits (max_splits)")

        super().__init__(split, integral, base_N)
        self.weighting_function = weighting_function
        self.active_N = active_N
        self.max_splits = max_splits
        self.n_splits = 0
        self.stopping_condition = stopping_condition
        self.queue = queue

    def __call__(self, problem: Problem, 
                 return_N: bool=False, return_containers: bool=False, return_std: bool=False):
        result = super().__call__(problem, return_N, return_containers, return_std, 
                                  integrand = problem.integrand)
        if return_N:
            result['n_splits'] = self.n_splits

        return result
    
    def construct_tree(self, root: Container, integrand: Callable):
        # reset number of splits
        self.n_splits = 0
        finished_containers = []
        q = self.queue
        q.put(root, 1)

        while (not q.empty()) and (self.n_splits < self.max_splits):

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

        # Empty queue
        while not q.empty():
            finished_containers.append(q.get())

        return finished_containers