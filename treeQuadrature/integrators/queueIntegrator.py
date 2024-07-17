import numpy as np

from .integrator import Integrator
from ..queues import ReservoirQueue
from ..container import Container
from ..splits import Split
from ..containerIntegration import ContainerIntegral
from ..exampleProblems import Problem
from typing import Callable

import inspect
import warnings


# Default finished condition will never prevent container being split
def default_stopping_condition(container: Container) -> bool: return False


default_queue = ReservoirQueue(accentuation_factor=100)


class QueueIntegrator(Integrator):
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
    num_splits : int or np.inf, optional
        a limit on the number of splits the integrator can perform
        Default : np.inf
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
    >>>     weighting_function=volume_weighting, num_splits=50, 
    >>>     stopping_condition=stopping_small_containers
    >>> )
    >>> estimate = integ_queue(problem)
    >>> print("error of random integral =", 
    >>>       str(100 * np.abs(estimate - problem.answer) / problem.answer), "%")
    """

    def __init__(
            self,
            base_N: int,
            split: Split,
            integral: ContainerIntegral,
            weighting_function: Callable,
            active_N: int=0,
            num_splits=np.inf,
            stopping_condition: Callable=default_stopping_condition,
            queue: ReservoirQueue=default_queue):
        
        if np.isinf(
                num_splits) and (
                stopping_condition == default_stopping_condition):
            raise Exception(
                "Integrator with never terminate - either provide a stopping" +
                "condition or a maximum number of splits (num_splits)")

        self.base_N = base_N
        self.split = split
        self.integral = integral
        self.weighting_function = weighting_function
        self.active_N = active_N
        self.num_splits = num_splits
        self.stopping_condition = stopping_condition
        self.queue = queue

    def __call__(self, problem: Problem, 
                 return_N: bool=False, return_containers: bool=False, return_std: bool=False):
        # Draw samples
        X = problem.d.rvs(self.base_N)
        y = problem.pdf(X)

        root = Container(X, y, mins=problem.lows, maxs=problem.highs)

        # Construct tree
        finished_containers, n_splits = self.construct_tree(root, integrand=problem.pdf)

        # uncertainty estimates
        if return_std:
            signature = inspect.signature(self.integral.containerIntegral)
            if 'return_std' in signature.parameters:
                results = [self.integral.containerIntegral(cont, problem.pdf, return_std=True)
                         for cont in finished_containers]
                contributions = [result[0] for result in results]
                stds = [result[1] for result in results]
            else:
                warnings.warn('integral does not have parameter return_std, will be ignored', 
                              UserWarning)
                return_std = False

                contributions = [self.integral.containerIntegral(cont, problem.pdf)
                            for cont in finished_containers]

        else: 
            # Integrate containers
            contributions = [self.integral.containerIntegral(cont, problem.pdf)
                            for cont in finished_containers]
            
        G = np.sum(contributions)
        N = sum([cont.N for cont in finished_containers])

        return_values = {'estimate' : G}

        if return_N:
            return_values['n_evals'] = N
            return_values['n_splits'] = n_splits
        if return_containers:
            return_values['containers'] = finished_containers
            return_values['contributions'] = contributions
        if return_std:
            return_values['stds'] = stds

        return return_values
    
    def construct_tree(self, root: Container, integrand: Callable):
        current_n_splits = 0
        finished_containers = []
        q = self.queue
        q.put(root, 1)

        while (not q.empty()) and (current_n_splits < self.num_splits):

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

                current_n_splits += 1

        # Empty queue
        while not q.empty():
            finished_containers.append(q.get())

        return finished_containers, current_n_splits