import matplotlib.pyplot as plt
import numpy as np

from .integrator import Integrator
from ..queues import ReservoirQueue
from ..container import Container
from ..splits import Split
from ..containerIntegration import ContainerIntegral
from ..exampleProblems import Problem

from typing import Callable
import warnings, inspect


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


class LimitedSampleIntegrator(Integrator):
    """
    Integrator that builds on from queueIntegrator with more friendly
    controls - just keeps sampling until all samples used up.
    User does not need to specify the stopping condition

    Parameters
    ----------
    N : int
        Total number of samples to use.
    base_N : int
        Number of base samples.
    active_N : int
        Number of active samples per iteration.
    split : function
        Function to split a container into sub-containers.
    integral : function
        Function to compute the integral over a container.
    weighting_function : function
        Function to compute the weight of a container.
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

    def __init__(
            self,
            N: int,
            base_N: int,
            active_N: int,
            split: Split,
            integral: ContainerIntegral,
            weighting_function: Callable,
            queue: ReservoirQueue=default_queue):
        
        self.N = N
        self.base_N = base_N
        self.active_N = active_N
        self.split = split
        self.integral = integral
        self.weighting_function = weighting_function
        self.queue = queue

    def __call__(self, problem: Problem, return_N: bool=False, return_containers: bool=False, return_std: bool=False):
        """
        Perform the integration process.

        Arguments
        ----------
        problem : Problem
            The integration problem to be solved
        return_N : bool
            if true, return the number of function evaluations
        return_containers : bool
            if true, return containers and their contributions as well
        return_std : bool
            if true, return the standard deviation estimate. 
            Ignored if integral does not give std estimate

        Returns
        -------
        dict
        """

        # Draw samples
        X = problem.d.rvs(self.base_N)
        y = problem.pdf(X)

        root = Container(X, y, mins=problem.lows, maxs=problem.highs)

        # construct tree
        finished_containers = self.construct_tree(root, integrand=problem.pdf)

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
        if return_containers:
            return_values['containers'] = finished_containers
            return_values['contributions'] = contributions
        if return_std:
            return_values['stds'] = stds

        return return_values
    

    def construct_tree(self, root: Container, integrand: Callable):
        """Actively refine the containers with samples"""
        q = self.queue
        q.put(root, 1)
        finished_containers = []
        num_samples_left = self.N - self.base_N

        while not q.empty():

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

        return finished_containers