import numpy as np

from .integrator import Integrator
from ..container import Container
from ..containerIntegration import ContainerIntegral
from ..splits import Split
from ..exampleProblems import Problem
from queue import SimpleQueue

import inspect
import warnings


class SimpleIntegrator(Integrator):
    '''
    A simple integrator with the following steps:
        - Draw <N> samples
        - Keep performing <split> method on containers...
        - ...until each container has less than <P> samples
        - Then perform  <integral> on each container and sum.

    Attributes
    ----------
    N : int
        total number of samples
    P : int
        maximum number of samples in each container
    split : Split
        a method to split a container (for tree construction)
    integral : ContainerIntegral 
        a method to evaluate the integral of f on a container
    
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

    def __init__(self, N, P, split: Split, integral: ContainerIntegral):
        # variable checks
        assert isinstance(N, int), "N must be an integer"
        assert isinstance(P, int), "P must be an integer"

        self.N = N
        self.P = P
        self.split = split
        self.integral = integral


    def __call__(self, problem: Problem, 
                 return_N: bool=False, return_containers: bool=False, return_std: bool=False):
        """
        Parameters 
        ----------
        problem : Problem
            the integration problem being solved
        return_N : bool
            if true, return the number of function evaluations
        return_containers : bool
            if true, return containers and their contributions as well
        return_std : bool
            if true, return the standard deviation estimate. 
            Ignored if integral does not give std estimate
        
        Return
        -------
        dict
            estimate : float
                estimated integral value
            n_evals : int
                number of function estiamtions
            containers : list[Container]
                list of Containers
            contribtions : list[float]
                contributions of each container in estimate
            stds : list[float]
                standard deviation of the integral estimate in each container
        """

        # Draw samples
        X = problem.d.rvs(self.N)
        y = problem.pdf(X)

        # initialise a container with all samples
        root = Container(X, y, mins=problem.lows, maxs=problem.highs)

        containers = self.construct_tree(root)

        # uncertainty estimates
        if return_std:
            signature = inspect.signature(self.integral.containerIntegral)
            if 'return_std' in signature.parameters:
                results = [self.integral.containerIntegral(cont, problem.pdf, return_std=True)
                         for cont in containers]
                contributions = [result[0] for result in results]
                stds = [result[1] for result in results]
            else:
                warnings.warn('integral does not have parameter return_std, will be ignored', 
                              UserWarning)
                return_std = False

                contributions = [self.integral.containerIntegral(cont, problem.pdf)
                            for cont in containers]

        else: 
            # Integrate containers
            contributions = [self.integral.containerIntegral(cont, problem.pdf)
                            for cont in containers]
            
        G = np.sum(contributions)
        N = sum([cont.N for cont in containers])

        return_values = {'estimate' : G}

        if return_N:
            return_values['n_evals'] = N
        if return_containers:
            return_values['containers'] = containers
            return_values['contributions'] = contributions
        if return_std:
            return_values['stds'] = stds

        return return_values

    def construct_tree(self, root):
        # Construct tree
        finished_containers = []
        q = SimpleQueue()
        q.put(root)

        while not q.empty():

            c = q.get()

            if c.N <= self.P:
                finished_containers.append(c)
            else:
                children = self.split.split(c)
                for child in children:
                    q.put(child)
            
        return finished_containers