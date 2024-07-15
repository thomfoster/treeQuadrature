import numpy as np

from ..container import Container
from queue import SimpleQueue

import inspect
import warnings


class SimpleIntegrator:
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
    split : function
        Attribute : Container
        Return : a list of two child containers [lcont, rcont]
    integral : function 
        Attributes : Container and the integrand function
        Return: float
            the estimated value of integral in the container 
        some basic integrals defined in containerIntegration 
    
    Methods
    -------
    __call__(problem, return_N, return_all)
        solves the problem given
    '''

    def __init__(self, N, P, split, integral):
        # variable checks
        assert isinstance(N, int), "N must be an integer"
        assert isinstance(P, int), "P must be an integer"

        self.N = N
        self.P = P
        self.split = split
        self.integral = integral


    def __call__(self, problem, return_N=False, return_all=False, return_std=False):
        """
        Parameters 
        ----------
        problem : Problem
            the integration problem being solved
        return_N : bool
            if true, return the number of containers
        return_all : bool
            if true, return containers and their contributions as well
        return_std : bool
            if true, return the standard deviation estimate. 
            Ignored if integral does not give std estimate
        
        Returns
        ------
        G : float
            estimated integral value
        N : int
            number of containers used
        containers : list
            list of Containers
        contribtions : list
            list of floats indicating contributions of each container in G
        """
        D = problem.D

        # Draw samples
        X = problem.d.rvs(self.N)
        y = problem.pdf(X)

        # initialise a container with all samples
        root = Container(X, y, mins=problem.lows, maxs=problem.highs)

        containers = self.construct_tree(root)

        # uncertainty estimates
        if return_std:
            signature = inspect.signature(self.integral)
            if 'return_std' in signature.parameters:
                results = [self.integral(cont, problem.pdf, return_std=True)
                         for cont in containers]
                contributions = [result[0] for result in results]
                stds = [result[0] for result in results]
            else:
                warnings.warn('integral does not have parameter return_std, will be ignored', 
                              UserWarning)
                return_std = False

                contributions = [self.integral(cont, problem.pdf)
                            for cont in containers]

        else: 
            # Integrate containers
            contributions = [self.integral(cont, problem.pdf)
                            for cont in containers]
            
        G = np.sum(contributions)
        N = sum([cont.N for cont in containers])

        return_values = [G]

        if return_N:
            return_values.append(N)
        if return_all:
            return_values.extend([N, containers, contributions])
        if return_std:
            return_values.append(stds)

        return tuple(return_values)

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
                children = self.split(c)
                for child in children:
                    q.put(child)
            
        return finished_containers