import numpy as np

from .treeIntegrator import TreeIntegrator
from ..containerIntegration import ContainerIntegral
from ..splits import Split
from queue import SimpleQueue


class SimpleIntegrator(TreeIntegrator):
    def __init__(self, base_N: int, P: int, split: Split, integral: ContainerIntegral):
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
        super().__init__(split, integral, base_N)
        self.P = P

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