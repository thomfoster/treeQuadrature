import numpy as np

from ..container import Container
from queue import SimpleQueue

class SimpleIntegrator:
    '''
    A simple integrator has the following pattern:
        - Draw <N> samples
        - Keep performing <split> method on containers...
        - ...until each container has less than <P> samples
        - Then perform  <integral> on each container and sum.
    '''
    
    def __init__(self, N, P, split, integral):
        self.N = N
        self.P = P
        self.split = split
        self.integral = integral
        
    def __call__(self, problem, return_N=False):
        D = problem.D
        
        # Draw samples
        X = problem.d.rvs(self.N)
        y = problem.pdf(X)
        
        root = Container(X, y, mins=[problem.low]*D, maxs=[problem.high]*D)
        
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

        # Integrate containers
        contributions = [self.integral(cont, problem.pdf) for cont in finished_containers]
        G = np.sum(contributions)
        N = sum([cont.N for cont in finished_containers])
        
        ret = (G,N) if return_N else G
        return ret