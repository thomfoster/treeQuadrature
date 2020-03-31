import numpy as np
import warnings

from ..queues import ReservoirQueue
from ..container import Container
from functools import partial

# Default finished condition will never prevent container being split
default_stopping_condition = lambda container: False
default_queue = ReservoirQueue(accentuation_factor=100)

class ReservoirIntegrator:

    def __init__(
        self, base_N, split, integral, weighting_function,
        active_N=0, num_splits=np.inf, stopping_condition=default_stopping_condition, 
        queue=default_queue):

        """
        Integrator that builds on from SimpleIntegrator with more customised queueing. 

        args:
        --------

        base_N - the base number of samples to draw from the problem distribution
        split  - the method to split the containers
        integral - the method to integrate the containers
        weighting_function - maps Container -> R to give priority of container in queue
        active_N - the number of active samples to draw before a container is split
        num_splits - a limit on the number of splits the integrator can perform
        stopping_condition - maps Container -> Bool to indicate whether container should no longer be split
        queue - where containers are pushed and popped from along with their weights
        """

        if np.isinf(num_splits) and stopping_condition == default_stopping_condition:
            raise Exception("Integrator with never terminate - either provide a stopping condition or a maximum number of splits")
        
        self.base_N = base_N
        self.split = split
        self.integral = integral
        self.weighting_function = weighting_function
        self.active_N = active_N
        self.num_splits = num_splits
        self.stopping_condition = stopping_condition
        self.queue = queue
        
    
    def __call__(self, problem, return_N=False, return_all=False):
        D = problem.D
        
        # Draw samples
        X = problem.d.rvs(self.base_N)
        y = problem.pdf(X)
        
        root = Container(X, y, mins=[problem.low]*D, maxs=[problem.high]*D)
        
        # Construct tree
        current_n_splits = 0
        finished_containers = []
        q = self.queue
        q.put(root,1)

        while (not q.empty()) and (current_n_splits < self.num_splits):

            c = q.get()

            if self.stopping_condition(c):
                finished_containers.append(c)
                
            else:
                children = self.split(c)
                for child in children:

                    if self.active_N > 0:
                        X = child.rvs(self.active_N)
                        y = problem.pdf(X)
                        child.add(X, y)

                    weight = self.weighting_function(child)
                    q.put(child, weight)

                current_n_splits += 1
        
        # Empty queue
        while not q.empty():
            finished_containers.append(q.get())

        # Integrate containers
        contributions = [self.integral(cont, problem.pdf) for cont in finished_containers]
        G = np.sum(contributions)
        N = sum([cont.N for cont in finished_containers])
        
        ret = (G,N) if return_N else G
        ret = (G, finished_containers, contributions, N, current_n_splits) if return_all else ret
        return ret