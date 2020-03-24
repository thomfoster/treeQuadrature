import wandb
import numpy as np
import sys

from problems import SimpleGaussian
from treeQuadrature import Container
from treeQuadrature.splits import kdSplit
from treeQuadrature.containerIntegration import midpointIntegral
from queue import SimpleQueue
from functools import partial

from datetime import datetime

Ds = list(range(1,11))  # dimensions to test
N = 100_000
P = 5
split = kdSplit
integral = midpointIntegral


# Define the Integrator we gonna be testing today
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
        
    def __call__(self, problem):
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
        
        return G
    

# Define the test for each dimension
def experiment(problem, integ):
    start_time = datetime.now()
    I_hat = integ(problem)
    end_time = datetime.now()
    
    d = {}
    d['D'] = problem.D
    d['N'] = integ.N
    d['pcntError'] = 100 * (I_hat - problem.answer) / problem.answer
    d['time'] =  (end_time - start_time).total_seconds()
    
    wandb.log(d)
    
def main():
    assert len(sys.argv) == 2, 'Usage: thisScript.py groupKey'
    
    # Set up experiment
    wandb.init(project="BoilerPlate")

    # Params for config logging
    wandb.config.Ds = Ds
    wandb.config.key = str(sys.argv[1])
    wandb.config.N = N
    wandb.config.P = P
    wandb.config.split = str(split)
    wandb.config.integral = str(integral)

    # Run the experiment
    for D in Ds:
        problem = SimpleGaussian(D)
        # make split, integral partials here if needed
        integ = SimpleIntegrator(N, P, split, integral)
        experiment(problem, integ)
        
if __name__ == '__main__':
    main()