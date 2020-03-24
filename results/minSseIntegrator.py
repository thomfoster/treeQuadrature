import wandb
import numpy as np
import sys

from problems import SimpleGaussian
from treeQuadrature import Container
from treeQuadrature.splits import kdSplit, minSseSplit
from treeQuadrature.containerIntegration import midpointIntegral, randomIntegral
from queue import SimpleQueue

from datetime import datetime

Ds = list(range(1,11))  # dimensions to test
N = 7_000
P = 10

# Define the Integrator we gonna be testing today
class SimpleIntegrator:
    '''Split N samples until each container has less than P samples.'''
    
    def __init__(self, N, P, integral):
        self.N = N
        self.P = P
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
                children = minSseSplit(c, problem.pdf)
                for child in children:
                    q.put(child)

        # Integrate containers
        contributions = [randomIntegral(cont, problem.pdf, n=100) for cont in finished_containers]
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
    
    integ = SimpleIntegrator(N, P, midpointIntegral)

    # Run the experiment
    for D in Ds:
        problem = SimpleGaussian(D)
        experiment(problem, integ)
        
if __name__ == '__main__':
    main()