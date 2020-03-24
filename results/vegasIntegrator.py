import wandb
import sys
import numpy as np

from tqdm import tqdm
from datetime import datetime

from problems import SimpleGaussian

# integrator specific imports
import vegas

# define global parameters
N = 10_000                  # num samples to draw
NITN = 10                   # vegas specific parameter
Ds = list(range(1,11))      # dimensions to test on

# class Integrator

class ShapeAdapter:
    def __init__(self, f):
        self.f = f
    def __call__(self, X):
        return self.f(X)[0,0]

class VegasIntegrator:
    def __init__(self, N, NITN):
        self.N = N
        self.NITN = NITN

    def __call__(self, problem):
        integ = vegas.Integrator([[-1.0, 1.0]]*problem.D)
        f = ShapeAdapter(problem.pdf)
        return integ(f, nitn=self.NITN, neval=self.N).mean
        

# def experiment(problem, integ)
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
    wandb.init(project="SimpleGaussian")

    # Params for config logging
    wandb.config.Ds = Ds
    wandb.config.key = str(sys.argv[1])
    wandb.config.N = N
    wandb.config.NITN = NITN
    
    integ = VegasIntegrator(N=N, NITN=NITN)

    # Run the experiment
    for D in Ds:
        problem = SimpleGaussian(D)
        experiment(problem, integ)
        
if __name__ == '__main__':
    main()