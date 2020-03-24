import wandb
import sys
import numpy as np

from tqdm import tqdm
from datetime import datetime

from problems import SimpleGaussian

# define global parameters
N = 10_000_000              # num samples to draw
Ds = list(range(1,11))      # dimensions to test on

# class Integrator
class Integrator:
    def __init__(self):
        pass
    def __call__(self, problem):
        pass

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
    wandb.init(project="BoilerPlate")

    # Params for config logging
    wandb.config.Ds = Ds
    wandb.config.key = str(sys.argv[1])
    wandb.config.N = N
    
    integ = Integrator()

    # Run the experiment
    for D in Ds:
        problem = SimpleGaussian(D)
        experiment(problem, integ)
        
if __name__ == '__main__':
    main()