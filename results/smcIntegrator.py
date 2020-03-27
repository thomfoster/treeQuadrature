import wandb
import sys
import numpy as np

from tqdm import tqdm
from datetime import datetime

from problems import QuadCamel

# define globals
N = 100_000  # num samples to draw
Ds = list(range(1,11))

# class Integrator
class SmcIntegrator:
    def __init__(self, N):
        self.N = N

    def __call__(self, problem):
        xs = problem.p.rvs(self.N)
        ys = problem.d.pdf(xs).reshape(-1)
        return np.mean(ys)

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
    
    integ = SmcIntegrator(N)

    # Run the experiment
    for D in Ds:
        problem = QuadCamel(D)
        experiment(problem, integ)
        
if __name__ == '__main__':
    main()