from treeQuadrature.compare_integrators import test_integrator_performance_with_params
from treeQuadrature.example_problems import Camel
from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.container_integrators import AdaptiveRbfIntegral
from treeQuadrature.trees import SimpleTree

import numpy as np
import os

n_repeat = 10
max_time = 500
Ds = np.arange(2, 14, 3)
integral = AdaptiveRbfIntegral(n_splits=0, max_redraw=3, n_samples=20)
integ = TreeIntegrator(1000, tree=SimpleTree(P=50), integral=integral)
integ.name = "TQ with Rbf"

script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    for D in Ds:
        params = {'base_N' : range(5000, 35000, 5000), 
                  'P' : range(20, 81, 15)}
    
        problem = Camel(D)
        output_path = os.path.join(script_dir, 
                                f"../test_results/tree_params/{integ.name}_{str(problem)}_results_{n_repeat}repeat.csv")
        test_integrator_performance_with_params(integ, problem,
                                                params, 
                                                output_file=output_path,
                                                n_repeat = n_repeat, 
                                                max_time=max_time)