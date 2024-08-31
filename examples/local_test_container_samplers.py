from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.example_problems import ProductPeak, ExponentialProduct, C0, CornerPeak, Oscillatory, SimpleGaussian, Camel, QuadCamel
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.container_integrators import AdaptiveRbfIntegral
from treeQuadrature.samplers import McmcSampler, SobolSampler
from treeQuadrature.compare_integrators import test_container_integrals
from treeQuadrature.trees import SimpleTree

import numpy as np
import os, json, argparse


parser = argparse.ArgumentParser(description="Compare Iterative Fitting scheme and even samples for various dimensions")
parser.add_argument('--dimensions', type=int, nargs='+', default=[2], help="List of problem dimensions (default: [2])")
args_parser = parser.parse_args()
Ds = args_parser.dimensions

args = {}

split = MinSseSplit()

args['P'] = 50
args['n_samples'] = 30
args['n_splits'] = 5
args['n_repeat'] = 10
args['range'] = 500

script_dir = os.path.dirname(os.path.abspath(__file__))
location_prefix = 'container_samplers/'
    

if __name__ == '__main__':
    for D in Ds:
        problems = [
            SimpleGaussian(D)
            # Camel(D),
            # ProductPeak(D, a=13),
            # C0(D, a=10),
            # CornerPeak(D, a=10),
            # ExponentialProduct(D)
        ]

        output_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}results_{D}D_{args['n_repeat']}repeat.csv")
        config_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}results_{D}D_{args['n_repeat']}repeat.json")

        args['N'] = 7000 + D * 500

        with open(config_file, 'w') as file:
            json.dump(args, file, indent=4)

        integral_uniform = AdaptiveRbfIntegral(n_samples=args['n_samples'], 
                                                max_redraw = 0,
                                                n_splits=0)
        integral_uniform.name = 'Uniform Sampling'
        integral_qmc = AdaptiveRbfIntegral(n_samples= args['n_samples'], max_redraw=0, 
                                       fit_residuals=False,
                                       n_splits=0, sampler=SobolSampler)
        integral_qmc.name = 'QMC using Sobol'
        integrals = [integral_uniform, integral_qmc]

        integ = TreeIntegrator(args['N'], tree=SimpleTree(P=args['P'], split=split), 
                           integral=None, sampler=McmcSampler())
            
        test_container_integrals(problems, integrals, integ, output_file, 
                                 n_repeat=args['n_repeat'])