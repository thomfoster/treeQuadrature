from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.exampleProblems import ProductPeakProblem, ExponentialProductProblem, C0Problem, CornerPeakProblem, OscillatoryProblem
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral, RbfIntegral
from treeQuadrature.samplers import McmcSampler
from treeQuadrature.compare_integrators import test_container_integrals

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
location_prefix = 'ablation_adaptive/'
    

if __name__ == '__main__':
    for D in Ds:
        problems = [
            ProductPeakProblem(D, a=13),
            C0Problem(D, a=2),
            CornerPeakProblem(D, a=10),
            ExponentialProductProblem(D), 
            OscillatoryProblem(D, a=np.array(10/ np.linspace(1, D, D)))
        ]

        output_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}results_{D}D_{args['n_repeat']}repeat.csv")
        config_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}results_{D}D_{args['n_repeat']}repeat.json")

        args['N'] = 7000 + D * 500

        with open(config_file, 'w') as file:
            json.dump(args, file, indent=4)

        integral_mean = AdaptiveRbfIntegral(n_samples=args['n_samples'], 
                                                max_redraw = 0,
                                                n_splits=0)
        integral_mean.name = 'Adaptive Rbf (mean)'
        integral = AdaptiveRbfIntegral(n_samples= args['n_samples'], max_redraw=0, 
                                       fit_residuals=False,
                                       n_splits=0)
        integral.name = 'Adaptive Rbf'
        integral_non_adaptive = RbfIntegral(n_samples= args['n_samples'], max_redraw=0, 
                                                n_splits=0, 
                                                range=args['range'])
        integral_non_adaptive.name = 'Non Adaptive Rbf'
        integrals = [integral_mean, integral, integral_non_adaptive]

        integ = SimpleIntegrator(base_N=args['N'], P=args['P'], split=split, 
                                 integral=None, 
                                 sampler=McmcSampler())
            
        test_container_integrals(problems, integrals, integ, output_file, 
                                 n_repeat=args['n_repeat'])