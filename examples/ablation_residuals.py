from git import Tree
from treeQuadrature.integrator import TreeIntegrator
from treeQuadrature.exampleProblems import ProductPeakProblem, ExponentialProductProblem, C0Problem, CornerPeakProblem, OscillatoryProblem
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral, KernelIntegral
from treeQuadrature.samplers import McmcSampler
from treeQuadrature.compare_integrators import test_container_integrals
from treeQuadrature.trees import SimpleTree

import numpy as np
import os, json, argparse


parser = argparse.ArgumentParser(description="Compare three container integrals for various dimensions on the selected benchmark problems")
parser.add_argument('--dimensions', type=int, nargs='+', default=[2], help="List of problem dimensions (default: [2])")
parser.add_argument('--P', type=int, default=50, help='Size of the largest container (default: 50)')
parser.add_argument('--n_samples', type=int, default=30, help='number of samples drawn from each container (default: 30)')
parser.add_argument('--n_repeat', type=int, default=20, help='Number of repetitions for each test (default: 20)')
parser.add_argument('--range', type=float, default=500, help='Search range of non-adaptive Rbf Integral, as a factor of initial length scale (default: 50)')


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    location_prefix = 'ablation_adaptive/'
    
    args = parser.parse_args()
    Ds = args.dimensions

    args_dict = vars(args).copy()

    # remove attributes already displayed in file names
    del args_dict['dimensions']
    del args_dict['n_repeat']

    split = MinSseSplit()

    for D in Ds:
        problems = [
            ProductPeakProblem(D, a=13),
            C0Problem(D, a=2),
            CornerPeakProblem(D, a=10),
            ExponentialProductProblem(D), 
            OscillatoryProblem(D, a=np.array(10/ np.linspace(1, D, D)))
        ]

        output_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}results_{D}D_{args.n_repeat}repeat.csv")
        config_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}configs_{D}D_{args.n_repeat}repeat.json")

        args_dict['N'] = 7000 + D * 500

        with open(config_file, 'w') as file:
            json.dump(args_dict, file, indent=4)

        integral_mean = AdaptiveRbfIntegral(n_samples=args.n_samples, 
                                                max_redraw = 0,
                                                n_splits=0)
        integral_mean.name = 'Adaptive Rbf (mean)'
        integral = AdaptiveRbfIntegral(n_samples= args.n_samples, max_redraw=0, 
                                       fit_residuals=False,
                                       n_splits=0)
        integral.name = 'Adaptive Rbf'
        integral_non_adaptive = KernelIntegral(n_samples= args.n_samples, max_redraw=0, 
                                                n_splits=0, 
                                                range=args.range)
        integral_non_adaptive.name = 'Non Adaptive Rbf'
        integrals = [integral_mean, integral, integral_non_adaptive]

        integ = TreeIntegrator(base_N=args_dict['N'], tree=SimpleTree(P=args.P, split=split), 
                                 integral=None, sampler=McmcSampler())
            
        test_container_integrals(problems, integrals, integ, output_file, 
                                 n_repeat=args.n_repeat)