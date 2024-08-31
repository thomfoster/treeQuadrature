from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.example_problems import ProductPeak, ExponentialProduct, C0, CornerPeak, Oscillatory, Ripple, Camel
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.container_integrators import AdaptiveRbfIntegral, MidpointIntegral, RandomIntegral
from treeQuadrature.samplers import McmcSampler
from treeQuadrature.compare_integrators import test_container_integrals

import numpy as np
import os, json, argparse


parser = argparse.ArgumentParser(description="Compare three container integrals for various dimensions on the selected benchmark problems")
parser.add_argument('--dimensions', type=int, nargs='+', default=[2], help="List of problem dimensions (default: [2])")
parser.add_argument('--P', type=int, default=50, help='Size of the largest container (default: 50)')
parser.add_argument('--n_samples', type=int, default=30, help='number of samples drawn from each container (default: 30)')
parser.add_argument('--n_repeat', type=int, default=20, help='Number of repetitions for each test (default: 20)')
    

if __name__ == '__main__':
    split = MinSseSplit()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    location_prefix = 'container_integrals/'

    args = parser.parse_args()
    Ds = args.dimensions 

    args_dict = vars(args).copy()
    # remove attributes already displayed in file names
    del args_dict['dimensions']
    del args_dict['n_repeat']

    for D in Ds:
        problems = [
            ProductPeak(D, a=10),
            C0(D, a=10),
            Camel(D),
            Ripple(D),
            CornerPeak(D, a=10),
            ExponentialProduct(D), 
            Oscillatory(D, a=np.array(10/ np.linspace(1, D, D)))
        ]

        output_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}results_{D}D_{args.n_repeat}repeat.csv")
        config_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}configs_{D}D_{args.n_repeat}repeat.json")

        args_dict['N'] = 7000 + D * 500

        # save configuration
        with open(config_file, 'w') as file:
            json.dump(args_dict, file, indent=4)

        integral_rbf = AdaptiveRbfIntegral(n_samples=args.n_samples, 
                                                max_redraw = 0,
                                                n_splits=0)
        integral_rbf.name = 'Rbf'

        integral_mean = RandomIntegral(n_samples= args.n_samples)
        integral_mean.name = 'Mean'

        integral_midpoint = MidpointIntegral()
        integral_midpoint.name = 'Midpoint'
        integrals = [integral_rbf, integral_mean, integral_midpoint]

        integ = TreeIntegrator(base_N=args_dict['N'], tree=SimpleTree(P=args.P, split=split), 
                                 integral=None, sampler=McmcSampler())
            
        test_container_integrals(problems, integrals, integ, output_file, 
                                 n_repeat=args.n_repeat)