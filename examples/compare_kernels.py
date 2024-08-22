from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.exampleProblems import QuadraticProblem
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral, PolyIntegral
from treeQuadrature.samplers import McmcSampler
from treeQuadrature.compare_integrators import test_container_integrals

import os, json, argparse


parser = argparse.ArgumentParser(description="Compare three container integrals for various dimensions on the selected benchmark problems")
parser.add_argument('--dimensions', type=int, nargs='+', default=[2], help="List of problem dimensions (default: [2])")
parser.add_argument('--P', type=int, default=50, help='Size of the largest container (default: 50)')
parser.add_argument('--n_samples', type=int, default=30, help='number of samples drawn from each container (default: 30)')
parser.add_argument('--n_repeat', type=int, default=20, help='Number of repetitions for each test (default: 20)')
    

if __name__ == '__main__':
    args = parser.parse_args()
    Ds = args.dimensions

    args_dict = vars(args).copy()
    # remove attributes already displayed in file names
    del args_dict['dimensions']
    del args_dict['n_repeat']

    split = MinSseSplit()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    location_prefix = 'rbf_poly/'

    for D in Ds:
        problems = [QuadraticProblem(D)]

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

        # quadratic kernel
        integral_poly = PolyIntegral(n_samples= args.n_samples, degrees=[2])
        integral_poly.name = 'Quadratic'

        integrals = [integral_rbf, integral_poly]

        integ = SimpleIntegrator(base_N=args_dict['N'], P=args.P, split=split, 
                                 integral=None, 
                                 sampler=McmcSampler())
            
        test_container_integrals(problems, integrals, integ, output_file, 
                                 n_repeat=args.n_repeat)