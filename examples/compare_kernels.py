from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.exampleProblems import QuadraticProblem
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral, PolyIntegral
from treeQuadrature.samplers import McmcSampler
from treeQuadrature.compare_integrators import test_container_integrals

import os, json, argparse


parser = argparse.ArgumentParser(description="Compare three container integrals for various dimensions on the selected benchmark problems")
parser.add_argument('--dimensions', type=int, nargs='+', default=[2], help="List of problem dimensions (default: [2])")
args_parser = parser.parse_args()
Ds = args_parser.dimensions

args = {}

split = MinSseSplit()

args['P'] = 40
args['n_samples'] = 30
args['n_repeat'] = 20
args['range'] = 500

script_dir = os.path.dirname(os.path.abspath(__file__))
location_prefix = 'rbf_poly/'
    

if __name__ == '__main__':
    for D in Ds:
        problems = [QuadraticProblem(D)]

        output_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}results_{D}D_{args['n_repeat']}repeat.csv")
        config_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}configs_{D}D_{args['n_repeat']}repeat.json")

        args['N'] = 7000 + D * 500

        # save configuration
        with open(config_file, 'w') as file:
            json.dump(args, file, indent=4)

        integral_rbf = AdaptiveRbfIntegral(n_samples=args['n_samples'], 
                                                max_redraw = 0,
                                                n_splits=0)
        integral_rbf.name = 'Rbf'

        integral_poly = PolyIntegral(n_samples= args['n_samples'])
        integral_poly.name = 'Polynomial'

        integrals = [integral_rbf, integral_poly]

        integ = SimpleIntegrator(base_N=args['N'], P=args['P'], split=split, 
                                 integral=None, 
                                 sampler=McmcSampler())
            
        test_container_integrals(problems, integrals, integ, output_file, 
                                 n_repeat=args['n_repeat'])