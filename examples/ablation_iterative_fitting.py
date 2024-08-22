from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.exampleProblems import ProductPeakProblem, RippleProblem, C0Problem, CornerPeakProblem, OscillatoryProblem
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral
from treeQuadrature.samplers import McmcSampler
from treeQuadrature import Container
from treeQuadrature.compare_integrators import load_existing_results, write_results

import numpy as np
from traceback import print_exc
import os, json, time, argparse

parser = argparse.ArgumentParser(description="Compare Iterative Fitting scheme and even samples for various dimensions")
parser.add_argument('--dimensions', type=int, nargs='+', default=[2], help="List of problem dimensions (default: [2])")

args_parser = parser.parse_args()
Ds = args_parser.dimensions

args = {}


split = MinSseSplit()

args['P'] = 50
args['n_samples'] = 20
args['max_redraw'] = 4
args['n_repeat'] = 10

script_dir = os.path.dirname(os.path.abspath(__file__))
location_prefix = 'ablation_iterative_fitting/'

integrator_names = ['Iterative GP', 'Even Sample GP']

if __name__ == '__main__':
    for D in Ds:
        problems = [
            ProductPeakProblem(D, a=13),
            C0Problem(D, a=2),
            CornerPeakProblem(D, a=10),
            OscillatoryProblem(D, a=np.array(10/ np.linspace(1, D, D))), 
            RippleProblem(D)
        ]

        output_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}results_{D}D_{args['n_repeat']}repeat.csv")
        config_file = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}configs_{D}D_{args['n_repeat']}repeat.json")

        args['N'] = 7000 + D * 500
        # threshold on R2 score
        args['threshold'] = max(1 - D / 10, 0.3)

        with open(config_file, 'w') as file:
            json.dump(args, file, indent=4)

        integral = AdaptiveRbfIntegral(n_samples=args['n_samples'], 
                                                max_redraw = args['max_redraw'],
                                                n_splits=0, 
                                                threshold=args['threshold'])
        integral_non_iter = AdaptiveRbfIntegral(n_samples= args['n_samples'], max_redraw=0, 
                                                n_splits=0)
        integ = SimpleIntegrator(base_N=args['N'], P=args['P'], split=split, 
                                 integral=integral, 
                                 sampler=McmcSampler())


        def test_two_integrators(problem):
            ### integrate iteratively 
            X, y = integ.sampler.rvs(integ.base_N, problem.lows, problem.highs,
                                        problem.integrand)
            
            root = Container(X, y, mins=problem.lows, maxs=problem.highs)
            finished_containers = integ.construct_tree(root)

            start_time = time.time()
            results, containers = integ.integrate_containers(finished_containers, problem)
            end_time = time.time()

            N = sum([cont.N for cont in containers])
            contributions = [result['integral'] for result in results]
            estimate = np.sum(contributions)
            return_values = {'estimate' : estimate}
            return_values['n_evals'] = N
            return_values['time'] = end_time - start_time

            ### distribute these samples evenly to non-iterative integrator
            N_per_container = int((N - integ.base_N) / len(finished_containers))

            integral_non_iter.n_samples = N_per_container
            integ.integral = integral_non_iter

            start_time = time.time()
            results_non_iter, containers_non_iter = integ.integrate_containers(finished_containers, problem)
            end_time = time.time()
            
            contributions_non_iter = [result['integral'] for result in results_non_iter]
            estimate_non_iter = np.sum(contributions_non_iter)
            return_values_non_iter = {'estimate' : estimate_non_iter}
            N_non_iter = sum([cont.N for cont in containers_non_iter])
            return_values_non_iter['n_evals'] = N_non_iter
            return_values_non_iter['time'] = end_time - start_time
            
            return return_values, containers, return_values_non_iter, containers_non_iter
            
        existing_results = load_existing_results(output_file)

        if existing_results:
            is_first_run = False
        else:
            is_first_run = True
        
        results = []

        fieldnames = [
            'integrator', 'problem', 'true_value', 'estimate', 'estimate_std', 'error_type', 
            'error', 'error_std', 'n_evals', 'n_evals_std', 'time_taken', 'errors']

        for problem in problems:
            problem_name = str(problem)
            print(f'testing Probelm: {problem_name}')

            key = ('Iterative GP', problem_name)
            if key in existing_results and existing_results[key]['estimate'] != '':
                print(f'Skipping {problem_name}: already completed.')
                results.append(existing_results[key])
                continue
            
            estimates = [[], []]
            n_evals_list = [[], []]
            time_list = [[], []]

            for _ in range(args['n_repeat']):
                try: 
                    result, containers, result_non_iter, containers_non_iter = test_two_integrators(problem)
                except Exception as e:
                    print(f'Error during integration on {problem_name}: {e}')
                    print_exc()
                    break

                estimates[0].append(result['estimate'])
                n_evals_list[0].append(result['n_evals'])
                estimates[1].append(result_non_iter['estimate'])
                n_evals_list[1].append(result_non_iter['n_evals'])
                time_list[0].append(result['time'])
                time_list[1].append(result_non_iter['time'])

            if len(estimates[0]) == args['n_repeat']:
                new_results = []
                for i in range(2):
                    integral_estimates = np.array(estimates[i])
                    avg_estimate = np.mean(integral_estimates)
                    avg_n_evals = int(np.mean(n_evals_list[i]))
                        
                    errors = 100 * (
                        integral_estimates - problem.answer) / problem.answer
                    avg_error = f'{np.median(errors):.4f} %'
                    error_std = f'{np.std(errors):.4f} %'
                    error_name = 'Signed Relative error'

                    new_results.append({
                        'integrator': integrator_names[i],
                        'problem': problem_name,
                        'true_value': problem.answer,
                        'estimate': avg_estimate,
                        'estimate_std': np.std(estimates),
                        'error_type': error_name,
                        'error': avg_error,
                        'error_std': error_std,
                        'n_evals': avg_n_evals,
                        'n_evals_std': np.std(n_evals_list),
                        'time_taken' : np.mean(time_list[i]),
                        'errors': errors
                    })
                
                # Update the existing results
                for i in range(2):
                    existing_results[(integrator_names[i], problem_name)] = new_results[i]

                # Write results incrementally to ensure recovery
                write_results(output_file, new_results, 
                                is_first_run, fieldnames)
            is_first_run = False
            
        write_results(output_file, list(existing_results.values()), True, fieldnames, mode='w')
        print(f'Results saved to {output_file}')