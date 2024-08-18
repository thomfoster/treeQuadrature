from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.exampleProblems import ProductPeakProblem, ExponentialProductProblem, C0Problem, CornerPeakProblem, OscillatoryProblem
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral
from treeQuadrature.samplers import LHSImportanceSampler
from treeQuadrature import Container
from treeQuadrature.compare_integrators import load_existing_results, write_results

import numpy as np
from traceback import print_exc
import os


args = {}
Ds = range(1, 16)

split = MinSseSplit()

args['P'] = 50
args['n_samples'] = 15
args['max_redraw'] = 5
args['n_splits'] = 5
args['n_repeat'] = 5

script_dir = os.path.dirname(os.path.abspath(__file__))
location_prefix = 'ablation_iterative_fitting/'

integrator_names = ['Iterative GP', 'Even Sample GP']

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
                                f"../test_results/{location_prefix}results_{D}D_{args.n_repeat}repeat.csv")

        args['N'] = max(int(10000 * D / 3), 7500)
        # threshold on R2 score
        args['threshold'] = max(1 - D / 10, 0.3)

        integral = AdaptiveRbfIntegral(n_samples=args['n_samples'], 
                                                max_redraw = args['max_redraw'],
                                                n_splits=args['n_splits'], 
                                                threshold=args['threshold'])
        integral_non_iter = AdaptiveRbfIntegral(n_samples= args['n_samples'], max_redraw=0, 
                                                n_splits=args['n_splits'])
        integ = SimpleIntegrator(args['N'], args['P'], split, integral, 
                                 sampler=LHSImportanceSampler())


        def test_two_integrators(problem):
            ### integrate iteratively 
            X, y = integ.sampler.rvs(integ.base_N, problem.lows, problem.highs,
                                        problem.integrand)
            
            root = Container(X, y, mins=problem.lows, maxs=problem.highs)
            finished_containers = integ.construct_tree(root)

            results, containers = integ.integrate_containers(finished_containers)

            N = sum([cont.N for cont in containers])

            ### distribute these samples evenly to non-iterative integrator
            N_per_container = int(N / len(finished_containers))

            integral_non_iter.n_samples = N_per_container
            integ.integral = integral_non_iter

            results_non_iter, containers_non_iter = integ.integrate_containers(finished_containers)

            contributions = [result['integral'] for result in results]
            estimate = np.sum(contributions)
            return_values = {'estimate' : estimate}
            return_values['n_evals'] = N

            contributions_non_iter = [result['integral'] for result in results_non_iter]
            estimate_non_iter = np.sum(contributions_non_iter)
            return_values_non_iter = {'estimate' : estimate_non_iter}
            return_values_non_iter['n_evals'] = sum([cont.N for cont in containers_non_iter])
            
            return return_values, containers, return_values_non_iter, containers_non_iter
            
        existing_results = load_existing_results(output_file)

        if existing_results:
            is_first_run = False
        else:
            is_first_run = True
        
        results = []

        for problem in problems:
            problem_name = str(problem)

            key = ('Iterative GP', problem_name)
            if key in existing_results and existing_results[key]['estimate'] != '':
                print(f'Skipping {problem_name}: already completed.')
                results.append(existing_results[key])
                continue
            
            estimates = [[], []]
            n_evals_list = [[], []]

            for _ in args['n_repeat']:
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
                        'errors': errors
                    })
                
            # Update the existing results
            for i in range(2):
                existing_results[(integrator_names[i], problem_name)] = new_results[i]

            # Write results incrementally to ensure recovery
            write_results(output_file, new_results, 
                            is_first_run)
            is_first_run = False
            
        write_results(output_file, list(existing_results.values()), True, mode='w')
