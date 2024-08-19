from treeQuadrature.integrators import SimpleIntegrator
from treeQuadrature.exampleProblems import ProductPeakProblem, ExponentialProductProblem, C0Problem, CornerPeakProblem, OscillatoryProblem
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral, RbfIntegral
from treeQuadrature.samplers import McmcSampler
from treeQuadrature import Container
from treeQuadrature.compare_integrators import load_existing_results, write_results

import numpy as np
from traceback import print_exc
import os, json, time


args = {}
Ds = range(2, 16)

split = MinSseSplit()

args['P'] = 50
args['n_samples'] = 30
args['range'] = 500
args['n_splits'] = 5
args['n_repeat'] = 10

script_dir = os.path.dirname(os.path.abspath(__file__))
location_prefix = 'ablation_adaptive/'

integrator_names = ['Adaptive Rbf', 'Rbf']

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

        integral_adaptive = AdaptiveRbfIntegral(n_samples=args['n_samples'], 
                                                max_redraw = 0,
                                                n_splits=args['n_splits'])
        integral_rbf = RbfIntegral(n_samples= args['n_samples'], max_redraw=0, 
                                                n_splits=args['n_splits'], 
                                                range=args['range'])
        integ = SimpleIntegrator(base_N=args['N'], P=args['P'], split=split, 
                                 integral=integral_adaptive, 
                                 sampler=McmcSampler())


        def test_two_integrators(problem):
            ### Adaptive Rbf
            X, y = integ.sampler.rvs(integ.base_N, problem.lows, problem.highs,
                                        problem.integrand)
            
            root = Container(X, y, mins=problem.lows, maxs=problem.highs)
            finished_containers = integ.construct_tree(root)

            print(f"integrating {integrator_names[0]}")
            start_time = time.time()
            results_0, containers_0 = integ.integrate_containers(finished_containers, problem)
            end_time = time.time()

            N = sum([cont.N for cont in containers_0])
            contributions = [result['integral'] for result in results_0]
            estimate = np.sum(contributions)
            return_values_0 = {'estimate' : estimate}
            return_values_0['n_evals'] = N
            return_values_0['time'] = end_time - start_time

            ### Usual Rbf
            integ.integral = integral_rbf

            print(f"integrating {integrator_names[1]}")
            start_time = time.time()
            results_1, containers_1 = integ.integrate_containers(finished_containers, problem)
            end_time = time.time()
            
            contributions = [result['integral'] for result in results_1]
            estimate = np.sum(containers_1)
            return_values_1 = {'estimate' : estimate}
            N = sum([cont.N for cont in contributions])
            return_values_1['n_evals'] = N
            return_values_1['time'] = end_time - start_time
            
            return return_values_0, return_values_1
            
        existing_results = load_existing_results(output_file)

        if existing_results:
            is_first_run = False
        else:
            is_first_run = True
        
        results = []

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
                    result_0, result_1 = test_two_integrators(problem)
                except Exception as e:
                    print(f'Error during integration on {problem_name}: {e}')
                    print_exc()
                    break

                estimates[0].append(result_0['estimate'])
                n_evals_list[0].append(result_0['n_evals'])
                time_list[0].append(result_0['time'])
                estimates[1].append(result_1['estimate'])
                n_evals_list[1].append(result_1['n_evals'])
                time_list[1].append(result_1['time'])
                

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
                                is_first_run)
            is_first_run = False
            
        write_results(output_file, list(existing_results.values()), True, mode='w')
        print(f'Results saved to {output_file}')