from treeQuadrature.integrators import SimpleIntegrator, Integrator
from treeQuadrature.exampleProblems import ProductPeakProblem, ExponentialProductProblem, C0Problem, CornerPeakProblem, OscillatoryProblem, Problem
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.containerIntegration import AdaptiveRbfIntegral, RbfIntegral, ContainerIntegral
from treeQuadrature.samplers import McmcSampler
from treeQuadrature import Container
from treeQuadrature.compare_integrators import load_existing_results, write_results

import numpy as np
from traceback import print_exc
import os, json, time
from typing import List


args = {}
Ds = range(2, 16)

split = MinSseSplit()

args['P'] = 50
args['n_samples'] = 30
args['n_splits'] = 5
args['n_repeat'] = 10

script_dir = os.path.dirname(os.path.abspath(__file__))
location_prefix = 'ablation_residuals/'

integrator_names = ['Adaptive Rbf (mean)', 'Adaptive Rbf', 'Non Adaptive Rbf']


def test_integrals_single_problem(problem, integrals: ContainerIntegral, 
                   integrator: Integrator):
    """Test container integrators on the same tree"""
    return_values = []
    for i, integral in enumerate(integrals):
        X, y = integrator.sampler.rvs(integ.base_N, problem.lows, 
                                    problem.highs,
                                    problem.integrand)
        
        root = Container(X, y, mins=problem.lows, maxs=problem.highs)
        finished_containers = integ.construct_tree(root)

        integrator.integral = integral

        print(f"integrating {integrator_names[i]}")
        start_time = time.time()
        results, containers = integ.integrate_containers(finished_containers, problem)
        end_time = time.time()

        N = sum([cont.N for cont in containers])
        contributions = [result['integral'] for result in results]
        estimate = np.sum(contributions)
        return_values[i] = {'estimate' : estimate, 
                            'n_evals' : N,
                            'time' : end_time - start_time}
    
    return return_values


def test_container_integrals(problems: List[Problem],  integrals: ContainerIntegral, 
                             integrator: Integrator, output_file: str):
    existing_results = load_existing_results(output_file)

    if existing_results:
        is_first_run = False
    else:
        is_first_run = True
    
    final_results = []

    for problem in problems:
        problem_name = str(problem)
        print(f'testing Probelm: {problem_name}')

        key = ('Iterative GP', problem_name)
        if key in existing_results and existing_results[key]['estimate'] != '':
            print(f'Skipping {problem_name}: already completed.')
            final_results.append(existing_results[key])
            continue
        
        estimates = [[] for _ in range(n_integrals)]
        n_evals_list = [[] for _ in range(n_integrals)]
        time_list = [[] for _ in range(n_integrals)]

        for _ in range(args['n_repeat']):
            try: 
                results = test_integrals_single_problem(problem, integrals, integrator)
            except Exception as e:
                print(f'Error during integration on {problem_name}: {e}')
                print_exc()
                break

            for j in range(n_integrals):
                estimates[j].append(results[j]['estimate'])
                n_evals_list[j].append(results[j]['n_evals'])
                time_list[j].append(results[j]['time'])
            

        if len(estimates[0]) == args['n_repeat']:
            new_results = []
            for i in range(n_integrals):
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
            for i in range(n_integrals):
                final_results[(integrator_names[i], problem_name)] = new_results[i]

            # Write results incrementally to ensure recovery
            write_results(output_file, new_results, 
                            is_first_run)
        is_first_run = False
        
    write_results(output_file, list(final_results.values()), True, mode='w')
    print(f'Results saved to {output_file}')
    

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
        integral = AdaptiveRbfIntegral(n_samples= args['n_samples'], max_redraw=0, 
                                       fit_residuals=False,
                                       n_splits=0)
        integral_non_adaptive = RbfIntegral(n_samples= args['n_samples'], max_redraw=0, 
                                                n_splits=0, 
                                                range=args['range'])
        integrals = [integral_mean, integral, integral_non_adaptive]

        n_integrals = len(integrals)

        integ = SimpleIntegrator(base_N=args['N'], P=args['P'], split=split, 
                                 integral=integral_mean, 
                                 sampler=McmcSampler())
            
        test_container_integrals(problems, integrals, integ, output_file)