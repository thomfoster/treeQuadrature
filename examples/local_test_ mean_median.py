from treeQuadrature.exampleProblems import Camel
from treeQuadrature.containerIntegration import RandomIntegral, RandomIntegral
from treeQuadrature.splits import MinSseSplit, KdSplit
from treeQuadrature.integrators import LimitedSampleIntegrator
from treeQuadrature.compare_integrators import test_integrators

import os

Ds = range(1, 8)

problems = [Camel(D) for D in Ds]

medianIntegral = RandomIntegral()
meanIntegral = RandomIntegral()
split = KdSplit()

integ_median = LimitedSampleIntegrator(N=2000, base_N=1000, 
                                 active_N=10, 
                                 split=split, integral=medianIntegral, 
                                 weighting_function=lambda container: container.volume)
integ_median.name = 'active TQ with median'

integ_mean = LimitedSampleIntegrator(N=2000, base_N=1000, 
                                 active_N=10, 
                                 split=split, integral=meanIntegral, 
                                 weighting_function=lambda container: container.volume)
integ_mean.name = 'active TQ with mean'

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 
                           f"../test_results/kdSplit_mean_median_compare.csv")

if __name__ == '__main__':
    test_integrators([integ_median, integ_mean],
                    problems=problems, 
                    output_file=output_path, n_repeat=10) 