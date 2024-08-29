from curses.ascii import SI
from treeQuadrature.exampleProblems import RippleProblem, SimpleGaussian, Camel, QuadraticProblem, C0Problem, OscillatoryProblem, CornerPeakProblem, ProductPeakProblem, ExponentialProductProblem, QuadCamel
from treeQuadrature.integrators import BatchGpIntegrator, DistributedGpTreeIntegrator, SmcIntegrator, DistributedTreeIntegrator, VegasIntegrator
from treeQuadrature.containerIntegration import RandomIntegral, KernelIntegral, AdaptiveRbfIntegral, PolyIntegral, IterativeRbfIntegral
from treeQuadrature.integrators.treeIntegrator import TreeIntegrator
from treeQuadrature.splits import MinSseSplit, KdSplit
from treeQuadrature.samplers import ImportanceSampler, UniformSampler, McmcSampler, SobolSampler, LHSImportanceSampler
from treeQuadrature import compare_integrators, Container
from treeQuadrature.trees import SimpleTree, LimitedSampleTree

import numpy as np

D = 2

### Set problem
# problem = Camel(D=D)
problem = SimpleGaussian(D=D)
# problem = RippleProblem(D=D)
# problem = QuadraticProblem(D=D)
# problem = OscillatoryProblem(D, a=np.array(10 / np.linspace(1, D, D)))
# problem = C0Problem(D, np.array([1.1] * D))
# problem = CornerPeakProblem(D=D, a=np.array([10]*D))
# problem = ProductPeakProblem(D=D, a=np.array([10]*D))
# problem = ExponentialProductProblem(D)

### set basic parameters
n_samples = 20
max_redraw = 4
max_n_samples = int(30_000 * (D/2))
min_container_samples = 32
random_split_proportion = 0.3

N = int(12_000 * (D/3))
P = 50

# for activeTQ
lsi_N = 10000 + 1000*D
active_N = 10

### Set ContainerIntegral
rbfIntegral_mean = KernelIntegral(max_redraw=max_redraw, threshold=0.9, n_splits=3, 
                               n_samples=n_samples)
rbfIntegral = KernelIntegral(max_redraw=max_redraw, threshold=0.9, n_splits=3, 
                            fit_residuals=False, 
                            n_samples=n_samples)
rbfIntegral_non_iter = KernelIntegral(max_redraw=0, n_splits=0, 
                                   n_samples=n_samples)

aRbf = AdaptiveRbfIntegral(n_samples=n_samples, max_redraw=0, n_splits=0, keep_samples=False)
aRbf_iniital = AdaptiveRbfIntegral(n_samples= n_samples, max_redraw=0, n_splits=0, keep_samples=True)
aRbf_qmc = AdaptiveRbfIntegral(n_samples= n_samples, max_redraw=0, n_splits=0, keep_samples=False,
                               sampler=SobolSampler())

iRbf = IterativeRbfIntegral(n_samples=n_samples, n_splits=0)
rmeanIntegral = RandomIntegral(n_samples=n_samples)
polyIntegral = PolyIntegral(n_samples=n_samples, degrees=[2, 3], max_redraw=0)

### set Sampler
iSampler = ImportanceSampler()
uSampler = UniformSampler()    
mcmcSampler = McmcSampler()
sobolSampler = SobolSampler()
lhsSampler = LHSImportanceSampler()

### set split
def side_lengths(container: Container):
    return container.maxs - container.mins

split = MinSseSplit()
split_random = MinSseSplit(dimension_weights=side_lengths,
                    dimension_proportion=random_split_proportion)
split_kd = KdSplit()

### set trees
simple_tree = SimpleTree(split=split, P=P)

random_tree = SimpleTree(split=split_random, P=P)

active_tree = LimitedSampleTree(N=lsi_N, active_N=active_N, split=split, weighting_function=lambda container: container.volume)

### Set integrators
integ_mean_uncontrolled = TreeIntegrator(N, tree=simple_tree, integral=rmeanIntegral, sampler=mcmcSampler)
integ_mean_uncontrolled.name = 'TQ with mean (unlimited)'

integ_is = TreeIntegrator(N, tree=simple_tree, integral=aRbf, sampler=iSampler)
integ_is.name = 'Importance sampler'

integ_unif = TreeIntegrator(N, tree=simple_tree, integral=aRbf, sampler=mcmcSampler)
integ_unif.name = 'MCMC sampler'

integ_poly = TreeIntegrator(N, tree=simple_tree, integral=polyIntegral, sampler=mcmcSampler)
integ_poly.name = 'TQ with polynomial'

integ_mean = DistributedTreeIntegrator(N, max_n_samples=max_n_samples, 
                                             integral=rmeanIntegral, sampler=mcmcSampler, tree=simple_tree)
integ_mean.name = 'TQ with mean'

integ_mean_random_split = DistributedTreeIntegrator(N, max_n_samples=max_n_samples, 
                                             integral=rmeanIntegral, sampler=mcmcSampler, tree=random_tree)
integ_mean_random_split.name = 'TQ with mean and random splitting'


integ_activeTQ = DistributedTreeIntegrator(N, max_n_samples=max_n_samples, 
                                             integral=rmeanIntegral, sampler=mcmcSampler, tree=active_tree)
integ_activeTQ.name = 'ActiveTQ'

integ_limitedGp = DistributedGpTreeIntegrator(N, max_n_samples=max_n_samples,
                                            integral=iRbf, sampler=lhsSampler, tree=simple_tree,
                                            max_container_samples=100)
integ_limitedGp.name = 'Distributed Rbf Integrator'



integ_rbf = DistributedTreeIntegrator(N, max_n_samples=max_n_samples, 
                                             integral=aRbf, sampler=mcmcSampler, tree=simple_tree, 
                                             min_container_samples=min_container_samples)
integ_rbf.name = 'TQ with RBF'

integ_rbf_qmc = DistributedTreeIntegrator(N, max_n_samples=max_n_samples, 
                                             integral=aRbf_qmc, sampler=mcmcSampler, tree=simple_tree, 
                                             min_container_samples=min_container_samples)
integ_rbf_qmc.name = 'TQ with RBF with QMC container sampler'

integ_rbf_initial = DistributedTreeIntegrator(N, max_n_samples=max_n_samples, 
                                             integral=aRbf_iniital, sampler=mcmcSampler, tree=simple_tree, 
                                             min_container_samples=min_container_samples)
integ_rbf_initial.name = 'TQ with RBF, keeping initial samples'

integ_rbf_non_adaptive = TreeIntegrator(N, tree=simple_tree, integral=rbfIntegral_mean, sampler=mcmcSampler)
integ_rbf_non_adaptive.name = 'TQ with non-adaptive RBF'

integ_batch = BatchGpIntegrator(N, P, split, rbfIntegral_non_iter, 
                               sampler = lhsSampler, max_n_samples=max_n_samples)
integ_batch.name = 'Batch GP'

integ_smc = SmcIntegrator(N=max_n_samples, sampler=UniformSampler())
integ_smc.name = 'SMC'

n_iter = 10
adaptive_iter = 0
vegas_n = int(max_n_samples / (n_iter + adaptive_iter))
integ_vegas = VegasIntegrator(vegas_n, n_iter, adaptive_iter)
integ_vegas.name = 'Vegas'

adaptive_iter = 5
vegas_n = int(max_n_samples / (n_iter + adaptive_iter))
integ_vegas_adaptive = VegasIntegrator(vegas_n, n_iter, adaptive_iter)
integ_vegas_adaptive.name = 'Adaptive Vegas'    

if __name__ == '__main__':
    print(f"maximum allowed samples: {max_n_samples}")
    compare_integrators([integ_activeTQ], plot=True, verbose=1,
                        xlim=[problem.lows[0], problem.highs[0]], 
                        ylim=[problem.lows[1], problem.highs[1]], 
                        problem=problem, dimensions=[0, 1], 
                        n_repeat=1, integrator_specific_kwargs={
                            'ActiveTQ': {'integrand' : problem.integrand, 
                                                        'max_iter' : 300}}, 
                            plot_samples=False, title='')
    # compare_integrators([integ_rbf_non_adaptive], plot=True, verbose=1,
    #                     xlim=[problem.lows[0], problem.highs[0]], 
    #                     problem=problem, dimensions=[0, 1], 
    #                     n_repeat=1, integrator_specific_kwargs={
    #                         'ActiveTQ': {'integrand' : problem.integrand}}, 
    #                         plot_samples=False, font_size=13)