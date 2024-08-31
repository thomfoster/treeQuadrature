from treeQuadrature.example_problems import (
    Ripple, SimpleGaussian, Camel, 
    Quadratic, C0, Oscillatory, CornerPeak, 
    ProductPeak, ExponentialProduct, QuadCamel
)
from treeQuadrature.integrators import (
    BatchGpIntegrator, DistributedGpTreeIntegrator, SmcIntegrator, 
    DistributedTreeIntegrator, VegasIntegrator
)
from treeQuadrature.container_integrators import (
    RandomIntegral, KernelIntegral, AdaptiveRbfIntegral, 
    PolyIntegral, IterativeRbfIntegral
)
from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.splits import MinSseSplit, KdSplit, UniformSplit
from treeQuadrature.samplers import (
    ImportanceSampler, UniformSampler, McmcSampler,
    SobolSampler, LHSImportanceSampler
)
from treeQuadrature import compare_integrators, Container
from treeQuadrature.trees import SimpleTree, LimitedSampleTree

import numpy as np

D = 2

### Set problem
problem = Camel(D=D)
# problem = SimpleGaussian(D=D)
# problem = Ripple(D=D)
# problem = Quadratic(D=D)
# problem = Oscillatory(D, a=np.array(10 / np.linspace(1, D, D)))
# problem = C0(D, np.array([1.1] * D))
# problem = CornerPeak(D=D, a=np.array([10]*D))
# problem = ProductPeak(D=D, a=np.array([10]*D))
# problem = ExponentialProduct(D)

### set basic parameters
n_samples = 20
max_redraw = 4
max_n_samples = int(20_000 * (D/2))
min_container_samples = 32
random_split_proportion = 0.5
batch_gp_grid_size = 0.03

N = int(12_000 * (D/3))
P = 50

# for activeTQ
lsi_N = 10000
active_N = 10
max_iter = 1000 + 100 * D

### Set ContainerIntegral
rbfIntegral_mean = KernelIntegral(max_redraw=max_redraw, threshold=0.9, n_splits=3, 
                               n_samples=n_samples)
rbfIntegral = KernelIntegral(max_redraw=max_redraw, threshold=0.9, n_splits=3, 
                            fit_residuals=False, 
                            n_samples=n_samples)
rbfIntegral_non_iter = KernelIntegral(max_redraw=0, n_splits=0, 
                                   n_samples=n_samples)

aRbf = AdaptiveRbfIntegral(
    n_samples=n_samples, max_redraw=0, n_splits=0, keep_samples=False)
aRbf_iniital = AdaptiveRbfIntegral(
    n_samples= n_samples, max_redraw=0, n_splits=0, keep_samples=True)
aRbf_qmc = AdaptiveRbfIntegral(
    n_samples= n_samples, max_redraw=0, n_splits=0, keep_samples=False,
    sampler=SobolSampler())

iRbf = IterativeRbfIntegral(n_samples=n_samples, n_splits=0)
rmeanIntegral = RandomIntegral(n_samples=n_samples)
polyIntegral = PolyIntegral(n_samples=n_samples, degrees=[2, 3], max_redraw=0)

# ===========
# set sampler
# ===========
iSampler = ImportanceSampler()
uSampler = UniformSampler()    
mcmcSampler = McmcSampler()
sobolSampler = SobolSampler()
lhsSampler = LHSImportanceSampler()

# =========
# set split
# =========
def side_lengths(container: Container):
    return container.maxs - container.mins

split = MinSseSplit()
split_random = MinSseSplit(dimension_weights=side_lengths,
                    dimension_proportion=random_split_proportion)
split_kd = KdSplit()
split_uniform = UniformSplit()

# ===========
# set Tree
# ===========
tree_simple = SimpleTree(split=split, P=P)

tree_random = SimpleTree(split=split_random, P=P)

tree_kd = SimpleTree(split=split_kd, P=P)

tree_uniform = SimpleTree(split=split_uniform, P=P)

tree_active = LimitedSampleTree(
    N=lsi_N, active_N=active_N, split=split,
    weighting_function=lambda container: container.volume)

### Set integrators
integ_mean_minsse = TreeIntegrator(
    N, tree=tree_simple, integral=rmeanIntegral, sampler=mcmcSampler)
integ_mean_minsse.name = 'TQ with mean (Min-SSE Split)'

integ_mean_kd = TreeIntegrator(
    N, tree=tree_kd, integral=rmeanIntegral, sampler=mcmcSampler)
integ_mean_kd.name = 'TQ with mean (KD Split)'

integ_mean_uniform = TreeIntegrator(
    N, tree=tree_uniform, integral=rmeanIntegral, sampler=mcmcSampler)
integ_mean_uniform.name = 'TQ with mean (Uniform Split)'

integ_is = TreeIntegrator(
    N, tree=tree_simple, integral=aRbf, sampler=iSampler)
integ_is.name = 'Importance sampler'

integ_mcmc = TreeIntegrator(
    N, tree=tree_simple, integral=aRbf, sampler=mcmcSampler)
integ_mcmc.name = 'MCMC sampler'

integ_poly = TreeIntegrator(
    N, tree=tree_simple, integral=polyIntegral, sampler=mcmcSampler)
integ_poly.name = 'TQ with polynomial'

integ_mean = DistributedTreeIntegrator(
    N, max_n_samples=max_n_samples,
    integral=rmeanIntegral, sampler=mcmcSampler,
    tree=tree_simple)
integ_mean.name = 'TQ with mean'

integ_mean_random_split = DistributedTreeIntegrator(
    N, max_n_samples=max_n_samples,
    integral=rmeanIntegral, sampler=mcmcSampler,
    tree=tree_random)
integ_mean_random_split.name = 'TQ with mean and random splitting'


integ_activeTQ = DistributedTreeIntegrator(
    N, max_n_samples=max_n_samples,
    integral=rmeanIntegral, sampler=mcmcSampler,
    tree=tree_active)
integ_activeTQ.name = 'ActiveTQ'

integ_limitedGp = DistributedGpTreeIntegrator(
    N, max_n_samples=max_n_samples,
    integral=iRbf, sampler=lhsSampler,
    tree=tree_simple,
    max_container_samples=100)
integ_limitedGp.name = 'Distributed Rbf Integrator'


integ_rbf = DistributedTreeIntegrator(
    N, max_n_samples=max_n_samples,
    integral=aRbf, sampler=mcmcSampler,
    tree=tree_simple,
    min_container_samples=min_container_samples)
integ_rbf.name = 'TQ with RBF'

integ_rbf_qmc = DistributedTreeIntegrator(
    N, max_n_samples=max_n_samples,
    integral=aRbf_qmc, sampler=mcmcSampler,
    tree=tree_simple,
    min_container_samples=min_container_samples)
integ_rbf_qmc.name = 'TQ with RBF with QMC container sampler'

integ_rbf_initial = DistributedTreeIntegrator(
    N, max_n_samples=max_n_samples,
    integral=aRbf_iniital, sampler=mcmcSampler,
    tree=tree_simple,
    min_container_samples=min_container_samples)
integ_rbf_initial.name = 'TQ with RBF, keeping initial samples'

integ_rbf_non_adaptive = TreeIntegrator(
    N, tree=tree_simple,
    integral=rbfIntegral_mean,
    sampler=mcmcSampler)
integ_rbf_non_adaptive.name = 'TQ with non-adaptive RBF'

integ_batch = BatchGpIntegrator(
    N, rbfIntegral_non_iter, tree=tree_simple,
    sampler = mcmcSampler, max_n_samples=max_n_samples,
    base_grid_scale=batch_gp_grid_size)
integ_batch.name = 'Batch GP'

integ_smc = SmcIntegrator(N=max_n_samples)
integ_smc.name = 'SMC'

n_iter = 10
adaptive_iter = 0
vegas_n = int(max_n_samples / (n_iter + adaptive_iter))
integ_vegas = VegasIntegrator(
    vegas_n, n_iter, adaptive_iter)
integ_vegas.name = 'Vegas'

adaptive_iter = 5
vegas_n = int(max_n_samples / (n_iter + adaptive_iter))
integ_vegas_adaptive = VegasIntegrator(
    vegas_n, n_iter, adaptive_iter)
integ_vegas_adaptive.name = 'Adaptive Vegas'    

if __name__ == '__main__':
    print(f"maximum allowed samples: {max_n_samples}")
    compare_integrators([integ_mean_random_split, integ_mean_minsse], plot=True, verbose=1,
                        xlim=[problem.lows[0], problem.highs[0]], 
                        ylim=[problem.lows[1], problem.highs[1]], 
                        problem=problem, dimensions=[0, 1], integrator_specific_kwargs=
                        {'ActiveTQ': {'max_iter' : max_iter}},
                        n_repeat=1, plot_samples=True, title='')
    # compare_integrators([integ_rbf_non_adaptive], plot=True, verbose=1,
    #                     xlim=[problem.lows[0], problem.highs[0]], 
    #                     problem=problem, dimensions=[0, 1], 
    #                     n_repeat=1, integrator_specific_kwargs={
    #                         'ActiveTQ': {'max_iter' : max_iter}}, 
    #                         plot_samples=False, font_size=13)