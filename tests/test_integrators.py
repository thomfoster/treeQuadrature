import pytest
import treeQuadrature as tq
import numpy as np

######################
# Initial IO checks
######################

@pytest.mark.parametrize("integrator_instance", [
    tq.integrators.SmcIntegrator(100),
    tq.integrators.VegasIntegrator(100, 10),
    tq.integrators.BayesMcIntegrator(100),
    tq.integrators.ISTreeIntegrator(
        100,
        tree=tq.trees.SimpleTree(split=tq.splits.KdSplit())),
])
def test_io(integrator_instance):
    """Checks each integrator has the desired IO for an integrator"""

    problem = tq.example_problems.SimpleGaussian(3)

    res = integrator_instance(problem)
    assert isinstance(res['estimate'], float)
    res = integrator_instance(problem, return_N=True)
    assert isinstance(res['n_evals'], int)
    if hasattr(integrator_instance, 'return_std'):
        res = integrator_instance(problem, return_N=True)
        assert isinstance(res['std'], float)
        assert res['std'] >= 0

@pytest.mark.parametrize("integrator_instance", [
    tq.integrators.TreeIntegrator(
        base_N=0, integral=tq.container_integrators.MidpointIntegral(),
        tree=tq.trees.LimitedSampleTree(
            N=500, active_N=100,
            split=tq.splits.KdSplit(),
            weighting_function=lambda container: container.volume)),
    tq.integrators.TreeIntegrator(
        base_N=500, tree=tq.trees.WeightedTree(
            split=tq.splits.KdSplit(),
            weighting_function=lambda container: container.volume, 
            max_splits=20, 
            stopping_condition=lambda container: container.N < 2),
        integral=tq.container_integrators.MidpointIntegral()),
    tq.integrators.TreeIntegrator(
        200, tree=tq.trees.SimpleTree(split=tq.splits.KdSplit()),
        integral=tq.container_integrators.MidpointIntegral()),
    tq.integrators.BatchGpIntegrator(
        200, tree=tq.trees.SimpleTree(split=tq.splits.KdSplit()),
        integral=tq.container_integrators.KernelIntegral(n_splits=0)),
    tq.integrators.VegasTreeIntegrator(200)
])
def test_treeIntegrator_io(integrator_instance):
    problem = tq.example_problems.SimpleGaussian(1)

    res = integrator_instance(problem)
    assert isinstance(res['estimate'], float)
    res = integrator_instance(problem, return_N=True)
    assert len(res) >= 2
    assert isinstance(res['n_evals'], int)
    res = integrator_instance(problem, return_containers=True)
    assert len(res) == 3
    assert all(isinstance(val, float) for val in res['contributions'])
    assert all(isinstance(cont, tq.Container) for cont in res['containers'])
    assert len(res['contributions']) == len(res['containers'])

@pytest.mark.parametrize("integrator_instance", [
    tq.integrators.TreeIntegrator(
        300, integral=tq.container_integrators.KernelIntegral(
            n_splits=0)),
    tq.integrators.TreeIntegrator(
        300, integral=tq.container_integrators.AdaptiveRbfIntegral(
            n_splits=0)),
    tq.integrators.TreeIntegrator(
        300, integral=tq.container_integrators.MedianIntegral()),
    tq.integrators.TreeIntegrator(
        1000, integral=tq.container_integrators.RandomIntegral()),
    tq.integrators.BayesMcIntegrator(200),
    tq.integrators.SmcIntegrator(300),
    tq.integrators.VegasTreeIntegrator(300),
    tq.integrators.BatchGpIntegrator(
        300, tree=tq.trees.SimpleTree(split=tq.splits.KdSplit()),
        integral=tq.container_integrators.KernelIntegral(n_splits=0))
])
def test_return_std(integrator_instance):
    problem = tq.example_problems.SimpleGaussian(1)

    res = integrator_instance(problem, return_std=True)
    if "TreeIntegrator" in str(integrator_instance) or (
        "BatchGpIntegrator" in str(integrator_instance)
    ):
        for std in res['stds']:
            assert isinstance(std, float)
            assert std >= 0
    else:
        assert isinstance(res['std'], float)
        assert res['std'] >= 0

########################################################################
# Checking all combos of inputs for simple and queue based integrators
########################################################################

splits = [
    tq.splits.KdSplit(),
    tq.splits.MinSseSplit(),
    tq.splits.UniformSplit()
]

integrals = [
    tq.container_integrators.MedianIntegral(),
    tq.container_integrators.MidpointIntegral(),
    tq.container_integrators.RandomIntegral(),
    tq.container_integrators.RandomIntegral(eval=np.median),
    tq.container_integrators.KernelIntegral(n_samples=5, n_tuning=1, 
                                        max_iter=100,
                                        max_redraw=0, n_splits=0)
]

queues = [
    tq.queues.ReservoirQueue(),
    tq.queues.PriorityQueue()
]

@pytest.mark.parametrize("D", [1,2,3])
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("P", [100])
@pytest.mark.parametrize("split", splits)
@pytest.mark.parametrize("integral", integrals)
def test_SimpleIntegrator(D, N, P, split, integral):
    # UniformSplit generates lot of empty containers,
    # not suitable for MedianIntegral
    if "MedianIntegral" in str(integral) and (
        "UniformSplit" in str(split)):
        return
    
    problem = tq.example_problems.SimpleGaussian(D)
    integ = tq.integrators.TreeIntegrator(
        N, tq.trees.SimpleTree(P=P, split=split), 
        integral=integral)
    _ = integ(problem)

@pytest.mark.parametrize("D", [1,2])
@pytest.mark.parametrize("base_N", [1000])
@pytest.mark.parametrize("split", splits)
@pytest.mark.parametrize("integral", integrals)
@pytest.mark.parametrize("weighting_function",
                         [lambda container: container.volume])
@pytest.mark.parametrize("active_N", [0, 10])
@pytest.mark.parametrize("max_splits", [np.inf, 50])
@pytest.mark.parametrize("stopping_condition",
                         [lambda container: container.N < 2])
@pytest.mark.parametrize("queue", queues)
def test_QueueIntegrator(
    D, base_N, split, integral, weighting_function,
    active_N, max_splits, stopping_condition, queue):

    # too slow for testing, too many containers
    if "KernelIntegral" in str(integral):
        return

    if "MedianIntegral" in str(integral) and "UniformSplit" in str(split):
        return

    if np.isinf(max_splits) and active_N > 0:
        return

    problem = tq.example_problems.SimpleGaussian(D)
    tree = tq.trees.WeightedTree(split=split, 
        weighting_function=weighting_function,
        active_N=active_N, max_splits=max_splits, 
        stopping_condition=stopping_condition, 
        queue=queue)
    integ = tq.integrators.TreeIntegrator(
        base_N=base_N, integral=integral, 
        tree=tree)
    res = integ(problem, return_N=True, return_containers=True)
    fcs = res['containers']
    ns = res['n_splits']
    N = res['n_evals']

    if "UniformSplit" in str(split):
        n_sub_splits = 2 ** D
    else:
        n_sub_splits = 2

    if "RandomIntegral" in str(integral): 
        # accounts for random samples used in container integration
        assert base_N + ns*n_sub_splits*active_N + \
            len(fcs)*integral.n_samples == N
    else:
        assert base_N + ns*n_sub_splits*active_N == N

    if not np.isinf(max_splits):
        assert ns == max_splits

@pytest.mark.parametrize("D", [1,2])
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("base_N, active_N",
                         [(0,100), (500,10), (1000, 100)])
@pytest.mark.parametrize("split", splits)
@pytest.mark.parametrize("integral", integrals)
@pytest.mark.parametrize("weighting_function",
                         [lambda container: container.volume])
@pytest.mark.parametrize("queue", queues)
def test_LimitedSampleIntegrator(
    D, N, base_N, active_N, split, integral, weighting_function, queue
    ):

    # too slow for testing, too many containers
    if "KernelIntegral" in str(integral):
        return

    if "MedianIntegral" in str(integral) and "UniformSplit" in str(split):
        return

    problem = tq.example_problems.SimpleGaussian(D)
    active_tree = tq.trees.LimitedSampleTree(
        N=N, active_N=active_N, 
        split=split, queue=queue,
        weighting_function=weighting_function)
    integ = tq.integrators.TreeIntegrator(
        base_N=base_N, 
        integral=integral, 
        tree=active_tree)
    _ = integ(problem)

@pytest.mark.parametrize("D", [1,2])
@pytest.mark.parametrize("max_samples", [15000, 20000])
@pytest.mark.parametrize("split", splits)
@pytest.mark.parametrize("integral", integrals)
def test_DistributedTreeIntegrator(D, max_samples, split, integral):
    problem = tq.example_problems.SimpleGaussian(D)
    integ = tq.integrators.DistributedTreeIntegrator(
        7500, max_samples, integral, max_container_samples=50)
    
    # UniformSplit generates too many containers
    if "UniformSplit" in str(split):
        return 

    # cannot control n_samples
    if "MedianIntegral" in str(integral) or (
        "MidpointIntegral" in str(integral)):
        return
    
    res = integ(problem, return_N=True)
    ## should not distribute more samples than max_samples
    assert res['n_evals'] <= max_samples