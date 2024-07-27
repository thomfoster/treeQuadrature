import pytest
import treeQuadrature as tq
import numpy as np

######################
# Initial IO checks
######################

@pytest.mark.parametrize("integrator_instance", [
    tq.integrators.SmcIntegrator(100),
    tq.integrators.VegasIntegrator(100, 10),
    tq.integrators.BayesMcIntegrator(100)
])
def test_io(integrator_instance):
    """Checks each integrator has the desired IO for an integrator"""

    problem = tq.exampleProblems.SimpleGaussian(3)

    res = integrator_instance(problem)
    assert isinstance(res['estimate'], float)
    res = integrator_instance(problem, return_N=True)
    assert isinstance(res['n_evals'], int)
    if hasattr(integrator_instance, 'return_std'):
        res = integrator_instance(problem, return_N=True)
        assert isinstance(res['std'], float)
        assert res['std'] >= 0

@pytest.mark.parametrize("integrator_instance", [
    tq.integrators.LimitedSampleIntegrator(
        N=500, base_N=0, active_N=100, 
        split=tq.splits.KdSplit(), integral=tq.containerIntegration.MidpointIntegral(), 
        weighting_function=lambda container: container.volume), 
    tq.integrators.QueueIntegrator(
        base_N=500, split=tq.splits.KdSplit(), integral=tq.containerIntegration.MidpointIntegral(), 
        weighting_function=lambda container: container.volume, 
        max_splits=20, stopping_condition=lambda container: container.N < 2),
    tq.integrators.SimpleIntegrator(200, 40, tq.splits.KdSplit(), tq.containerIntegration.MidpointIntegral()),
    tq.integrators.GpTreeIntegrator(200, 40, tq.splits.KdSplit(), 
                                    tq.containerIntegration.RbfIntegral(n_splits=0), 
                                    grid_size=0.1)
])
def test_treeIntegrator_io(integrator_instance):
    problem = tq.exampleProblems.SimpleGaussian(1)

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

########################################################################
# Checking all combos of inputs for simple and queue based integrators
########################################################################

splits = [
    tq.splits.KdSplit(),
    tq.splits.MinSseSplit(),
    tq.splits.UniformSplit()
]

integrals = [
    tq.containerIntegration.MedianIntegral(),
    tq.containerIntegration.MidpointIntegral(),
    tq.containerIntegration.RandomIntegral(),
    tq.containerIntegration.SmcIntegral(),
    tq.containerIntegration.RbfIntegral(n_samples=5, n_tuning=1, max_iter=100,
                                        max_redraw=1, n_splits=0)
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
    # UniformSplit generates lot of empty containers, not suitable for MedianIntegral
    if "MedianIntegral" in str(integral) and "UniformSplit" in str(split):
        return
    
    problem = tq.exampleProblems.SimpleGaussian(D)
    integ = tq.integrators.SimpleIntegrator(N, P, split, integral)
    _ = integ(problem)

@pytest.mark.parametrize("D", [1,2])
@pytest.mark.parametrize("base_N", [1000])
@pytest.mark.parametrize("split", splits)
@pytest.mark.parametrize("integral", integrals)
@pytest.mark.parametrize("weighting_function", [lambda container: container.volume])
@pytest.mark.parametrize("active_N", [0, 10])
@pytest.mark.parametrize("max_splits", [np.inf, 50])
@pytest.mark.parametrize("stopping_condition", [lambda container: container.N < 2])
@pytest.mark.parametrize("queue", queues)
def test_QueueIntegrator(
    D, base_N, split, integral, weighting_function,
    active_N, max_splits, stopping_condition, queue):

    # too slow for testing, too many containers
    if "RbfIntegral" in str(integral):
        return

    if "MedianIntegral" in str(integral) and "UniformSplit" in str(split):
        return

    if np.isinf(max_splits) and active_N > 0:
        return

    problem = tq.exampleProblems.SimpleGaussian(D)
    integ = tq.integrators.QueueIntegrator(
        base_N=base_N, split=split, integral=integral, 
        weighting_function=weighting_function,
        active_N=active_N, max_splits=max_splits, 
        stopping_condition=stopping_condition, 
        queue=queue)
    res = integ(problem, return_N=True, return_containers=True)
    fcs = res['containers']
    ns = res['n_splits']
    N = res['n_evals']

    if "UniformSplit" in str(split):
        n_sub_splits = 2 ** D
    else:
        n_sub_splits = 2

    if "RandomIntegral" in str(integral) or "SmcIntegral" in str(integral): 
        # accounts for random samples used in container integration
        assert base_N + ns*n_sub_splits*active_N + len(fcs)*integral.n == N
    else:
        assert base_N + ns*n_sub_splits*active_N == N

    if not np.isinf(max_splits):
        assert ns == max_splits

@pytest.mark.parametrize("D", [1,2])
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("base_N, active_N", [(0,100), (500,10), (1000, 100)])
@pytest.mark.parametrize("split", splits)
@pytest.mark.parametrize("integral", integrals)
@pytest.mark.parametrize("weighting_function", [lambda container: container.volume])
@pytest.mark.parametrize("queue", queues)
def test_LimitedSampleIntegrator(
    D, N, base_N, active_N, split, integral, weighting_function, queue
    ):

    # too slow for testing, too many containers
    if "RbfIntegral" in str(integral):
        return

    if "MedianIntegral" in str(integral) and "UniformSplit" in str(split):
        return

    problem = tq.exampleProblems.SimpleGaussian(D)
    integ = tq.integrators.LimitedSampleIntegrator(
        N=N, base_N=base_N, 
        active_N=active_N, split=split, 
        integral=integral, weighting_function=weighting_function, 
        queue=queue
    )
    _ = integ(problem)