import pytest
import treeQuadrature as tq
import numpy as np
from typing import List

######################
# Initial IO checks
######################

@pytest.mark.parametrize("integrator_instance", [
    tq.integrators.SmcIntegrator(100),
    tq.integrators.VegasIntegrator(100, 10),
    tq.integrators.SimpleIntegrator(100, 50, tq.splits.KdSplit(), tq.containerIntegration.MidpointIntegral()),
])
def test_io(integrator_instance):
    """Checks each integrator has the desired IO for an integrator"""

    problem = tq.exampleProblems.SimpleGaussian(3)

    res = integrator_instance(problem)
    assert isinstance(res['estimate'], float)
    res = integrator_instance(problem, return_N=True)
    assert isinstance(res['n_evals'], int)
    # res = integrator_instance(problem, return_all=True)
    # assert len(res) >= 2
    # res = integrator_instance(problem, return_N=True, return_all=True)
    # assert len(res) >= 2


########################################################################
# Checking all combos of inputs for simple and queue based integrators
########################################################################

splits = [
    tq.splits.KdSplit(),
    tq.splits.MinSseSplit()
]

integrals = [
    tq.containerIntegration.MedianIntegral(),
    tq.containerIntegration.MidpointIntegral(),
    tq.containerIntegration.RandomIntegral(),
    tq.containerIntegration.SmcIntegral(),
    tq.containerIntegration.RbfIntegral(n_samples=5, n_tuning=1, max_iter=100)
]

queues = [
    tq.queues.ReservoirQueue(),
    tq.queues.PriorityQueue()
]

@pytest.mark.parametrize("D", [1,2,4])
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("P", [100])
@pytest.mark.parametrize("split", splits)
@pytest.mark.parametrize("integral", integrals)
def test_SimpleIntegrator(D, N, P, split, integral):
    problem = tq.exampleProblems.SimpleGaussian(D)
    integ = tq.integrators.SimpleIntegrator(N, P, split, integral)
    res = integ(problem, return_N = True)

@pytest.mark.parametrize("D", [1,2])
@pytest.mark.parametrize("base_N", [1000])
@pytest.mark.parametrize("split", splits)
@pytest.mark.parametrize("integral", integrals)
@pytest.mark.parametrize("weighting_function", [lambda container: container.volume])
@pytest.mark.parametrize("active_N", [0, 10])
@pytest.mark.parametrize("num_splits", [np.inf, 50])
@pytest.mark.parametrize("stopping_condition", [lambda container: container.N < 2])
@pytest.mark.parametrize("queue", queues)
def test_QueueIntegrator(
    D, base_N, split, integral, weighting_function,
    active_N, num_splits, stopping_condition, queue):

    if np.isinf(num_splits) and active_N > 0:
        return

    problem = tq.exampleProblems.SimpleGaussian(D)
    integ = tq.integrators.QueueIntegrator(
        base_N, split, integral, weighting_function,
        active_N, num_splits, stopping_condition, queue)
    res = integ(problem, return_N=True, return_containers=True)
    fcs = res['containers']
    ns = res['n_splits']
    N = res['n_evals']

    assert isinstance(res['estimate'], float)
    assert isinstance(res['n_evals'], int)
    assert isinstance(N, int)
    assert isinstance(ns, int)
    assert isinstance(fcs, list)
    assert all(isinstance(cont, tq.Container) for cont in fcs)

    if "RandomIntegral" in str(integral) or "SmcIntegral" in str(integral): 
        # accounts for random samples used in container integration
        assert base_N + ns*2*active_N + len(fcs)*integral.n == N
    elif "RbfIntegral" in str(integral):
        # accounts for random samples used in container integration
        assert base_N + ns*2*active_N + len(fcs)*integral.n_samples == N    
    else:
        assert base_N + ns*2*active_N == N

    if not np.isinf(num_splits):
        assert ns == num_splits

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

    problem = tq.exampleProblems.SimpleGaussian(D)
    integ = tq.integrators.LimitedSampleIntegrator(
        N, base_N, active_N, split, integral, weighting_function, queue
    )
    res = integ(problem, return_N=True)

    assert isinstance(res['estimate'], float)
    assert isinstance(res['n_evals'], int)