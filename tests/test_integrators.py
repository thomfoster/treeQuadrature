import pytest
import treeQuadrature as tq
import numpy as np
from functools import partial

######################
# Initial IO checks
######################

@pytest.mark.parametrize("integrator_instance", [
    tq.integrators.SmcIntegrator(100),
    tq.integrators.VegasIntegrator(100, 10),
    tq.integrators.SimpleIntegrator(100, 50, tq.splits.kdSplit, tq.containerIntegration.midpointIntegral),
])
def test_io(integrator_instance):
    """Checks each integrator has the desired IO for an integrator"""

    problem = tq.exampleProblems.SimpleGaussian(3)

    I = integrator_instance(problem)
    I, N = integrator_instance(problem, return_N=True)
    res = integrator_instance(problem, return_all=True)
    assert len(res) >= 2
    res = integrator_instance(problem, return_N=True, return_all=True)
    assert len(res) >= 2


########################################################################
# Checking all combos of inputs for simple and queue based integrators
########################################################################

splits = [
    tq.splits.kdSplit,
    tq.splits.minSseSplit
]

integrals = [
    tq.containerIntegration.medianIntegral,
    tq.containerIntegration.midpointIntegral,
    tq.containerIntegration.randomIntegral,
    tq.containerIntegration.smcIntegral
]

queues = [
    tq.queues.ReservoirQueue,
    tq.queues.PriorityQueue
]

@pytest.mark.parametrize("D", [1,2, 10])
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("P", [100])
@pytest.mark.parametrize("split", splits)
@pytest.mark.parametrize("integral", integrals)
def test_SimpleIntegrator(D, N, P, split, integral):
    problem = tq.exampleProblems.SimpleGaussian(D)
    integ = tq.integrators.SimpleIntegrator(N, P, split, integral)
    I = integ(problem)

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
    I, N, fcs, cs, ns = integ(problem, return_all=True)

    if "randomIntegral" in str(integral): 
        assert base_N + ns*2*active_N + len(fcs)*10 == N
    elif "smcIntegral" in str(integral):
        assert base_N + ns*2*active_N + len(fcs)*10 == N
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
    I, N, fcs, cs, ns = integ(problem, return_all=True)

