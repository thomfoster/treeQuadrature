import pytest
import numpy as np
import treeQuadrature as tq

TESTING_DIMS = [1,2,15]

@pytest.fixture(params=TESTING_DIMS)
def simple_container(request):
    """standard populated container"""

    D = request.param
    d = tq.example_problems.distributions.MultivariateNormal(D, mean=[0.0]*D, cov=0.5)
    X = d.rvs(100)
    y = d.pdf(X)
    
    mins = np.array([-5.0]*D)
    maxs = np.array([5.0]*D)

    cont = tq.Container(X, y, mins=mins, maxs=maxs)

    cont.name = "simple_container"
    return cont


@pytest.fixture(params=TESTING_DIMS)
def empty_container(request):
    """container with volume > 0 but no samples"""

    D = request.param
    X = np.empty(shape=(0,D))
    y = np.empty(shape=(0,1))

    mins = np.array([-5.0]*D)
    maxs = np.array([5.0]*D)

    cont = tq.Container(X, y, mins=mins, maxs=maxs)
    cont.name = "empty_container"
    return cont


@pytest.fixture(params=TESTING_DIMS)
def boundary_container(request):
    """container with samples on corners of container"""
    
    D = request.param

    mins = np.array([-5.0]*D)
    maxs = np.array([5.0]*D)

    X = np.array([mins, maxs])
    y = np.ones(shape=(2,1))

    cont = tq.Container(X, y, mins=mins, maxs=maxs)
    cont.name = "boundary_container"
    return cont


@pytest.fixture(params=TESTING_DIMS)
def closed_container(request):
    """container with 0 volume"""

    D = request.param
    X = np.array([[0.0]*D, [0.0]*D])
    y = np.array([[1.0], [1.0]])
    
    mins  = np.array([0.0]*D)
    maxs = np.array([0.0]*D)

    cont = tq.Container(X, y, mins=mins, maxs=maxs)
    cont.name = "closed_container"
    return cont


@pytest.fixture(params=[
    'simple_container', 
    'empty_container',
    'closed_container'
    ])
def container(request, simple_container, empty_container, closed_container, boundary_container):
    """
    Pretty hacky "Meta" container to allow the parametrization of multiple containers.
    See https://github.com/pytest-dev/pytest/issues/349 for more answers.
    """
    switch = {
        "simple_container": simple_container, 
        "empty_container": empty_container,
        "closed_container": closed_container,
        "boundary_container": boundary_container
        }

    return switch[request.param]