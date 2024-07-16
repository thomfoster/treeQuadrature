import pytest

import treeQuadrature as tq
import numpy as np

from .conftest import container

@pytest.mark.parametrize("D", [1,2,5])
def test_init(D):
    X = np.array([
        [-10]*D,
        [10]*D
    ])
    y = np.array([[0.0]]*2)
    cont = tq.Container(X, y)

    assert not cont.is_finite
    assert np.isnan(cont.midpoint)
    assert cont.D == D

    mins = np.array([-10]*D)
    maxs = np.array([10]*D)
    
    cont = tq.Container(X, y, mins=mins, maxs=maxs)
    assert cont.is_finite
    assert cont.midpoint.ndim == 1
    assert cont.midpoint.shape == (D,)
    assert np.all(cont.midpoint == np.array([0]*D))

    assert cont._X.N == 2
    assert np.all(cont._X.contents == X)

@pytest.mark.parametrize("x, expected", [
    (-50.0, "Fail"),
    (-5.0, "Success"),
    (0.0, "Success"),
    (5.0, "Success"),
    (10.0, "Fail")
])
def test_add_method(container, x, expected):
    D = container.D
    new_X = np.array([[x]*D])
    new_y = np.array([[0.0]]*1)

    # Closed container should reject all but x = 0.0
    if container.name == "closed_container":
        if x == 0.0:
            N = container.N
            container.add(new_X, new_y)
            assert container.N == N + 1
        else:
            try:
                container.add(new_X, new_y)
            except ValueError:
                pass
    
    # The other containers should go as labelled
    else:
        if expected == "Fail":
            try:
                container.add(new_X, new_y)
            except ValueError:
                pass
        else:
            container.add(new_X, new_y)


@pytest.mark.parametrize("n_samples", [0, 1,100])
def test_rvs_method(container, n_samples):
    samples = container.rvs(n_samples)

    # shape test
    assert samples.ndim == 2
    assert samples.shape[0] == n_samples
    assert samples.shape[1] == container.D

    # naive coverage test
    if n_samples == 100:
        range_samples = np.max(samples, axis=0) - np.min(samples, axis=0)
        range_container = container.maxs - container.mins
        assert np.all(range_samples >= 0.8 * range_container)

    # addition test
    y = np.ones(shape=(n_samples, 1))
    container.add(samples, y)  # checks samples inside bounds


@pytest.mark.parametrize("force_test_to_repeat", list(range(10)))
def test_split_method(container, force_test_to_repeat):
    # test split on random point
    for split_dimension in range(0, container.D):
        split_value = container.rvs(1)[0, split_dimension]
        lc, rc = container.split(split_dimension, split_value)
        assert np.isclose(lc.volume + rc.volume, container.volume)
        assert lc.N + rc.N == container.N
        for D in range(container.D):
            if D == split_dimension:
                assert lc.maxs[D] == rc.mins[D] == split_value
            else:
                assert lc.mins[D] == rc.mins[D] == container.mins[D]
                assert lc.maxs[D] == rc.maxs[D] == container.maxs[D]

    # test split on a sample point
    if container.N > 0:
        for split_dimension in range(0, container.D):
            chosen_idx = np.random.choice(list(range(container.N)))
            split_value = container.X[chosen_idx, split_dimension]
            lc, rc = container.split(split_dimension, split_value)
            assert np.isclose(lc.volume + rc.volume, container.volume)
            assert lc.N + rc.N == container.N
            for D in range(container.D):
                if D == split_dimension:
                    assert lc.maxs[D] == rc.mins[D] == split_value
                else:
                    assert lc.mins[D] == rc.mins[D] == container.mins[D]
                    assert lc.maxs[D] == rc.maxs[D] == container.maxs[D]
            


def test_volume_method(container):
    if container.name == "closed_container":
        assert container.volume == 0
    else:
        assert container.volume == 10 ** container.D


@pytest.mark.parametrize("D, mins, maxs", [
    (1, [-np.inf], [np.inf]),
    (1, [-1.0], [np.inf]),
    (1, [-np.inf], [1.0]),
    (2, [-np.inf, -np.inf], [np.inf, np.inf]),
    (2, [-1.0, -np.inf], [1.0, np.inf]),
    (2, [-np.inf, -1.0], [np.inf, 1.0]),
    (2, [-1.0, -1.0], [1.0, 1.0]),
    (3, [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
    (3, [-1.0, -np.inf, -np.inf], [1.0, np.inf, np.inf]),
    (3, [-np.inf, -1.0, -np.inf], [np.inf, 1.0, np.inf]),
    (3, [-np.inf, -np.inf, -1.0], [np.inf, np.inf, 1.0])
])
def test_infinite_container(D, mins, maxs):
    # create empty container
    X = np.empty(shape=(0,D))
    y = np.empty(shape=(0,1))

    cont = tq.container.Container(X, y, mins=mins, maxs=maxs)
    samples = cont.rvs(10)
    assert cont.filter_points(samples, return_bool=True), 'some samples not in the container'