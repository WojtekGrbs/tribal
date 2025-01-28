import numpy as np
from tribal.algorithms.src.mstl.binaries.mstl import *
import pytest


def are_results_equal(result, expected_result):
    if len(result) != len(expected_result):
        return False
    for arr1, arr2 in zip(result, expected_result):
        for i in range(len(arr1)):
            elem1 = arr1[i]
            elem2 = arr2[i]
            if not elem1 == pytest.approx(elem2):
                return False
    return True


def test_prim_mst_deterministic():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    expected_connections, expected_weights = prim_mst(X, metric='euclidean')
    np_expected_connections = expected_connections.to_numpy_all()
    np_expected_weights = expected_weights.to_numpy_all()
    for i in range(100):
        connections, weights = prim_mst(X, metric='euclidean')
        np_connections = connections.to_numpy_all()
        np_weights = weights.to_numpy_all()
        assert are_results_equal(np_connections, np_expected_connections)
        assert are_results_equal(np_weights, np_expected_weights)


def test_prim_mst_two_points():
    X = np.array([[0, 0], [1, 1]], dtype=np.float64)
    connections, weights = prim_mst(X, metric='euclidean')

    expected_connections = [[1], [0]]
    expected_weights = [[np.sqrt(2)], [np.sqrt(2)]]

    assert are_results_equal(connections.to_numpy_all(), expected_connections)
    assert are_results_equal(weights.to_numpy_all(), expected_weights)


def test_prim_mst_non_euclidean_metric():
    X = np.array([[0, 0], [1, 1], [3, 0]], dtype=np.float64)
    connections, weights = prim_mst(X, metric='cityblock')

    expected_connections = [[1, 2], [0], [0]]
    expected_weights = [[2, 3], [2], [3]]

    assert are_results_equal(connections.to_numpy_all(), expected_connections)
    assert are_results_equal(weights.to_numpy_all(), expected_weights)


def test_prim_mst_large_graph():
    np.random.seed(123)
    X = np.random.rand(100, 2)
    connections, weights = prim_mst(X, metric='euclidean')
    np_connections = connections.to_numpy_all()
    np_weights = weights.to_numpy_all()
    assert len(np_connections) == 100
    assert len(np_weights) == 100
    for conn, w in zip(np_connections, np_weights):
        assert len(conn) == len(w)


if __name__ == "__main__":
    test_prim_mst_deterministic()
    test_prim_mst_two_points()
    test_prim_mst_non_euclidean_metric()
    test_prim_mst_large_graph()
