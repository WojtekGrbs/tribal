import numpy as np
from tribal.algorithms.src.mstl.binaries.mstl import *
import pytest


def initialize_graph_wrapper(data_sparse, count, data_type_flg, max_size):
    graph_wrapper = GraphWrapper(count, data_type_flg, max_size)
    for i in range(len(data_sparse)):
        data_array = data_sparse[i]
        for j in range(len(data_array)):
            graph_wrapper.append(i, data_array[j])
    return graph_wrapper


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


def test_find_all_paths_with_labels():
    connections_sparse = [
        [1],
        [0, 2, 3],
        [1],
        [1]
    ]

    mst_connections = initialize_graph_wrapper(connections_sparse, 4, 0, 3)
    labels = np.array([1, -1, -1, 2], dtype=np.int8)
    start = 0

    result = find_all_paths_with_labels(mst_connections, labels, start)
    expected = [[0, 1, 3]]
    assert result == expected


def test_find_subgraph_with_labels():
    connections_sparse = [
        [1],
        [0, 2, 3],
        [1],
        [1, 4, 5],
        [3],
        [3, 6],
        [5]
    ]

    weights_sparse = [
        [1],
        [1, 1, 1],
        [1],
        [1, 1, 1],
        [1],
        [1, 1],
        [1]
    ]

    mst_connections = initialize_graph_wrapper(connections_sparse, 7, 0, 3)
    mst_weights = initialize_graph_wrapper(weights_sparse, 7, 1, 3)
    labels = np.array([1, -1, -1, 2, -1, 2, 3], dtype=np.int8)

    conections, weights = find_subgraph_with_labels(mst_connections, mst_weights, labels)
    expected_connections = [
        [1],
        [0, 3],
        [],
        [1],
        [],
        [6],
        [5]
    ]

    expected_weights = [
        [1],
        [1, 1],
        [],
        [1],
        [],
        [1],
        [1]
    ]

    assert are_results_equal(conections.to_numpy_all(), expected_connections)
    assert are_results_equal(weights.to_numpy_all(), expected_weights)

def test_pop_max():

    weights_sparse = [
        [0.67856],
        [0.67856, 0.45631, 1.2345],
        [0.45631],
        [1.2345, 0.76834],
        [0.76834]
    ]

    mst_weights = initialize_graph_wrapper(weights_sparse, 5, 1, 3)

    first_idx, snd_idx = mst_weights.pop_max()

    assert first_idx == 1
    assert snd_idx == 3

def test_make_labels():

    connections_sparse = [
        [1],
        [0, 2, 3],
        [1],
        [1, 4],
        [3]
    ]
    labels = np.array([1, -1, -1, -1, -1], dtype=np.int8)
    visited = set()
    start_node = 0
    label = 1

    connections_sparse = initialize_graph_wrapper(connections_sparse, 5, 0, 3)
    make_labels(connections_sparse, start_node, label, visited, labels)
    expected_labels = np.array([1, 1, 1, 1, 1])

    assert np.array_equal(labels, expected_labels)



if __name__ == "__main__":
    test_find_all_paths_with_labels()
    test_find_subgraph_with_labels()
    test_pop_max()
    test_make_labels()
