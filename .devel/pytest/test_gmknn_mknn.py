import numpy as np
from scipy.spatial import distance
from tribal.algorithms.src.gmknn.binaries.mknn import GraphWrapper, mutual_knn, informative_edges
import pytest
from tribal.algorithms.algorithm_input import _AlgorithmInput

def test_mutual_knn():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [9.0, 9.0],
        [10.0, 10.0]
    ])
    k = 2
    mutual_connections, mutual_weights, _, __ = mutual_knn(X, k)
    # Check mutual neighbors
    for i, conn in enumerate(mutual_connections.to_numpy_all()):
        assert len(conn) <= k, \
            f"Vertex {i} should have at most {k} mutual neighbors."

    # Check weights consistency
    dist = distance.cdist(X, X)
    for i, conn in enumerate(mutual_connections.to_numpy_all()):
        for j, neighbor in enumerate(conn):
            expected_weight = dist[i, neighbor]
            actual_weight = mutual_weights.to_numpy(i)[j]
            assert expected_weight == pytest.approx(actual_weight), \
                f"Weight between {i} and {neighbor} should be {expected_weight}, but got {actual_weight}."
    setup_solution = [np.array([1]), np.array([0,2]), np.array([1,3]), np.array([2]), np.array([5]), np.array([4])]
    assert all(np.array_equal(a, np.sort(b)) for a, b in zip(setup_solution, mutual_connections.to_numpy_all()))


def test_informative_edges_correctness():
    X = np.array([
        [0.5, 0.0],
        [0.5, 0.5],
        [0.5, 1],
        [0, 0.5],
        [1, 0.5],
        [0.75, 0.5]
    ])
    labels = np.array([-1,-1,-1,-1,-1,0])
    X_in = _AlgorithmInput(X, labels)

    connections = GraphWrapper(10, 0, 10)  # INT64 connections
    weights = GraphWrapper(10, 1, 10)  # FLOAT32 weights
    setup_connections = [
        (0, 1), (0, 4), (1,0), (1,3), (1,5), (1,2), (2,1), (2,3), (3,1), (3,2), (4,5), (4,0),
        (5,4), (5,1)
    ]
    for con in setup_connections:
        connections.append(con[0], con[1])
        weights.append(con[0], np.linalg.norm(X[con[0]]-X[con[1]]))
    connections_after, _ = informative_edges(X_in, 3, connections, weights)
    setup_solution = [np.array([1]), np.array([0,2,3,5]), np.array([1]),
                                                 np.array([1]), np.array([5]), np.array([1,4])]
    assert all(np.array_equal(a, np.sort(b)) for a, b in zip(setup_solution, connections_after.to_numpy_all()))
    


if __name__ == "__main__":
    test_mutual_knn()
    test_informative_edges_correctness()