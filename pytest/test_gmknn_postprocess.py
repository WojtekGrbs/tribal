from tribal.algorithms.src.gmknn.binaries.gmknn_connect import connect_unlabeled
from tribal.algorithms.src.gmknn.binaries.gmknn_bfs import find_connected_components
from tribal.algorithms.src.gmknn.binaries.mknn import GraphWrapper
from tribal.algorithms.algorithm_input import _AlgorithmInput
from scipy.spatial import KDTree
import numpy as np

def test_connect_unlabeled():    
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

    connections = GraphWrapper(6, 0, 10, lookup_opt=True)  # INT64 connections
    weights = GraphWrapper(6, 1, 10)  # FLOAT32 weights
    setup_connections = [
        (0, 1), (1,0), (1,3), (1,2), (2,1), (3,1), (4,5),
        (5,4)
    ]
    for con in setup_connections:
        connections.append(con[0], con[1])
        weights.append(con[0], np.linalg.norm(X[con[0]]-X[con[1]]))

    _kd_tree = KDTree(X)
    knn_weights_sparse_handler, knn_connections_sparse = _kd_tree.query(X, k=3, workers=-1)
    # Remove self-connections
    knn_weights_sparse = knn_weights_sparse_handler[:, 1:].astype(np.float32)
    knn_connections_sparse = knn_connections_sparse[:, 1:]

    c, cv = find_connected_components(X_in, connections)
    connections_after, _ = connect_unlabeled(knn_connections_sparse, knn_weights_sparse, c, cv, connections, weights)

    setup_solution = [np.array([1,5]), np.array([0,2,3]), np.array([1]),
                                                np.array([1]), np.array([5]), np.array([0,4])]
    assert all(np.array_equal(a, np.sort(b)) for a, b in zip(setup_solution, connections_after.to_numpy_all()))


if __name__ == "__main__":
    test_connect_unlabeled()