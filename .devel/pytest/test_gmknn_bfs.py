from tribal.algorithms.src.gmknn.binaries.gmknn_bfs import find_connected_components
from tribal.algorithms.src.gmknn.binaries.mknn import GraphWrapper
from tribal.algorithms.algorithm_input import _AlgorithmInput
import numpy as np

def test_bfs_single_node():
    # unlabeled scenario
    X_in = np.array([[0,0]])
    labels = np.array([-1])

    X = _AlgorithmInput(X_in, labels)

    connections = GraphWrapper(1, 0, 1) # empty connections
    result = find_connected_components(X, connections)

    assert result == ([([0], 0)], np.array([0], dtype=np.uint8))

    # labeled scenario
    X_in = np.array([[0,0]])
    labels = np.array([0])

    X = _AlgorithmInput(X_in, labels)

    connections = GraphWrapper(1, 0, 1) # empty connections
    result = find_connected_components(X, connections)

    assert result == ([([0], 1)], np.array([1], dtype=np.uint8))

def test_bfs_label_array():
    X_in = np.random.uniform(0,1,size=(40,1))
    labels = np.concatenate((np.array([0]), np.zeros(38)-1, np.array([0])))

    X = _AlgorithmInput(X_in, labels)

    connections = GraphWrapper(40, 0, 50) # empty connections
    for i in range(18):
        connections.append(0, i+1)
        connections.append(i+1, 0)
    connections.append(39, 38)
    connections.append(38, 39)

    result = find_connected_components(X, connections)
    for comp in result[0]:
        is_labeled = comp[1]
        node_list = comp[0]
        for nde in node_list:
            assert result[1][nde] == is_labeled

def test_bfs_disconnected_basic():
    X_in = np.array([[0,0], [1,1]])
    labels = np.array([-1, 0])

    X = _AlgorithmInput(X_in, labels)

    connections = GraphWrapper(2, 0, 1) # empty connections
    result = find_connected_components(X, connections)

def test_bfs_disconnected():
    X_in = np.random.uniform(0,1,size=(100,1))
    labels = np.concatenate((np.array([0]), np.zeros(98)-1, np.array([0])))

    X = _AlgorithmInput(X_in, labels)

    connections = GraphWrapper(100, 0, 50) # empty connections
    for i in range(49):
        connections.append(0, i+1)
        connections.append(i+1, 0)

        connections.append(99, 98-i)
        connections.append(98-i, 99)
    result = find_connected_components(X, connections)


    assert len(result[0]) == 2 # found two components
    assert result[0][0][1] == result[0][1][1] == 1 # both components are marked as labeled
    assert len(result[0][0]) == len(result[0][1]) # both are of equal size

def test_bfs_isolated():
    X_in = np.random.uniform(0,1,size=(20,1))
    labels = np.concatenate((np.array([0]), np.zeros(19)-1))

    X = _AlgorithmInput(X_in, labels)

    connections = GraphWrapper(20, 0, 50) # empty connections
    for i in range(18):
        connections.append(0, i+1)
        connections.append(i+1, 0)

    result = find_connected_components(X, connections)
    assert len(result[0]) == 2 # found two components
    assert min(len(result[0][0][0]), len(result[0][1][0])) == 1 # smaller of size 1
    assert max(len(result[0][0][0]), len(result[0][1][0])) == 19 # bigger of size 50-1
    
if __name__ == "__main__":
    test_bfs_single_node()
    test_bfs_label_array()
    test_bfs_disconnected_basic()
    test_bfs_disconnected()
    test_bfs_isolated()

