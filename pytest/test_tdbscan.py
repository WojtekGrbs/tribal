import numpy as np
from tribal.transductive_clustering import TDBSCAN

def test_single_cluster():
    X = np.array([
        [0, 0],
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1]
    ])
    y = np.array([0, -1, -1, -1, -1])
    tdbscan = TDBSCAN(eps=1.5, k=4)
    assert np.all(tdbscan.fit_predict(X, y) == np.array([0, 0, 0, 0, 0]))

def test_insufficient_radius():
    X = np.array([
        [0, 0],
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1]
    ])
    y = np.array([0, -1, -1, -1, -1])
    tdbscan = TDBSCAN(eps=.5, k=4)  # insufficient radius
    assert np.all(tdbscan.fit_predict(X, y) == np.array([0, -1, -1, -1, -1]))

def test_insufficient_neighbors_count():
    X = np.array([
        [0, 0],
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1],
        [-2, -1]
    ])
    y = np.array([0, -1, -1, -1, -1, -1])
    tdbscan = TDBSCAN(eps=1.5, k=7)  # insufficient neighbor count
    assert np.all(tdbscan.fit_predict(X, y) == np.array([0,0,0,0,0, -1]))


def test_two_clusters():
    # First cluster
    X1 = np.array([
        [0, 0],
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1]
    ])
    X = np.concatenate((X1, X1+np.array([10, 10])))
    y = np.array([0, -1, -1, -1, -1,
                  1, -1, -1, -1, -1])
    tdbscan = TDBSCAN(eps=1.5, k=4)
    assert np.all(tdbscan.fit_predict(X, y) == np.array([0, 0, 0, 0, 0,
                                                 1, 1, 1, 1, 1]))


def test_single_cluster_outlier():
    X = np.array([
        [0, 0],
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1],
        [10, 10]
    ])
    y = np.array([0, -1, -1, -1, -1, -1])
    tdbscan = TDBSCAN(eps=1.5, k=4)
    assert np.all(tdbscan.fit_predict(X, y) == np.array([0, 0, 0, 0, 0, -1]))

def test_new_clusters():
    X1 = np.array([
        [0, 0],
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1]
    ])
    X = np.concatenate((X1, X1+np.array([10, 10])))
    y = np.array([0, -1, -1, -1, -1,
                  -1, -1, -1, -1, -1])
    tdbscan = TDBSCAN(eps=1.5, k=4)
    assert np.all(tdbscan.fit_predict(X, y) == np.array([0, 0, 0, 0, 0,
                                                 -2, -2, -2, -2, -2]))

def test_no_new_clusters():
    X1 = np.array([
        [0, 0],
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1]
    ])
    X = np.concatenate((X1, X1+np.array([10, 10])))
    y = np.array([0, -1, -1, -1, -1,
                  -1, -1, -1, -1, -1])
    tdbscan = TDBSCAN(eps=1.5, k=4, new_clusters=False)
    assert np.all(tdbscan.fit_predict(X, y) == np.array([0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0]))

def test_new_clusters_outliers():
    X1 = np.array([
        [0, 0],
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1],
        [-1000, -1000]  # outlier
    ])
    X = np.concatenate((X1, X1+np.array([10, 10])))
    y = np.array([0, -1, -1, -1, -1, -1,
                  -1, -1, -1, -1, -1, -1])
    tdbscan = TDBSCAN(eps=1.5, k=4)
    assert np.all(tdbscan.fit_predict(X, y) == np.array([0, 0, 0, 0, 0, -1,
                                                 -2, -2, -2, -2, -2, -1]))

# Test no new clusters no matter the shape of the data
def test_multiple_no_new_clusters():
    for _ in range(10):
        X = np.random.random(size=(100, 3))
        y = np.array([0] + [-1 for __ in range(99)])
        tdbscan = TDBSCAN(eps=1.5, k=4, new_clusters=False)
        for ans in tdbscan.fit_predict(X, y):
            assert ans in (0, -1)


def test_multiple_new_clusters():
    X_base = np.array([
        [0, 0],
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1]
    ])
    for i in range(30):
        X_concat_tuple = (X_base + np.array([10*j, 10*j]) for j in range(i+2))
        y_concat_tuple = ([j, -1, -1, -1, -1] for j in range(i+2))
        y_correct_concat_tuple = ([j]*5 for j in range(i+2))
        X = np.concatenate(tuple(X_concat_tuple))
        y = np.concatenate(tuple(y_concat_tuple))
        y_correct = np.concatenate(tuple(y_correct_concat_tuple))

        tdbscan = TDBSCAN(eps=1.5, k=4)
        assert np.all(tdbscan.fit_predict(X, y) == y_correct)

if __name__ == "__main__":
    test_single_cluster()
    test_insufficient_neighbors_count()
    test_two_clusters()
    test_single_cluster_outlier()
    test_new_clusters()
    test_no_new_clusters()
    test_new_clusters_outliers()
    test_multiple_no_new_clusters()
    test_multiple_new_clusters()
