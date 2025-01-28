import numpy as np
from tribal.transductive_clustering import mstl, gmknn, TDBSCAN
from sklearn.datasets import make_circles

def test_mstl_one_point():
    X = np.asmatrix([[0, 0, 0], [1, 1, 1]])
    y = [1, -1]
    mstl_model = mstl()
    y_predict = mstl_model.fit_predict(X, y)
    assert np.array_equal(y_predict, [1, 1])


def test_gmknn_one_point():
    X = np.asmatrix([[0, 0, 0], [1, 1, 1]])
    y = [1, -1]
    gmknn_model = gmknn(k=1)
    y_predict = gmknn_model.fit_predict(X, y)
    assert np.array_equal(y_predict, [1, 1])


def test_tdbscan_one_point():
    X = np.asmatrix([[0, 0, 0], [1, 1, 1]])
    y = [1, -1]
    tdbscan_model = TDBSCAN(eps=2, k = 1)
    y_predict = tdbscan_model.fit_predict(X, y)
    assert np.array_equal(y_predict, [1, 1])


def generate_circles():
    X, labels = make_circles(n_samples=1000, factor=0.5, noise=0.03)
    sorted_indices = np.argsort(labels)
    X = X[sorted_indices]
    labels = labels[sorted_indices]
    y = np.full(1000, -1)
    y[250] = 0
    y[750] = 1

    return X, y, labels


def test_mstl_circles():
    X, y, labels = generate_circles()
    mstl_model = mstl()
    y_predict = mstl_model.fit_predict(X, y)
    assert np.array_equal(y_predict, labels)


def test_gmknn_circles():
    X, y, labels = generate_circles()
    gmknn_model = gmknn(k=10)
    y_predict = gmknn_model.fit_predict(X, y)
    assert np.array_equal(y_predict, labels)


def test_tdbscan_circles():
    X, y, labels = generate_circles()
    tdbscan_model = TDBSCAN(eps = 0.1, k = 5)
    y_predict = tdbscan_model.fit_predict(X, y)
    assert np.array_equal(y_predict, labels)


def generate_square():
    num_points_per_cluster = 250
    total_points = num_points_per_cluster * 4
    cluster_centers = [(0, 0), (0, 10), (10, 0), (10, 10)]
    std_dev = 0.5

    data = []
    for i, center in enumerate(cluster_centers):
        x_center, y_center = center
        x_points = np.random.normal(loc=x_center, scale=std_dev, size=num_points_per_cluster)
        y_points = np.random.normal(loc=y_center, scale=std_dev, size=num_points_per_cluster)
        cluster_data = np.column_stack((x_points, y_points))
        data.append(cluster_data)

    X = np.vstack(data)
    y = np.full(total_points, -1)

    for i in range(4):
        cluster_start = i * num_points_per_cluster
        cluster_end = (i + 1) * num_points_per_cluster
        random_index = np.random.randint(cluster_start, cluster_end)
        y[random_index] = i + 1
    labels = np.repeat([1, 2, 3, 4], num_points_per_cluster)

    return X, y, labels

def test_mstl_square():
    X, y, labels = generate_square()
    mstl_model = mstl()
    y_predict = mstl_model.fit_predict(X, y)

    assert np.array_equal(y_predict, labels)

def test_gmknn_square():
    X, y, labels = generate_square()
    gmknn_model = gmknn(k=10)
    y_predict = gmknn_model.fit_predict(X, y)

    assert np.array_equal(y_predict, labels)

def test_tdbscan_square():
    X, y, labels = generate_square()
    tdbscan_model = TDBSCAN(eps = 1, k = 5)
    y_predict = tdbscan_model.fit_predict(X, y)

    assert np.array_equal(y_predict, labels)


if __name__ == "__main__":
    test_mstl_one_point()
    test_gmknn_one_point()
    test_tdbscan_one_point()
    test_mstl_circles()
    test_gmknn_circles()
    test_tdbscan_circles()
    test_mstl_square()
    test_gmknn_square()
    test_tdbscan_square()
