import numpy as np
from sklearn.neighbors import NearestNeighbors

EPS = 1e-5


def cos_metric(a, b):
    return 1 - np.dot(a, b) / np.dot(a, a)**(1/2) / np.dot(b, b)**(1/2)


def euclid_metric(a, b):
    return (np.abs(a - b)**2).sum()**(1/2)


def uniform(u: np.array):
    return (np.float64(0.5) * u) * (np.abs(u) < 1.0)


def triangular(u: np.array):
    return (1 - np.abs(u)) * (np.abs(u) < 1.0)


def epanechnikov(u: np.array):
    return (3 / 4 * (1 - u ** 2)) * (np.abs(u) < 1.0)


def chebyshev_metric(a, b):
    return np.abs(a - b).max()


def gaussian(u: np.array):
    return 1.0 / np.sqrt(2.0 * np.pi) * np.exp((u ** 2) / -2.0)


def _get_metric(metric: str):
    if metric == KNN.CHEBYSHEV_METRIC:
        return chebyshev_metric
    elif metric == KNN.COS_METRIC:
        return cos_metric
    elif metric == KNN.EUCLID_METRIC:
        return euclid_metric
    else:
        raise Exception('Unexpected metric')


class KNN:

    EUCLID_METRIC = 'euclid'
    CHEBYSHEV_METRIC = 'chebyshev'
    COS_METRIC = 'cos'

    def __init__(self,
                 k_neighbors=None,
                 window_size=None,
                 metric="euclid",
                 kernel=uniform):
        self.x = None
        self.y = None
        self.weights = None
        if k_neighbors is None and window_size is None:
            raise Exception('n_neighbors or window_type must not be None')
        self.k_neighbors = k_neighbors
        self.window_size = window_size
        self.probabilities = None
        self.kernel = kernel

        if metric is str:
            metric = _get_metric(metric)
        else:
            metric = metric

        self.nn = NearestNeighbors(metric=metric)

    def fit(self, x, y):
        if len(x) != len(y):
            raise Exception("expected same length")
        if x is not np.array:
            x = np.array(x)
        if y is not np.array:
            y = np.array(y)
        self.x = x
        self.y = y
        self.nn.fit(x)

    def predict(self, x_n, weights=None):
        return list(map(lambda x: self.predict_one(x, weights), x_n))

    def predict_one(self, x_n, weights=None):
        if weights is None:
            self.weights = np.ones(len(self.x))
        else:
            self.weights = weights

        x_n = [x_n]
        if self.k_neighbors is not None:
            n_distances, n_indexes = self.nn.kneighbors(x_n, self.k_neighbors + 1, return_distance=True)
            indexes = n_indexes[0][:-1]
            max_distance = n_distances[0][-1]
            distances = n_distances[0][:-1]
        else:
            max_distance = self.window_size
            n_distances, n_indexes = self.nn.radius_neighbors(x_n, self.window_size, return_distance=True)
            distances = n_distances[0]
            if len(distances) == 0:
                raise Exception("alone in the radius :(")
            indexes = n_indexes[0]
        norm_distances = distances / (max_distance + EPS)
        f_distances = self.kernel(norm_distances) * self.weights[indexes]
        y_n = self.y[indexes]
        sum_distance = sum(f_distances)
        y_distances = {}
        for i in range(len(f_distances)):
            clazz = y_n[i]
            if clazz not in y_distances:
                y_distances[clazz] = 0
            y_distances[clazz] += f_distances[i]
        probabilities = {k: v / (sum_distance + EPS) for k, v in y_distances.items()}
        self.probabilities = probabilities
        return sorted(probabilities.items(), key=lambda v: v[1], reverse=True)[0][0]


if __name__ == '__main__':
    pass
