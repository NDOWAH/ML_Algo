import numpy as np
from numpy import sqrt, sum
from collections import Counter


def euclidean(x, y):
    return sqrt(sum((x-y)**2))


class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, y_train):
        y_prediction = [self._predict(y_train) for x in self.y_train]
        return y_prediction

    def _predict(self, x):
        distance = [euclidean(x, X_train) for X_train in self.X_train]
        nearest_distance = np.argsort(distance)[:self.k]
        nearest_label = [self.y_train[i] for i in nearest_distance]
        voted_class = Counter(nearest_label).most_common(1)
        return voted_class[0][0]



