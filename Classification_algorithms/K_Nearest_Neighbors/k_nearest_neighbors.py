import numpy as np
import scipy.stats as sstats
from Classification_algorithms.common import AbstractClassifier


class KNN(AbstractClassifier):

    def __init__(self):
        super().__init__()
        self.k = None

    def __str__(self):
        return "K-Nearest Neighbors"

    def train(self, train_x, train_y, k=5, **kwargs):
        super().train(train_x, train_y)
        self.k = k

    def create_proximity_matrix(self, train, test):
        proximity_matrix = np.array(np.einsum('ij, ij ->i', train, train)[:, None]) + \
                           np.array(np.einsum('ij, ij ->i', test, test)) - np.array(2 * train.dot(test.T))
        return np.argsort(proximity_matrix, axis=0)[:self.k, :]

    def classify(self, x, **kwargs):
        distance_matrix = self.create_proximity_matrix(self.train_x, x)
        targets = np.array(self.train_y)[distance_matrix]

        predictions = sstats.mode(targets).mode
        predictions = predictions.ravel()

        return predictions

