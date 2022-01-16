import numpy as np
import scipy.stats as sstats


def create_proximity_matrix(train, test, k=5):
    proximity_matrix = np.array(np.einsum('ij, ij ->i', train, train)[:, None]) + \
                       np.array(np.einsum('ij, ij ->i', test, test)) - np.array(2 * train.dot(test.T))
    return np.argsort(proximity_matrix, axis=0)[:k, :]


def KNN(train_X, train_Y, test_X, k=5):
    distance_matrix = create_proximity_matrix(train_X, test_X, k=k)
    targets = np.array(train_Y)[distance_matrix]

    predictions = sstats.mode(targets).mode
    predictions = predictions.ravel()

    return predictions
