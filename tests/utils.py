import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

from Classification_algorithms.common import AbstractClassifier


def cross_validation(x: pd.DataFrame, y: pd.DataFrame, classifier: AbstractClassifier, folds, **kwargs):
    kf = KFold(n_splits=folds)
    err_rate_list = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier.train(x_train, y_train, **kwargs)
        results = classifier.classify(x_test, **kwargs)

        err_rate = 1 - np.sum(results == y_test) / results.size
        err_rate_list.append(err_rate)

    kwargs['err_rate'] = np.mean(err_rate_list).round(3)
    kwargs['standard_deviation'] = np.std(np.array(err_rate_list))
    kwargs['classifier'] = str(classifier)

    return kwargs
