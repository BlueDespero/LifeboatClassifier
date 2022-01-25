import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import datetime
from pathlib import Path

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


def save_results(classifier: AbstractClassifier, **kwargs):
    """
        In kwargs you can set training and test datasets as well as name of file in which
        results will be saved. However, if not specified it will work on default values.
    """

    now = datetime.datetime.now()
    filename = kwargs.get('filename', str(classifier) + '_' + now.strftime("%Y-%m-%d_%H-%M") + '.csv')

    root_abs_path = str(get_project_root())

    test = kwargs.get('test', pd.read_csv(root_abs_path + '/data/test.csv'))
    train_x = kwargs.get('train_x', pd.read_csv(root_abs_path + '/data/train.csv'))
    train_y = kwargs.get('train_y', pd.read_csv(root_abs_path + '/data/train.csv'))

    # Some preprocessing if datasets are default
    if 'train_x' not in kwargs:
        train_x['Name'] = train_x['Name'].apply(lambda full_name: full_name.split(',')[0])
        train_x = train_x.iloc[:, 2:]

    if 'train_y' not in kwargs:
        train_y.rename(columns={"Survived": "target"}, inplace=True)
        train_y = train_y.iloc[:, 1]

    if 'test' not in kwargs:
        test['Name'] = test['Name'].apply(lambda full_name: full_name.split(',')[0])
        test = test.iloc[:, 1:]

    classifier.train(train_x, train_y, **kwargs)
    results = classifier.classify(test, **kwargs)

    df = pd.DataFrame({'PassengerId': np.arange(892, 1310), 'Survived': results})

    df.to_csv(root_abs_path + "\Classification_algorithms\Predictions\\" + filename.replace(' ', '_'), index=False)


def get_project_root() -> Path:
    return Path(__file__).parent.parent
