from tests.utils import cross_validation
from Classification_algorithms.Naive_Bayes_Classifier.naive_bayes import NaiveBayesClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('../../data/train_num.csv')
test = pd.read_csv('../../data/test_num.csv')

train_x = train.iloc[:, 2:]
train_y = train.iloc[:, 1]

n = 10

error_rates = []
standard_deviations = []

for i in range(1, n + 1):
    result = cross_validation(train_x, train_y, classifier=NaiveBayesClassifier(), folds=10, threshold=i)
    error_rates.append(result['err_rate'])
    standard_deviations.append(result['standard_deviation'])

fig, axs = plt.subplots(2)
fig.suptitle('Naive Bayes Classifier')
axs[0].plot(np.arange(n), error_rates, label='error rate', c='r')
axs[1].plot(np.arange(n), standard_deviations, label='standard deviation')

axs[1].set_xlabel('threshold')

axs[0].legend()
axs[1].legend()

plt.show()
