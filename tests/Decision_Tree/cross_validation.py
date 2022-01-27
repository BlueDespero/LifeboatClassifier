from tests.utils import cross_validation
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from Classification_algorithms.Decision_Tree.decision_tree import DecisionTree

train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data/test.csv')

train['Name'] = train['Name'].apply(lambda full_name: full_name.split(',')[0])
train.rename(columns={"Survived": "target"}, inplace=True)

train_x = train.iloc[:, 2:]
train_y = train.iloc[:, 1]

k = 5

error_rates = []
standard_deviations = []

criterions = ["infogain", "infogain_ratio", "mean_err_rate", "gini"]
for criterion in tqdm(criterions):
    result = cross_validation(train_x, train_y, classifier=DecisionTree(), folds=k, criterion=criterion)
    error_rates.append(result['err_rate'])
    standard_deviations.append(result['standard_deviation'])

fig, axs = plt.subplots(2)
fig.suptitle('Naive Bayes Classifier')
axs[0].bar(criterions, error_rates, color='r')
axs[0].set_ylabel('error rate')
axs[1].bar(criterions, standard_deviations)
axs[1].set_ylabel('standard deviation')

plt.show()
