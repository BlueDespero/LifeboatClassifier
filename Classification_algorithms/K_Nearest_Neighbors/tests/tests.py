import pandas as pd

from Classification_algorithms.K_Nearest_Neighbors.k_nearest_neighbors import KNN

train = pd.read_csv("../../../data/train_num.csv")
test = pd.read_csv("../../../data/test_num.csv")

train_x = train.iloc[:, 2:]
train_y = train.iloc[:, 1]

results = KNN(train_x, train_y, test.iloc[:, 1:], k=10)

test.insert(1, column='Survived', value=results)

test.iloc[:, :2].to_csv('../../Predictions/k_nearest_neighbors.csv', index=False)
