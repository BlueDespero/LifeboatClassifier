from category_encoders import LeaveOneOutEncoder, MEstimateEncoder, BaseNEncoder, OneHotEncoder

from Classification_algorithms.K_Nearest_Neighbors.k_nearest_neighbors import KNN
from utils.data_preprocessing import get_data

train_x, train_y, test, _ = get_data(["../../data/train.csv", "../../data/test.csv"], encoder=LeaveOneOutEncoder,
                                     keep_cols=["Sex", "Age", "Pclass", "Fare"])

knn = KNN()
knn.train(train_x, train_y, k=3, distance="matching")
results = knn.classify(test)

test['Survived'] = results[0]
test['PassengerId'] = range(892, 1310)

test[['PassengerId', 'Survived']].to_csv(
    '../../Classification_algorithms/K_Nearest_Neighbors/predictions/pruned/k_nearest_neighbors.csv',
    index=False)
