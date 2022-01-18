from category_encoders import BinaryEncoder

from Classification_algorithms.K_Nearest_Neighbors.k_nearest_neighbors import KNN
from utils.data_preprocessing import get_data

train_x, train_y, test, _ = get_data(["../../data/train.csv", "../../data/test.csv"], encoder=BinaryEncoder)
# train_x, train_y, test, _ = get_data(["../../data/train_num.csv", "../../data/test_num.csv"])

knn = KNN()
knn.train(train_x, train_y, k=10)
results = knn.classify(test)

test['Survived'] = results

test[['PassengerId', 'Survived']].to_csv('../../Classification_algorithms/Predictions/k_nearest_neighbors.csv',
                                         index=False)
