from category_encoders import BinaryEncoder, OneHotEncoder

from Classification_algorithms.K_Nearest_Neighbors.k_nearest_neighbors import KNN
from utils.data_preprocessing import get_data

train_x, train_y, test, _ = get_data(["../../data/train.csv", "../../data/test.csv"], encoder=OneHotEncoder)
# train_x, train_y, test, _ = get_data(["../../data/train_num.csv", "../../data/test_num.csv"])

knn = KNN()
knn.train(train_x, train_y, k=16, distance="matching")
results = knn.classify(test)

test['Survived'] = results[0]

test[['PassengerId', 'Survived']].to_csv(
    '../../Classification_algorithms/K_Nearest_Neighbors/predictions/k_nearest_neighbors.csv',
    index=False)
